# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for constructing the model."""
import copy
from typing import Any, Mapping, Optional, Union

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf

from .. import confidence
from . import features, modules
from .. import residue_constants


def get_confidence_metrics(
    prediction_result: Mapping[str, Any]) -> Mapping[str, Any]:
  """Post processes prediction_result to get confidence metrics."""

  confidence_metrics = {}
  confidence_metrics['plddt'] = confidence.compute_plddt(
      prediction_result['predicted_lddt']['logits'])
  if 'predicted_aligned_error' in prediction_result:
    confidence_metrics.update(confidence.compute_predicted_aligned_error(
        prediction_result['predicted_aligned_error']['logits'],
        prediction_result['predicted_aligned_error']['breaks']))
    confidence_metrics['ptm'] = confidence.predicted_tm_score(
        prediction_result['predicted_aligned_error']['logits'],
        prediction_result['predicted_aligned_error']['breaks'])

  return confidence_metrics


class RunModel:
  """Container for JAX model."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               params: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None):
    self.config = config
    self.params = params

    def _forward_fn(batch):
      model = modules.AlphaFold(self.config.model)
      return model(
          batch,
          is_training=False,
          ensemble_representations=True)

    transformed = hk.transform(_forward_fn)
    self.apply = jax.jit(transformed.apply)
    self.init = jax.jit(transformed.init)
    # Opt-in early stopping of recycling (see `predict`). None keeps the stock in-graph recycling loop.
    self.recycle_tolerance: Optional[float] = None
    self.last_num_passes: Optional[int] = None
    self._step = None

  def init_params(self, feat: features.FeatureDict, random_seed: int = 0):
    """Initializes the model parameters.

    If none were provided when this class was instantiated then the parameters
    are randomly initialized.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.
      random_seed: A random seed to use to initialize the parameters if none
        were set when this class was initialized.
    """
    if not self.params:
      # Init params randomly.
      rng = jax.random.PRNGKey(random_seed)
      self.params = hk.data_structures.to_mutable_dict(
          self.init(rng, feat))
      logging.warning('Initialized parameters randomly')

  def process_features(
      self,
      raw_features: Union[tf.train.Example, features.FeatureDict],
      random_seed: int) -> features.FeatureDict:
    """Processes features to prepare for feeding them into the model.

    Args:
      raw_features: The output of the data pipeline either as a dict of NumPy
        arrays or as a tf.train.Example.
      random_seed: The random seed to use when processing the features.

    Returns:
      A dict of NumPy feature arrays suitable for feeding into the model.
    """
    if isinstance(raw_features, dict):
      return features.np_example_to_features(
          np_example=raw_features,
          config=self.config,
          random_seed=random_seed)
    else:
      return features.tf_example_to_features(
          tf_example=raw_features,
          config=self.config,
          random_seed=random_seed)

  def _single_pass(self):
    """One compiled AlphaFold pass (no in-graph recycling) that also returns the recycled inputs."""
    if self._step is None:
      config = copy.deepcopy(self.config)
      config.model.num_recycle = 0

      def _forward(batch):
        return modules.AlphaFold(config.model)(
            batch, is_training=False, ensemble_representations=True,
            return_representations=True)

      self._step = jax.jit(hk.transform(_forward).apply)
    return self._step

  def _predict_early_stop(self, feat: features.FeatureDict, tolerance: float):
    """Recycling driven from Python so it can stop once the structure has converged.

    Runs the same passes as the in-graph loop, feeding each pass's positions, first MSA row and pair
    representation into the next, and stops when the RMS change of the C-alpha distance matrix between
    two successive passes falls below `tolerance` (Angstrom). With a tolerance of 0 it runs every
    recycle and reproduces the stock result. The distance-matrix change is the only value copied to
    the host per pass.
    """
    num_recycle = int(self.config.model.num_recycle)
    step = self._single_pass()
    batch_size, num_residues = feat['aatype'].shape
    resample = bool(self.config.model.resample_msa_in_recycling)
    num_ensemble = batch_size // (num_recycle + 1) if resample else batch_size
    emb = self.config.model.embeddings_and_evoformer
    prev = {
        'prev_pos': jnp.zeros([num_residues, residue_constants.atom_type_num, 3]),
        'prev_msa_first_row': jnp.zeros([num_residues, emb.msa_channel]),
        'prev_pair': jnp.zeros([num_residues, num_residues, emb.pair_channel]),
    }
    rng = jax.random.PRNGKey(0)
    last_dist, result, passes = None, None, 0
    for k in range(num_recycle + 1):
      window = slice(k * num_ensemble, (k + 1) * num_ensemble) if resample else slice(None)
      batch = {key: value[window] for key, value in feat.items()}
      for key, value in prev.items():  # recycled inputs enter as per-ensemble-member features
        batch[key] = jnp.broadcast_to(value, (num_ensemble,) + tuple(value.shape))
      result = step(self.params, rng, batch)
      passes += 1
      positions = result['structure_module']['final_atom_positions']
      prev = {
          'prev_pos': positions,
          'prev_msa_first_row': result['representations']['msa_first_row'],
          'prev_pair': result['representations']['pair'],
      }
      ca = positions[:, residue_constants.atom_order['CA']]
      dist = jnp.linalg.norm(ca[:, None] - ca[None], axis=-1)
      if last_dist is not None and float(jnp.sqrt(jnp.mean((dist - last_dist) ** 2))) < tolerance:
        break
      last_dist = dist
    self.last_num_passes = passes
    result = dict(result)
    del result['representations']
    return result

  def predict(self, feat: features.FeatureDict) -> Mapping[str, Any]:
    """Makes a prediction by inferencing the model on the provided features.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.

    Returns:
      A dictionary of model outputs.

    If `recycle_tolerance` is set (and the model recycles), recycling stops early once the structure
    has converged (see `_predict_early_stop`); otherwise the stock in-graph recycling loop runs.
    """
    self.init_params(feat)
    if self.recycle_tolerance is not None and self.config.model.num_recycle > 0:
      result = self._predict_early_stop(feat, self.recycle_tolerance)
    else:
      self.last_num_passes = int(self.config.model.num_recycle) + 1
      result = self.apply(self.params, jax.random.PRNGKey(0), feat)
    # get_confidence_metrics reads the logits on the host, which synchronises
    # the device; no separate block_until_ready sweep over every output is needed.
    result.update(get_confidence_metrics(result))
    return result
