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

"""Functions for building the input features for the AlphaFold model.

Only the MSA-only path is kept: yunta supplies its own (paired) MSAs and never
searches for templates, so the template featurizer, mmCIF parsing and the
HH-suite/kalign wrappers are not needed.
"""

from typing import Mapping, Sequence
import logging

import numpy as np

from .. import residue_constants
from . import parsers

FeatureDict = Mapping[str, np.ndarray]

TEMPLATE_FEATURES = {
      'template_aatype': np.float32,
      'template_all_atom_masks': np.float32,
      'template_all_atom_positions': np.float32,
      'template_domain_names': object,
      'template_sequence': object,
      'template_sum_probs': np.float32}

_INVALID = 255
_AA_TO_ID = np.full(256, _INVALID, dtype=np.uint8)
for _aa, _i in residue_constants.HHBLITS_AA_TO_ID.items():
  _AA_TO_ID[ord(_aa)] = _i


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=object)
  features['residue_index'] = np.arange(num_res, dtype=np.int32)
  features['seq_length'] = np.full((num_res,), num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=object)
  return features


def _encode(sequences: Sequence[str], num_res: int) -> np.ndarray:
  """Integer-encodes equal-length aligned sequences in one vectorised pass."""
  if not sequences:
    return np.zeros((0, num_res), dtype=np.int32)
  raw = np.frombuffer(''.join(sequences).encode('latin-1'), dtype=np.uint8)
  ids = _AA_TO_ID[raw]
  if (ids == _INVALID).any():
    bad = chr(raw[np.flatnonzero(ids == _INVALID)[0]])
    raise KeyError(bad)  # same exception type as the former dict lookup
  return ids.reshape(len(sequences), num_res).astype(np.int32)


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  kept = []
  deletion_matrix = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      kept.append(sequence)
      deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

  num_res = len(msas[0][0])
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = _encode(kept, num_res)
  features['num_alignments'] = np.full((num_res,), len(kept), dtype=np.int32)
  return features


class FoldDataPipeline:
  """Assembles the input features from supplied MSAs (no search, no templates)."""

  def process(self, input_sequence: str,
              input_description: str,
              input_msas: list) -> FeatureDict:
    """Creates features for ``input_sequence`` from pre-built MSAs."""
    num_res = len(input_sequence)

    parsed_msas = []
    parsed_delmat = []
    for custom_msa in input_msas:
      parsed_msa, parsed_deletion_matrix = parsers.parse_a3m(custom_msa)
      parsed_msas.append(parsed_msa)
      parsed_delmat.append(parsed_deletion_matrix)

    template_features = {
        name: np.array([], dtype=dtype)
        for name, dtype in TEMPLATE_FEATURES.items()}

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)
    msa_features = make_msa_features(
        msas=parsed_msas, deletion_matrices=parsed_delmat)

    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    return {**sequence_features, **msa_features, **template_features}
