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

"""Parsing of the pre-aligned (A3M-like) MSAs that yunta supplies."""

import string
from typing import List, Sequence, Tuple

DeletionMatrix = Sequence[Sequence[int]]

_DELETION_TABLE = str.maketrans('', '', string.ascii_lowercase)


def parse_a3m(sequences: Sequence[str]) -> Tuple[List[str], DeletionMatrix]:
  """Parses sequences and the deletion matrix from a list of A3M rows.

  Args:
    sequences: A3M rows (not a file string); the first is the query sequence.
      Lower-case letters are insertions relative to the query.

  Returns:
    A tuple of:
      * The sequences with insertions removed (they may contain duplicates).
      * The deletion matrix: ``deletion_matrix[i][j]`` is the number of residues
        deleted from sequence ``i`` immediately before aligned position ``j``.

  The paired MSAs yunta builds have no insertions, so rows without lower-case
  letters skip the per-character loop entirely.
  """
  aligned_sequences = []
  deletion_matrix = []
  for msa_sequence in sequences:
    stripped = msa_sequence.translate(_DELETION_TABLE)
    aligned_sequences.append(stripped)
    if len(stripped) == len(msa_sequence):  # no insertions
      deletion_matrix.append([0] * len(stripped))
      continue
    deletion_vec = []
    deletion_count = 0
    for j in msa_sequence:
      if j.islower():
        deletion_count += 1
      else:
        deletion_vec.append(deletion_count)
        deletion_count = 0
    deletion_matrix.append(deletion_vec)
  return aligned_sequences, deletion_matrix
