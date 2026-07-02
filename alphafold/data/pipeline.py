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

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.data.tools import jackhmmer
from alphafold.data.tools import mmseqs2
import pickle
import numpy as np
import json


FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]

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
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(msas: Sequence[parsers.Msa], is_mmseqs: bool = False) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index], is_mmseqs)
      species_ids.append(identifiers.species_id.encode('utf-8'))

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    if max_sto_sequences is not None:
      result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
    else:
      result = msa_runner.query(input_fasta_path)[0]
    with open(msa_out_path, 'w') as f:
      f.write(result[msa_format])
  else:
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences is not None:
      precomputed_msa = parsers.truncate_stockholm_msa(
          msa_out_path, max_sto_sequences)
      result = {'sto': precomputed_msa}
    elif max_sto_sequences is not None:
      with open(msa_out_path, 'r') as f:
        result = {msa_format: "\n".join(f.read().split("\n")[:max_sto_sequences*2])}
    else:
      with open(msa_out_path, 'r') as f:
        result = {msa_format: f.read()}
  return result

def parse_restraints(restraint_file: str):
  with open(restraint_file) as rf:
    restraints = [line.strip().split("\t")[:5] for line in rf]

    for res in restraints:
      if "," in res[-1]:
        res[-1] = res[-1].split(",")
      else:
        res[-1] = [res[-1]]
  return restraints

def parse_noe_restraints(restraint_file: str, allowed_types = ["HB", "HB2", "HB3", "CB"]):
  """
  2 PHE  HA      3 ARG  H       3.50  #peak 27
  2 PHE  QB      2 PHE  QD      5.00  #peak 28
  ...
  """
  restraints = []
  with open(restraint_file) as rf:
    for line in rf:
      res1 = line[:3]
      res2 = line[15:18]
      max_d = line[31:36]
      atm1 = line[9:13].strip()
      atm2 = line[24:28].strip()
      #print(line.rstrip())
      if atm1 in allowed_types and atm2 in allowed_types:
        restraints.append((res1, res2, max_d))
        print("accepted", res1, res2, max_d)
  return restraints

class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               mmseqs2_binary_path: Optional [str],
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               mmseqs2_uniref_database_path: Optional[str],
               mmseqs2_env_database_path: Optional[str],
               uniref30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               bfd_max_hits: int = 10000,
               mmseqs2_max_hits: int = 10000,
               use_precomputed_msas: bool = False,
               restraint_file: str = None,
               approximate_restraint: bool = False):
    """Initializes the data pipeline."""
    self._use_small_bfd = use_small_bfd
    if mmseqs2_binary_path:
      self.mmseqs2_runner = mmseqs2.MMseqs2(
          binary_path=mmseqs2_binary_path,
          uniref_db=mmseqs2_uniref_database_path,
          metagenomic_db=mmseqs2_env_database_path,
          gpu = ("gpu" in mmseqs2_uniref_database_path),
          )
    else:
      self.mmseqs2_runner = None
      self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=uniref90_database_path)
      if use_small_bfd:
        self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=small_bfd_database_path)
      else:
        self.hhblits_bfd_uniref_runner = hhblits.HHBlits(
            binary_path=hhblits_binary_path,
            databases=[bfd_database_path, uniref30_database_path])
      self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=mgnify_database_path)
    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.bfd_max_hits = bfd_max_hits
    self.mmseqs2_max_hits = mmseqs2_max_hits
    self.use_precomputed_msas = use_precomputed_msas
    self.restraint_file = restraint_file
    self.approximate_restraint = approximate_restraint

  def process(self, input_fasta_path: str, msa_output_dir: str, chain_id=None, resume_train: str = None) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    uniref90_msa = None
    mgnify_msa = None
    mmseqs2_msa = None

    if self.mmseqs2_runner:
      mmseqs2_out_path = os.path.join(msa_output_dir, 'mmseqs2_hits.a3m')
      mmseqs2_result = run_msa_tool(
        msa_runner=self.mmseqs2_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=mmseqs2_out_path,
        msa_format='a3m',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.mmseqs2_max_hits)
      mmseqs2_msa = parsers.parse_a3m(mmseqs2_result['a3m'])
    else:
      uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
      jackhmmer_uniref90_result = run_msa_tool(
          msa_runner=self.jackhmmer_uniref90_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=uniref90_out_path,
          msa_format='sto',
          use_precomputed_msas=self.use_precomputed_msas,
          max_sto_sequences=self.uniref_max_hits)
      uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
      logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))

      mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
      jackhmmer_mgnify_result = run_msa_tool(
          msa_runner=self.jackhmmer_mgnify_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=mgnify_out_path,
          msa_format='sto',
          use_precomputed_msas=self.use_precomputed_msas,
          max_sto_sequences=self.mgnify_max_hits)
      mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
      logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    # BFD
      if self._use_small_bfd:
        bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
        jackhmmer_small_bfd_result = run_msa_tool(
            msa_runner=self.jackhmmer_small_bfd_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=bfd_out_path,
            msa_format='sto',
            use_precomputed_msas=self.use_precomputed_msas,
            max_sto_sequences=self.bfd_max_hits)
        bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
      else:
        bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniref_hits.a3m')
        hhblits_bfd_uniref_result = run_msa_tool(
            msa_runner=self.hhblits_bfd_uniref_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=bfd_out_path,
            msa_format='a3m',
            use_precomputed_msas=self.use_precomputed_msas,
            max_sto_sequences=self.bfd_max_hits)
        bfd_msa = parsers.parse_a3m(hhblits_bfd_uniref_result['a3m'])
      logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    # TEMPLATES
    """
    if mmseqs2_msa:
      msa_for_templates = mmseqs2_result['a3m']
    else:
      msa_for_templates = jackhmmer_uniref90_result['sto']
      msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
      msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
          msa_for_templates)

    pdb_hits_out_path = os.path.join(
        msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
    if not self.use_precomputed_msas or not os.path.isfile(pdb_hits_out_path):
      if self.template_searcher.input_format == 'sto':
        pdb_templates_result = self.template_searcher.query(msa_for_templates, chain_id, actually_an_a3m=(self.mmseqs2_runner is not None))
      elif self.template_searcher.input_format == 'a3m':
        uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates) if not self.mmseqs2_runner else msa_for_templates
        pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m, chain_id)
      else:
        raise ValueError('Unrecognized template input format: '
                        f'{self.template_searcher.input_format}')

      with open(pdb_hits_out_path, 'w') as f:
        f.write(pdb_templates_result)
    else: # read a pre-existing pdb_hits.sto file
      with open(pdb_hits_out_path) as f:
        pdb_templates_result = f.read()
    """
    pdb_template_hits = []

    templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        hits=pdb_template_hits)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)
    if mmseqs2_msa:
      msas = [mmseqs2_msa]
      logging.info('MMseqs2 MSA size: %d sequences.', len(mmseqs2_msa))
    else:
      msas = (uniref90_msa, bfd_msa, mgnify_msa)
    msa_features = make_msa_features(tuple(msas), is_mmseqs=(self.mmseqs2_runner is not None))

    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_result.features['template_domain_names'].shape[0])
    
    if self.restraint_file is not None:
      lower_breaks = np.linspace(2.3125, 21.6875, 64)
      lower_breaks = np.square(lower_breaks)
      upper_breaks = np.concatenate([lower_breaks[1:],
                                  np.array([1e8], dtype=np.float32)], axis=-1)
      # a list of restraints
      restraints = parse_restraints(self.restraint_file)
      res_dgram = np.zeros((sequence_features["aatype"].shape[0], sequence_features["aatype"].shape[0], 64))

      for i, res in enumerate(restraints):
        if not res:
          continue
        print(res)
        # A 10 A 100 8.0
        _, res1, _, res2, d = res
        res_idx1 = int(res1) - 1
        res_idx2 = int(res2) - 1

        if len(d) == 1:
          dist2 = float(d[0]) ** 2
          res_dgram[res_idx1, res_idx2] = res_dgram[res_idx2, res_idx1] = (dist2 > lower_breaks) if self.approximate_restraint else (dist2 > lower_breaks) * (dist2 < upper_breaks)
          sequence_features["restraints_are_distributions"] = False
        else:
          assert len(d) == 64, f"ERROR: restraints need to be either a single float or a comma-separated list of 64 floats {d}"
          d = np.array(d).astype("float")
          res_dgram[res_idx1, res_idx2] = res_dgram[res_idx2, res_idx1] = d

          sequence_features["restraints_are_distributions"] = True

      sequence_features["restraints_dgram"] = res_dgram

    sequence_features["restraints_pair_bias"] = np.zeros((sequence_features["aatype"].shape[0],
                                                          sequence_features["aatype"].shape[0],
                                                          128))
    if resume_train:
      logging.info(f"Loading previous checkpoint: {resume_train}")
      try:
        sequence_features["restraints_pair_bias"] = pickle.load(open(resume_train, "rb"))["restraints_pair_bias"]
      except:
        logging.warning(f"Couldn't load previos checkpoint, re-initializing instead...")

    return {**sequence_features, **msa_features, **templates_result.features}
