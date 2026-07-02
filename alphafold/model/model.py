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
from typing import Any, Mapping, Optional, Union

from absl import logging
from alphafold.common import confidence
from alphafold.model import features
from alphafold.model import modules
from alphafold.model import modules_multimer
import haiku as hk
import jax
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf
import tree
import optax
import jax.numpy as jnp

def get_confidence_metrics(
    prediction_result: Mapping[str, Any],
    multimer_mode: bool) -> Mapping[str, Any]:
  """Post processes prediction_result to get confidence metrics."""
  confidence_metrics = {}
  confidence_metrics['plddt'] = confidence.compute_plddt(
      prediction_result['predicted_lddt']['logits'])
  if 'predicted_aligned_error' in prediction_result:
    confidence_metrics.update(confidence.compute_predicted_aligned_error(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks']))
    confidence_metrics['ptm'] = confidence.predicted_tm_score(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks'],
        asym_id=None)
    if multimer_mode:
      # Compute the ipTM only for the multimer model.
      confidence_metrics['iptm'] = confidence.predicted_tm_score(
          logits=prediction_result['predicted_aligned_error']['logits'],
          breaks=prediction_result['predicted_aligned_error']['breaks'],
          asym_id=prediction_result['predicted_aligned_error']['asym_id'],
          interface=True)
      confidence_metrics['ranking_confidence'] = (
          0.8 * confidence_metrics['iptm'] + 0.2 * confidence_metrics['ptm'])

  if not multimer_mode:
    # Monomer models use mean pLDDT for model ranking.
    confidence_metrics['ranking_confidence'] = np.mean(
        confidence_metrics['plddt'])

  return confidence_metrics


def make_batch(examples):
  return jax.tree.map(lambda *a: jnp.stack(a), *examples)


def get_from_batch(results):
  return jax.tree.map(lambda a: a[0], results)


class RunModel:
  """Container for JAX model."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               params: Optional[Mapping[str, Mapping[str, jax.Array]]] = None):
    self.config = config
    self.params = params
    self.multimer_mode = config.model.global_config.multimer_mode
    self.loss_type = "SO_CE"
    self.pae_loss_w = 0.0
    self.label_smoothing = 0.0

    if self.multimer_mode:
      def _forward_fn(batch):
        model = modules_multimer.AlphaFold(self.config.model)
        return model(
            batch,
            is_training=False)
    else:
      def _forward_fn(batch):
        model = modules.AlphaFold(self.config.model)
        out = model(
            batch,
            is_training=False,
            compute_loss=False,
            ensemble_representations=False)

        return out

    self.batch_apply = jax.vmap(hk.transform(_forward_fn).apply, in_axes=(None, 0, 0))
    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init = jax.jit(hk.transform(_forward_fn).init)
    self.loss = jax.jit(jax.value_and_grad(self.loss_fn, allow_int=False, has_aux=True))
    self.opt_state = None
    self.grad_features = ["restraints_pair_bias"]
    self.solver = optax.adam(learning_rate=0.01)

  def calculate_pae(self, logits, breaks):
    aligned_distance_error_probs = jax.nn.softmax(logits)
    step = (breaks[1] - breaks[0])

    # Add half-step to get the center
    bin_centers = breaks + step / 2
    # Add a catch-all bin at the end.
    bin_centers = jnp.concatenate([bin_centers, jnp.array([bin_centers[-1] + step])],
                               axis=0)
    pae = jnp.sum(aligned_distance_error_probs * bin_centers, axis=-1)

    return pae

  def loss_fn(self, train_x, x, params, random_seed, pae_w=0.0):
    y = self.batch_apply(params, random_seed, {**train_x, **x})

    distogram_loss = jax.vmap(self._distogram_log_loss)(logits=y["distogram"]["logits"],
                                              bin_edges=y["distogram"]["bin_edges"],
                                              dgram_restraints=x["restraints_dgram"])["loss"]
    distogram_loss = jnp.mean(distogram_loss)

    pae_loss = jax.vmap(self.calculate_pae)(logits=y['predicted_aligned_error']['logits'],
                                            breaks=y['predicted_aligned_error']['breaks'])

    pae_loss = jnp.mean(pae_loss) * pae_w
    return distogram_loss+pae_loss, y

  def crop(self, feat, crop_size=512):
    protein_length = feat["aatype"].shape[-1]
    crop_size = min(protein_length, crop_size)
    half_crop = crop_size // 2
    quarter_crop = half_crop // 2
    crop_feat = {}
    
    restr_ixs = np.where(np.sum(feat["restraints_dgram"], axis=-1) > 0)
    sel_restr = np.random.randint(len(restr_ixs[-1]))
    restr_i, restr_j = restr_ixs[-2][sel_restr], restr_ixs[-1][sel_restr]
    logging.info(f"Selected restraint: {restr_ixs[-2][sel_restr]}, {restr_ixs[-1][sel_restr]}: {np.squeeze(feat['restraints_dgram'])[restr_i, restr_j]}")

    crop_1_start = min(restr_i, restr_j) - quarter_crop
    crop_1_start = max(0, crop_1_start) # if negative
    crop_1_start = min(protein_length - crop_size, crop_1_start) # if out of bounds
    crop_1_end = crop_1_start + half_crop

    crop_2_start = max(restr_i, restr_j) - quarter_crop
    crop_2_start = max(crop_2_start, crop_1_end) # if overlapping crops
    crop_2_start = min(protein_length - half_crop, crop_2_start) # if out of bounds
    crop_2_end = crop_2_start + half_crop
    
    keep_index =  np.sort(np.unique(np.concatenate((np.arange(crop_1_start, crop_1_end), np.arange(crop_2_start, crop_2_end)))))
    logging.info(f"Crop info: {crop_1_start}-{crop_1_end}, {crop_2_start}-{crop_2_end}")
    
    for feat_name in feat.keys():
      if self.multimer_mode:
        if feat_name in ["aatype", "asym_id", "sym_id", "entity_id", 
                        "deletion_mean", "all_atom_mask", "all_atom_positions", "entity_mask",
                        "seq_mask", "residue_index"]:
          crop_feat[feat_name] = feat[feat_name][keep_index]
        elif feat_name in ["msa", "template_aatype", "template_all_atom_mask", "template_all_atom_positions",
                          "deletion_matrix", "bert_mask", "msa_mask"]:
          crop_feat[feat_name] = feat[feat_name][:, keep_index]
        elif "restraints" in feat_name:
          crop_feat[feat_name] = feat[feat_name][jnp.ix_(keep_index, keep_index)]
        elif feat_name == "seq_length":
          crop_feat["seq_length"] = np.asarray(crop_size)
        else:
          crop_feat[feat_name] = feat[feat_name]
      else:
        if feat_name in ["aatype", "seq_mask", "residue_index", "template_aatype", "msa_mask", "extra_msa", "extra_msa_mask", "bert_mask", "true_msa", 
                         "extra_has_deletion", "extra_deletion_value"]:
          crop_feat[feat_name] = feat[feat_name][..., keep_index]
        elif feat_name in ["template_all_atom_positions", "template_all_atom_masks", "template_pseudo_beta", "template_pseudo_beta_mask", "msa_feat"]:
          crop_feat[feat_name] = feat[feat_name][:, :, keep_index]
        elif feat_name in ["target_feat", "atom37_atom_exists", "residx_atom37_to_atom14", "residx_atom14_to_atom37", "atom14_atom_exists"]:
          crop_feat[feat_name] = feat[feat_name][:, keep_index]
        elif "restraints" in feat_name:
          crop_feat[feat_name] = feat[feat_name][:, :, keep_index][:, keep_index, :]
        else:
          crop_feat[feat_name] = feat[feat_name]
    return crop_feat, keep_index

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


  def init_bias(self, feature_dict):
    logging.info("Reinitializing learned features and optimizer state")
    feature_dict["restraints_pair_bias"] = np.zeros_like(feature_dict["restraints_pair_bias"])
    self.init_opt = True
    return feature_dict
  
  def set_kl_loss(self):
    self.loss_type = "KL"

  def set_sigmoid_ce_loss(self):
    self.loss_type = "SI_CE"

  def set_softmax_ce_loss(self):
    self.loss_type = "SO_CE"

  def set_pae_w(self, w):
    self.pae_loss_w = w

  def set_label_smoothing(self, s):
    logging.info("Setting label smoothing to: %f", s)
    self.label_smoothing = s

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

    if self.multimer_mode:
      return raw_features

    # Single-chain mode.
    if isinstance(raw_features, dict):
      feat = features.np_example_to_features(
          np_example=raw_features,
          config=self.config,
          random_seed=random_seed)
      return feat
    else:
      return features.tf_example_to_features(
          tf_example=raw_features,
          config=self.config,
          random_seed=random_seed)

  def eval_shape(self, feat: features.FeatureDict) -> jax.ShapeDtypeStruct:
    self.init_params(feat)
    logging.info('Running eval_shape with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))
    shape = jax.eval_shape(self.apply, self.params, jax.random.PRNGKey(0), feat)
    logging.info('Output shape was %s', shape)
    return shape

  def _distogram_log_loss(self, logits, bin_edges, dgram_restraints):
    """Log loss of a distogram."""
    num_bins = logits.shape[-1]
    assert len(logits.shape) == 3
    sq_breaks = jnp.square(bin_edges)

    dist2 = dgram_restraints
    square_mask = jnp.sum(dist2, axis = -1) > 0 #dgram_mask

    if self.loss_type == "SO_CE":
      errors = modules.softmax_cross_entropy(
          labels=dgram_restraints, logits=logits, label_smoothing=self.label_smoothing)
    elif self.loss_type == "KL":
      errors = modules.kl_divergence(
          labels=dgram_restraints, logits=logits)
    else:
      errors = jnp.mean(modules.sigmoid_cross_entropy(
          labels=dgram_restraints, logits=logits), axis=-1)
      #errors = modules.partial_label_ce(labels=dgram_restraints, logits=logits)
    avg_error = (
        jnp.sum(errors * square_mask, axis=(-2, -1)) /
        (1e-6 + jnp.sum(square_mask, axis=(-2, -1))))
    dist2 = dist2[..., 0]
    return dict(loss=avg_error, true_dist=jnp.sqrt(1e-6 + dist2))

  def predict(self,
              feat: features.FeatureDict,
              random_seed: int,
              ) -> Mapping[str, Any]:
    """Makes a prediction by inferencing the model on the provided features.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.
      random_seed: The random seed to use when running the model. In the
        multimer model this controls the MSA sampling.

    Returns:
      A dictionary of model outputs.
    """
    self.init_params(feat)
    logging.info('Running predict with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))

    result = self.apply(self.params, jax.random.PRNGKey(random_seed), feat)
    if "restraints_dgram" in feat:
      distogram_loss = jnp.mean(self._distogram_log_loss(logits=result["distogram"]["logits"],
                                                bin_edges=result["distogram"]["bin_edges"],
                                                dgram_restraints=feat["restraints_dgram"])["loss"])
      logging.info("Distogram loss: %f", distogram_loss)
      result["loss"] = distogram_loss
    # This block is to ensure benchmark timings are accurate. Some blocking is
    # already happening when computing get_confidence_metrics, and this ensures
    # all outputs are blocked on.
    jax.tree.map(lambda x: x.block_until_ready(), result)
    result.update(
        get_confidence_metrics(result, multimer_mode=self.multimer_mode))
    
    logging.info('Output shape was %s',
                 tree.map_structure(lambda x: x.shape, result))
    return result


  def train(self,
              feat: features.FeatureDict,
              random_seed: int,
              crop_size = 384,
              batch_size = 4,
              ) -> Mapping[str, Any]:

    length = feat["aatype"].shape[-1]

    crops = [self.crop(feat, crop_size) for i in range(batch_size)]

    crop_feat = make_batch([c[0] for c in crops])
    crop_index = [c[1] for c in crops]

    logging.info('Running train with shape(feat) = %s and loss type = %s',
                 tree.map_structure(lambda x: x.shape, {k:crop_feat[k] for k in crop_feat if k in self.grad_features}), self.loss_type)
    self.init_params(crop_feat)

    aux, grads = self.loss({k:crop_feat[k] for k in crop_feat if k in self.grad_features},
                           {k:crop_feat[k] for k in crop_feat if k not in self.grad_features},
                           self.params,
                           random_seed=jax.random.split(jax.random.PRNGKey(random_seed), batch_size),
                           pae_w=self.pae_loss_w)
    distogram_loss = aux[0]
    results = aux[1]

    result = get_from_batch(results)
    update_feat = {k:crop_feat[k] for k in self.grad_features}

    if not self.opt_state or self.init_opt:
      self.opt_state = self.solver.init(update_feat)
      self.init_opt = False

    updates, self.opt_state = self.solver.update(grads, self.opt_state, update_feat)

    for i, c_i in enumerate(crop_index):
      for grad_feature in self.grad_features:
        if self.multimer_mode:
          u = updates[grad_feature][i]
          feat[grad_feature][jnp.ix_(c_i, c_i)] += u
        else:
          u = updates[grad_feature][i]
          feat[grad_feature][jnp.ix_(jnp.array([0]), c_i, c_i)] += u

    result["distogram_loss"] = distogram_loss
    for grad_feature in self.grad_features:
      result[grad_feature] = feat[grad_feature]

    result.update(
       get_confidence_metrics(result, multimer_mode=self.multimer_mode))

    return result, {k:feat[k] for k in feat if k in self.grad_features}, crops[0][0]
