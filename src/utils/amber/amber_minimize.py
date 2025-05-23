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

"""Restrained Amber Minimization of a structure."""

import io
import jax
import time
import ml_collections
import numbers
import collections

import jax.numpy as jnp
import numpy as np

from absl import logging
from simtk import openmm
from simtk import unit
from simtk.openmm import app as openmm_app
from simtk.openmm.app.internal.pdbstructure import PdbStructure
from typing import Collection, Optional, Sequence, Dict

from src.utils.amber import cleanup, protein, residue_constants, utils


ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms


def will_restrain(atom: openmm_app.Atom, rset: str) -> bool:
  """Returns True if the atom will be restrained by the given restraint set."""

  if rset == "non_hydrogen":
    return atom.element.name != "hydrogen"
  elif rset == "c_alpha":
    return atom.name == "CA"


def _add_restraints(
    system: openmm.System,
    reference_pdb: openmm_app.PDBFile,
    stiffness: unit.Unit,
    rset: str,
    exclude_residues: Sequence[int]):
  """Adds a harmonic potential that restrains the system to a structure."""
  assert rset in ["non_hydrogen", "c_alpha"]

  force = openmm.CustomExternalForce(
      "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
  force.addGlobalParameter("k", stiffness)
  for p in ["x0", "y0", "z0"]:
    force.addPerParticleParameter(p)

  for i, atom in enumerate(reference_pdb.topology.atoms()):
    if atom.residue.index in exclude_residues:
      continue
    if will_restrain(atom, rset):
      force.addParticle(i, reference_pdb.positions[i])
  logging.info("Restraining %d / %d particles.",
               force.getNumParticles(), system.getNumParticles())
  system.addForce(force)


def _openmm_minimize(
    pdb_str: str,
    max_iterations: int,
    tolerance: unit.Unit,
    stiffness: unit.Unit,
    restraint_set: str,
    exclude_residues: Sequence[int],
    use_gpu: bool):
  """Minimize energy via openmm."""

  pdb_file = io.StringIO(pdb_str)
  pdb = openmm_app.PDBFile(pdb_file)

  force_field = openmm_app.ForceField("amber99sb.xml")
  constraints = openmm_app.HBonds
  system = force_field.createSystem(
      pdb.topology, constraints=constraints)
  if stiffness > 0 * ENERGY / (LENGTH**2):
    _add_restraints(system, pdb, stiffness, restraint_set, exclude_residues)

  integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
  platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
  simulation = openmm_app.Simulation(
      pdb.topology, system, integrator, platform)
  simulation.context.setPositions(pdb.positions)

  ret = {}
  state = simulation.context.getState(getEnergy=True, getPositions=True)
  ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
  ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
  simulation.minimizeEnergy(maxIterations=max_iterations,
                            tolerance=tolerance)
  state = simulation.context.getState(getEnergy=True, getPositions=True)
  ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
  ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
  ret["min_pdb"] = _get_pdb_string(simulation.topology, state.getPositions())
  return ret


def _get_pdb_string(topology: openmm_app.Topology, positions: unit.Quantity):
  """Returns a pdb string provided OpenMM topology and positions."""
  with io.StringIO() as f:
    openmm_app.PDBFile.writeFile(topology, positions, f)
    return f.getvalue()


def _check_cleaned_atoms(pdb_cleaned_string: str, pdb_ref_string: str):
  """Checks that no atom positions have been altered by cleaning."""
  cleaned = openmm_app.PDBFile(io.StringIO(pdb_cleaned_string))
  reference = openmm_app.PDBFile(io.StringIO(pdb_ref_string))

  cl_xyz = np.array(cleaned.getPositions().value_in_unit(LENGTH))
  ref_xyz = np.array(reference.getPositions().value_in_unit(LENGTH))

  for ref_res, cl_res in zip(reference.topology.residues(),
                             cleaned.topology.residues()):
    assert ref_res.name == cl_res.name
    for rat in ref_res.atoms():
      for cat in cl_res.atoms():
        if cat.name == rat.name:
          if not np.array_equal(cl_xyz[cat.index], ref_xyz[rat.index]):
            raise ValueError(f"Coordinates of cleaned atom {cat} do not match "
                             f"coordinates of reference atom {rat}.")


def _check_residues_are_well_defined(prot: protein.Protein):
  """Checks that all residues contain non-empty atom sets."""
  if (prot.atom_mask.sum(axis=-1) == 0).any():
    raise ValueError("Amber minimization can only be performed on proteins with"
                     " well-defined residues. This protein contains at least"
                     " one residue with no atoms.")

def _check_atom_mask_is_ideal(prot):
  """Sanity-check the atom mask is ideal, up to a possible OXT."""
  atom_mask = prot.atom_mask
  ideal_atom_mask = protein.ideal_atom_mask(prot)
  utils.assert_equal_nonterminal_atom_types(atom_mask, ideal_atom_mask)

def squared_difference(x, y):
  return jnp.square(x - y)

def clean_protein(
    prot: protein.Protein,
    checks: bool = True):
  """Adds missing atoms to Protein instance.

  Args:
    prot: A `protein.Protein` instance.
    checks: A `bool` specifying whether to add additional checks to the cleaning
      process.

  Returns:
    pdb_string: A string of the cleaned protein.
  """
  _check_atom_mask_is_ideal(prot)

  # Clean pdb.
  prot_pdb_string = protein.to_pdb(prot)
  pdb_file = io.StringIO(prot_pdb_string)
  alterations_info = {}
  fixed_pdb = cleanup.fix_pdb(pdb_file, alterations_info)
  fixed_pdb_file = io.StringIO(fixed_pdb)
  pdb_structure = PdbStructure(fixed_pdb_file)
  cleanup.clean_structure(pdb_structure, alterations_info)

  logging.info("alterations info: %s", alterations_info)

  # Write pdb file of cleaned structure.
  as_file = openmm_app.PDBFile(pdb_structure)
  pdb_string = _get_pdb_string(as_file.getTopology(), as_file.getPositions())
  if checks:
    _check_cleaned_atoms(pdb_string, prot_pdb_string)
  return pdb_string


def make_atom14_positions(prot):
  """Constructs denser atom positions (14 dimensions instead of 37)."""
  restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
  restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
  restype_atom14_mask = []

  for rt in residue_constants.restypes:
    atom_names = residue_constants.restype_name_to_atom14_names[
        residue_constants.restype_1to3[rt]]

    restype_atom14_to_atom37.append([
        (residue_constants.atom_order[name] if name else 0)
        for name in atom_names
    ])

    atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
    restype_atom37_to_atom14.append([
        (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
        for name in residue_constants.atom_types
    ])

    restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

  # Add dummy mapping for restype "UNK".
  restype_atom14_to_atom37.append([0] * 14)
  restype_atom37_to_atom14.append([0] * 37)
  restype_atom14_mask.append([0.] * 14)

  restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
  restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
  restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

  # Create the mapping for (residx, atom14) --> atom37, i.e. an array
  # with shape (num_res, 14) containing the atom37 indices for this protein.
  residx_atom14_to_atom37 = restype_atom14_to_atom37[prot["aatype"]]
  residx_atom14_mask = restype_atom14_mask[prot["aatype"]]

  # Create a mask for known ground truth positions.
  residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
      prot["all_atom_mask"], residx_atom14_to_atom37, axis=1).astype(np.float32)

  # Gather the ground truth positions.
  residx_atom14_gt_positions = residx_atom14_gt_mask[:, :, None] * (
      np.take_along_axis(prot["all_atom_positions"],
                         residx_atom14_to_atom37[..., None],
                         axis=1))

  prot["atom14_atom_exists"] = residx_atom14_mask
  prot["atom14_gt_exists"] = residx_atom14_gt_mask
  prot["atom14_gt_positions"] = residx_atom14_gt_positions

  prot["residx_atom14_to_atom37"] = residx_atom14_to_atom37

  # Create the gather indices for mapping back.
  residx_atom37_to_atom14 = restype_atom37_to_atom14[prot["aatype"]]
  prot["residx_atom37_to_atom14"] = residx_atom37_to_atom14

  # Create the corresponding mask.
  restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
  for restype, restype_letter in enumerate(residue_constants.restypes):
    restype_name = residue_constants.restype_1to3[restype_letter]
    atom_names = residue_constants.residue_atoms[restype_name]
    for atom_name in atom_names:
      atom_type = residue_constants.atom_order[atom_name]
      restype_atom37_mask[restype, atom_type] = 1

  residx_atom37_mask = restype_atom37_mask[prot["aatype"]]
  prot["atom37_atom_exists"] = residx_atom37_mask

  # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
  # alternative ground truth coordinates where the naming is swapped
  restype_3 = [
      residue_constants.restype_1to3[res] for res in residue_constants.restypes
  ]
  restype_3 += ["UNK"]

  # Matrices for renaming ambiguous atoms.
  all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
  for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
    correspondences = np.arange(14)
    for source_atom_swap, target_atom_swap in swap.items():
      source_index = residue_constants.restype_name_to_atom14_names[
          resname].index(source_atom_swap)
      target_index = residue_constants.restype_name_to_atom14_names[
          resname].index(target_atom_swap)
      correspondences[source_index] = target_index
      correspondences[target_index] = source_index
      renaming_matrix = np.zeros((14, 14), dtype=np.float32)
      for index, correspondence in enumerate(correspondences):
        renaming_matrix[index, correspondence] = 1.
    all_matrices[resname] = renaming_matrix.astype(np.float32)
  renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])

  # Pick the transformation matrices for the given residue sequence
  # shape (num_res, 14, 14).
  renaming_transform = renaming_matrices[prot["aatype"]]

  # Apply it to the ground truth positions. shape (num_res, 14, 3).
  alternative_gt_positions = np.einsum("rac,rab->rbc",
                                       residx_atom14_gt_positions,
                                       renaming_transform)
  prot["atom14_alt_gt_positions"] = alternative_gt_positions

  # Create the mask for the alternative ground truth (differs from the
  # ground truth mask, if only one of the atoms in an ambiguous pair has a
  # ground truth position).
  alternative_gt_mask = np.einsum("ra,rab->rb",
                                  residx_atom14_gt_mask,
                                  renaming_transform)

  prot["atom14_alt_gt_exists"] = alternative_gt_mask

  # Create an ambiguous atoms mask.  shape: (21, 14).
  restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
  for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
    for atom_name1, atom_name2 in swap.items():
      restype = residue_constants.restype_order[
          residue_constants.restype_3to1[resname]]
      atom_idx1 = residue_constants.restype_name_to_atom14_names[resname].index(
          atom_name1)
      atom_idx2 = residue_constants.restype_name_to_atom14_names[resname].index(
          atom_name2)
      restype_atom14_is_ambiguous[restype, atom_idx1] = 1
      restype_atom14_is_ambiguous[restype, atom_idx2] = 1

  # From this create an ambiguous_mask for the given sequence.
  prot["atom14_atom_is_ambiguous"] = (
      restype_atom14_is_ambiguous[prot["aatype"]])

  return prot



def between_residue_bond_loss(
    pred_atom_positions: jnp.ndarray,  # (N, 37(14), 3)
    pred_atom_mask: jnp.ndarray,  # (N, 37(14))
    residue_index: jnp.ndarray,  # (N)
    aatype: jnp.ndarray,  # (N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0
) -> Dict[str, jnp.ndarray]:
  """Flat-bottom loss to penalize structural violations between residues.

  This is a loss penalizing any violation of the geometry around the peptide
  bond between consecutive amino acids. This loss corresponds to
  Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

  Args:
    pred_atom_positions: Atom positions in atom37/14 representation
    pred_atom_mask: Atom mask in atom37/14 representation
    residue_index: Residue index for given amino acid, this is assumed to be
      monotonically increasing.
    aatype: Amino acid type of given residue
    tolerance_factor_soft: soft tolerance factor measured in standard deviations
      of pdb distributions
    tolerance_factor_hard: hard tolerance factor measured in standard deviations
      of pdb distributions

  Returns:
    Dict containing:
      * "c_n_loss_mean": Loss for peptide bond length violations
      * "ca_c_n_loss_mean": Loss for violations of bond angle around C spanned
          by CA, C, N
      * "c_n_ca_loss_mean": Loss for violations of bond angle around N spanned
          by C, N, CA
      * "per_residue_loss_sum": sum of all losses for each residue
      * "per_residue_violation_mask": mask denoting all residues with violation
          present.
  """
  assert len(pred_atom_positions.shape) == 3
  assert len(pred_atom_mask.shape) == 2
  assert len(residue_index.shape) == 1
  assert len(aatype.shape) == 1

  # Get the positions of the relevant backbone atoms.
  this_ca_pos = pred_atom_positions[:-1, 1, :]  # (N - 1, 3)
  this_ca_mask = pred_atom_mask[:-1, 1]         # (N - 1)
  this_c_pos = pred_atom_positions[:-1, 2, :]   # (N - 1, 3)
  this_c_mask = pred_atom_mask[:-1, 2]          # (N - 1)
  next_n_pos = pred_atom_positions[1:, 0, :]    # (N - 1, 3)
  next_n_mask = pred_atom_mask[1:, 0]           # (N - 1)
  next_ca_pos = pred_atom_positions[1:, 1, :]   # (N - 1, 3)
  next_ca_mask = pred_atom_mask[1:, 1]          # (N - 1)
  has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(
      jnp.float32)

  # Compute loss for the C--N bond.
  c_n_bond_length = jnp.sqrt(
      1e-6 + jnp.sum(squared_difference(this_c_pos, next_n_pos), axis=-1))

  # The C-N bond to proline has slightly different length because of the ring.
  next_is_proline = (
      aatype[1:] == residue_constants.resname_to_idx["PRO"]).astype(jnp.float32)
  gt_length = (
      (1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
      + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
  gt_stddev = (
      (1. - next_is_proline) *
      residue_constants.between_res_bond_length_stddev_c_n[0] +
      next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
  c_n_bond_length_error = jnp.sqrt(1e-6 +
                                   jnp.square(c_n_bond_length - gt_length))
  c_n_loss_per_residue = jax.nn.relu(
      c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
  mask = this_c_mask * next_n_mask * has_no_gap_mask
  c_n_loss = jnp.sum(mask * c_n_loss_per_residue) / (jnp.sum(mask) + 1e-6)
  c_n_violation_mask = mask * (
      c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

  # Compute loss for the angles.
  ca_c_bond_length = jnp.sqrt(1e-6 + jnp.sum(
      squared_difference(this_ca_pos, this_c_pos), axis=-1))
  n_ca_bond_length = jnp.sqrt(1e-6 + jnp.sum(
      squared_difference(next_n_pos, next_ca_pos), axis=-1))

  c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[:, None]
  c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[:, None]
  n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[:, None]

  ca_c_n_cos_angle = jnp.sum(c_ca_unit_vec * c_n_unit_vec, axis=-1)
  gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
  gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
  ca_c_n_cos_angle_error = jnp.sqrt(
      1e-6 + jnp.square(ca_c_n_cos_angle - gt_angle))
  ca_c_n_loss_per_residue = jax.nn.relu(
      ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
  mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
  ca_c_n_loss = jnp.sum(mask * ca_c_n_loss_per_residue) / (jnp.sum(mask) + 1e-6)
  ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error >
                                  (tolerance_factor_hard * gt_stddev))

  c_n_ca_cos_angle = jnp.sum((-c_n_unit_vec) * n_ca_unit_vec, axis=-1)
  gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
  gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
  c_n_ca_cos_angle_error = jnp.sqrt(
      1e-6 + jnp.square(c_n_ca_cos_angle - gt_angle))
  c_n_ca_loss_per_residue = jax.nn.relu(
      c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
  mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
  c_n_ca_loss = jnp.sum(mask * c_n_ca_loss_per_residue) / (jnp.sum(mask) + 1e-6)
  c_n_ca_violation_mask = mask * (
      c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

  # Compute a per residue loss (equally distribute the loss to both
  # neighbouring residues).
  per_residue_loss_sum = (c_n_loss_per_residue +
                          ca_c_n_loss_per_residue +
                          c_n_ca_loss_per_residue)
  per_residue_loss_sum = 0.5 * (jnp.pad(per_residue_loss_sum, [[0, 1]]) +
                                jnp.pad(per_residue_loss_sum, [[1, 0]]))

  # Compute hard violations.
  violation_mask = jnp.max(
      jnp.stack([c_n_violation_mask,
                 ca_c_n_violation_mask,
                 c_n_ca_violation_mask]), axis=0)
  violation_mask = jnp.maximum(
      jnp.pad(violation_mask, [[0, 1]]),
      jnp.pad(violation_mask, [[1, 0]]))

  return {"c_n_loss_mean": c_n_loss,  # shape ()
          "ca_c_n_loss_mean": ca_c_n_loss,  # shape ()
          "c_n_ca_loss_mean": c_n_ca_loss,  # shape ()
          "per_residue_loss_sum": per_residue_loss_sum,  # shape (N)
          "per_residue_violation_mask": violation_mask  # shape (N)
         }


def between_residue_clash_loss(
    atom14_pred_positions: jnp.ndarray,  # (N, 14, 3)
    atom14_atom_exists: jnp.ndarray,  # (N, 14)
    atom14_atom_radius: jnp.ndarray,  # (N, 14)
    residue_index: jnp.ndarray,  # (N)
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5
) -> Dict[str, jnp.ndarray]:
  """Loss to penalize steric clashes between residues.

  This is a loss penalizing any steric clashes due to non bonded atoms in
  different peptides coming too close. This loss corresponds to the part with
  different residues of
  Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

  Args:
    atom14_pred_positions: Predicted positions of atoms in
      global prediction frame
    atom14_atom_exists: Mask denoting whether atom at positions exists for given
      amino acid type
    atom14_atom_radius: Van der Waals radius for each atom.
    residue_index: Residue index for given amino acid.
    overlap_tolerance_soft: Soft tolerance factor.
    overlap_tolerance_hard: Hard tolerance factor.

  Returns:
    Dict containing:
      * "mean_loss": average clash loss
      * "per_atom_loss_sum": sum of all clash losses per atom, shape (N, 14)
      * "per_atom_clash_mask": mask whether atom clashes with any other atom
          shape (N, 14)
  """
  assert len(atom14_pred_positions.shape) == 3
  assert len(atom14_atom_exists.shape) == 2
  assert len(atom14_atom_radius.shape) == 2
  assert len(residue_index.shape) == 1

  # Create the distance matrix.
  # (N, N, 14, 14)
  dists = jnp.sqrt(1e-10 + jnp.sum(
      squared_difference(
          atom14_pred_positions[:, None, :, None, :],
          atom14_pred_positions[None, :, None, :, :]),
      axis=-1))

  # Create the mask for valid distances.
  # shape (N, N, 14, 14)
  dists_mask = (atom14_atom_exists[:, None, :, None] *
                atom14_atom_exists[None, :, None, :])

  # Mask out all the duplicate entries in the lower triangular matrix.
  # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
  # are handled separately.
  dists_mask *= (
      residue_index[:, None, None, None] < residue_index[None, :, None, None])

  # Backbone C--N bond between subsequent residues is no clash.
  c_one_hot = jax.nn.one_hot(2, num_classes=14)
  n_one_hot = jax.nn.one_hot(0, num_classes=14)
  neighbour_mask = ((residue_index[:, None, None, None] +
                     1) == residue_index[None, :, None, None])
  c_n_bonds = neighbour_mask * c_one_hot[None, None, :,
                                         None] * n_one_hot[None, None, None, :]
  dists_mask *= (1. - c_n_bonds)

  # Disulfide bridge between two cysteines is no clash.
  cys_sg_idx = residue_constants.restype_name_to_atom14_names["CYS"].index("SG")
  cys_sg_one_hot = jax.nn.one_hot(cys_sg_idx, num_classes=14)
  disulfide_bonds = (cys_sg_one_hot[None, None, :, None] *
                     cys_sg_one_hot[None, None, None, :])
  dists_mask *= (1. - disulfide_bonds)

  # Compute the lower bound for the allowed distances.
  # shape (N, N, 14, 14)
  dists_lower_bound = dists_mask * (atom14_atom_radius[:, None, :, None] +
                                    atom14_atom_radius[None, :, None, :])

  # Compute the error.
  # shape (N, N, 14, 14)
  dists_to_low_error = dists_mask * jax.nn.relu(
      dists_lower_bound - overlap_tolerance_soft - dists)

  # Compute the mean loss.
  # shape ()
  mean_loss = (jnp.sum(dists_to_low_error)
               / (1e-6 + jnp.sum(dists_mask)))

  # Compute the per atom loss sum.
  # shape (N, 14)
  per_atom_loss_sum = (jnp.sum(dists_to_low_error, axis=[0, 2]) +
                       jnp.sum(dists_to_low_error, axis=[1, 3]))

  # Compute the hard clash mask.
  # shape (N, N, 14, 14)
  clash_mask = dists_mask * (
      dists < (dists_lower_bound - overlap_tolerance_hard))

  # Compute the per atom clash.
  # shape (N, 14)
  per_atom_clash_mask = jnp.maximum(
      jnp.max(clash_mask, axis=[0, 2]),
      jnp.max(clash_mask, axis=[1, 3]))

  return {"mean_loss": mean_loss,  # shape ()
          "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
          "per_atom_clash_mask": per_atom_clash_mask  # shape (N, 14)
         }

def within_residue_violations(
    atom14_pred_positions: jnp.ndarray,  # (N, 14, 3)
    atom14_atom_exists: jnp.ndarray,  # (N, 14)
    atom14_dists_lower_bound: jnp.ndarray,  # (N, 14, 14)
    atom14_dists_upper_bound: jnp.ndarray,  # (N, 14, 14)
    tighten_bounds_for_loss=0.0,
) -> Dict[str, jnp.ndarray]:
  """Loss to penalize steric clashes within residues.

  This is a loss penalizing any steric violations or clashes of non-bonded atoms
  in a given peptide. This loss corresponds to the part with
  the same residues of
  Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

  Args:
    atom14_pred_positions: Predicted positions of atoms in
      global prediction frame
    atom14_atom_exists: Mask denoting whether atom at positions exists for given
      amino acid type
    atom14_dists_lower_bound: Lower bound on allowed distances.
    atom14_dists_upper_bound: Upper bound on allowed distances
    tighten_bounds_for_loss: Extra factor to tighten loss

  Returns:
    Dict containing:
      * "per_atom_loss_sum": sum of all clash losses per atom, shape (N, 14)
      * "per_atom_clash_mask": mask whether atom clashes with any other atom
          shape (N, 14)
  """
  assert len(atom14_pred_positions.shape) == 3
  assert len(atom14_atom_exists.shape) == 2
  assert len(atom14_dists_lower_bound.shape) == 3
  assert len(atom14_dists_upper_bound.shape) == 3

  # Compute the mask for each residue.
  # shape (N, 14, 14)
  dists_masks = (1. - jnp.eye(14, 14)[None])
  dists_masks *= (atom14_atom_exists[:, :, None] *
                  atom14_atom_exists[:, None, :])

  # Distance matrix
  # shape (N, 14, 14)
  dists = jnp.sqrt(1e-10 + jnp.sum(
      squared_difference(
          atom14_pred_positions[:, :, None, :],
          atom14_pred_positions[:, None, :, :]),
      axis=-1))

  # Compute the loss.
  # shape (N, 14, 14)
  dists_to_low_error = jax.nn.relu(
      atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
  dists_to_high_error = jax.nn.relu(
      dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
  loss = dists_masks * (dists_to_low_error + dists_to_high_error)

  # Compute the per atom loss sum.
  # shape (N, 14)
  per_atom_loss_sum = (jnp.sum(loss, axis=1) +
                       jnp.sum(loss, axis=2))

  # Compute the violations mask.
  # shape (N, 14, 14)
  violations = dists_masks * ((dists < atom14_dists_lower_bound) |
                              (dists > atom14_dists_upper_bound))

  # Compute the per atom violations.
  # shape (N, 14)
  per_atom_violations = jnp.maximum(
      jnp.max(violations, axis=1), jnp.max(violations, axis=2))

  return {"per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
          "per_atom_violations": per_atom_violations  # shape (N, 14)
         }

def batched_gather(params, indices, axis=0, batch_dims=0):
  """Implements a JAX equivalent of `tf.gather` with `axis` and `batch_dims`."""
  take_fn = lambda p, i: jnp.take(p, i, axis=axis)
  for _ in range(batch_dims):
    take_fn = jax.vmap(take_fn)
  return take_fn(params, indices)

def find_structural_violations(
    batch: Dict[str, jnp.ndarray],
    atom14_pred_positions: jnp.ndarray,  # (N, 14, 3)
    config: ml_collections.ConfigDict
    ):
  """Computes several checks for structural violations."""

  # Compute between residue backbone violations of bonds and angles.
  connection_violations = between_residue_bond_loss(
      pred_atom_positions=atom14_pred_positions,
      pred_atom_mask=batch["atom14_atom_exists"].astype(jnp.float32),
      residue_index=batch["residue_index"].astype(jnp.float32),
      aatype=batch["aatype"],
      tolerance_factor_soft=config.violation_tolerance_factor,
      tolerance_factor_hard=config.violation_tolerance_factor)

  # Compute the Van der Waals radius for every atom
  # (the first letter of the atom name is the element type).
  # Shape: (N, 14).
  atomtype_radius = jnp.array([
      residue_constants.van_der_waals_radius[name[0]]
      for name in residue_constants.atom_types
  ])
  atom14_atom_radius = batch["atom14_atom_exists"] * batched_gather(
      atomtype_radius, batch["residx_atom14_to_atom37"])

  # Compute the between residue clash loss.
  between_residue_clashes = between_residue_clash_loss(
      atom14_pred_positions=atom14_pred_positions,
      atom14_atom_exists=batch["atom14_atom_exists"],
      atom14_atom_radius=atom14_atom_radius,
      residue_index=batch["residue_index"],
      overlap_tolerance_soft=config.clash_overlap_tolerance,
      overlap_tolerance_hard=config.clash_overlap_tolerance)

  # Compute all within-residue violations (clashes,
  # bond length and angle violations).
  restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
      overlap_tolerance=config.clash_overlap_tolerance,
      bond_length_tolerance_factor=config.violation_tolerance_factor)
  atom14_dists_lower_bound = batched_gather(
      restype_atom14_bounds["lower_bound"], batch["aatype"])
  atom14_dists_upper_bound = batched_gather(
      restype_atom14_bounds["upper_bound"], batch["aatype"])
  within_residue_violations_val = within_residue_violations(
      atom14_pred_positions=atom14_pred_positions,
      atom14_atom_exists=batch["atom14_atom_exists"],
      atom14_dists_lower_bound=atom14_dists_lower_bound,
      atom14_dists_upper_bound=atom14_dists_upper_bound,
      tighten_bounds_for_loss=0.0)

  # Combine them to a single per-residue violation mask (used later for LDDT).
  per_residue_violations_mask = jnp.max(jnp.stack([
      connection_violations["per_residue_violation_mask"],
      jnp.max(between_residue_clashes["per_atom_clash_mask"], axis=-1),
      jnp.max(within_residue_violations_val["per_atom_violations"],
              axis=-1)]), axis=0)

  return {
      "between_residues": {
          "bonds_c_n_loss_mean":
              connection_violations["c_n_loss_mean"],  # ()
          "angles_ca_c_n_loss_mean":
              connection_violations["ca_c_n_loss_mean"],  # ()
          "angles_c_n_ca_loss_mean":
              connection_violations["c_n_ca_loss_mean"],  # ()
          "connections_per_residue_loss_sum":
              connection_violations["per_residue_loss_sum"],  # (N)
          "connections_per_residue_violation_mask":
              connection_violations["per_residue_violation_mask"],  # (N)
          "clashes_mean_loss":
              between_residue_clashes["mean_loss"],  # ()
          "clashes_per_atom_loss_sum":
              between_residue_clashes["per_atom_loss_sum"],  # (N, 14)
          "clashes_per_atom_clash_mask":
              between_residue_clashes["per_atom_clash_mask"],  # (N, 14)
      },
      "within_residues": {
          "per_atom_loss_sum":
              within_residue_violations_val["per_atom_loss_sum"],  # (N, 14)
          "per_atom_violations":
              within_residue_violations_val["per_atom_violations"],  # (N, 14),
      },
      "total_per_residue_violations_mask":
          per_residue_violations_mask,  # (N)
  }

def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
  """Masked mean."""
  if drop_mask_channel:
    mask = mask[..., 0]

  mask_shape = mask.shape
  value_shape = value.shape

  assert len(mask_shape) == len(value_shape)

  if isinstance(axis, numbers.Integral):
    axis = [axis]
  elif axis is None:
    axis = list(range(len(mask_shape)))
  assert isinstance(axis, collections.Iterable), (
      "axis needs to be either an iterable, integer or 'None'")

def extreme_ca_ca_distance_violations(
    pred_atom_positions: jnp.ndarray,  # (N, 37(14), 3)
    pred_atom_mask: jnp.ndarray,  # (N, 37(14))
    residue_index: jnp.ndarray,  # (N)
    max_angstrom_tolerance=1.5
    ) -> jnp.ndarray:
  """Counts residues whose Ca is a large distance from its neighbour.

  Measures the fraction of CA-CA pairs between consecutive amino acids that are
  more than "max_angstrom_tolerance" apart.

  Args:
    pred_atom_positions: Atom positions in atom37/14 representation
    pred_atom_mask: Atom mask in atom37/14 representation
    residue_index: Residue index for given amino acid, this is assumed to be
      monotonically increasing.
    max_angstrom_tolerance: Maximum distance allowed to not count as violation.
  Returns:
    Fraction of consecutive CA-CA pairs with violation.
  """
  this_ca_pos = pred_atom_positions[:-1, 1, :]  # (N - 1, 3)
  this_ca_mask = pred_atom_mask[:-1, 1]         # (N - 1)
  next_ca_pos = pred_atom_positions[1:, 1, :]  # (N - 1, 3)
  next_ca_mask = pred_atom_mask[1:, 1]  # (N - 1)
  has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(
      jnp.float32)
  ca_ca_distance = jnp.sqrt(
      1e-6 + jnp.sum(squared_difference(this_ca_pos, next_ca_pos), axis=-1))
  violations = (ca_ca_distance -
                residue_constants.ca_ca) > max_angstrom_tolerance
  mask = this_ca_mask * next_ca_mask * has_no_gap_mask
  return mask_mean(mask=mask, value=violations)

def compute_violation_metrics(
    batch: Dict[str, jnp.ndarray],
    atom14_pred_positions: jnp.ndarray,  # (N, 14, 3)
    violations: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
  """Compute several metrics to assess the structural violations."""

  ret = {}
  extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
      pred_atom_positions=atom14_pred_positions,
      pred_atom_mask=batch["atom14_atom_exists"].astype(jnp.float32),
      residue_index=batch["residue_index"].astype(jnp.float32))
  ret["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
  ret["violations_between_residue_bond"] = mask_mean(
      mask=batch["seq_mask"],
      value=violations["between_residues"][
          "connections_per_residue_violation_mask"])
  ret["violations_between_residue_clash"] = mask_mean(
      mask=batch["seq_mask"],
      value=jnp.max(
          violations["between_residues"]["clashes_per_atom_clash_mask"],
          axis=-1))
  ret["violations_within_residue"] = mask_mean(
      mask=batch["seq_mask"],
      value=jnp.max(
          violations["within_residues"]["per_atom_violations"], axis=-1))
  ret["violations_per_residue"] = mask_mean(
      mask=batch["seq_mask"],
      value=violations["total_per_residue_violations_mask"])
  return ret


def find_violations(prot_np: protein.Protein):
  """Analyzes a protein and returns structural violation information.

  Args:
    prot_np: A protein.

  Returns:
    violations: A `dict` of structure components with structural violations.
    violation_metrics: A `dict` of violation metrics.
  """
  batch = {
      "aatype": prot_np.aatype,
      "all_atom_positions": prot_np.atom_positions.astype(np.float32),
      "all_atom_mask": prot_np.atom_mask.astype(np.float32),
      "residue_index": prot_np.residue_index,
  }

  batch["seq_mask"] = np.ones_like(batch["aatype"], np.float32)
  batch = make_atom14_positions(batch)

  violations = find_structural_violations(
      batch=batch,
      atom14_pred_positions=batch["atom14_gt_positions"],
      config=ml_collections.ConfigDict(
          {"violation_tolerance_factor": 12,  # Taken from model config.
           "clash_overlap_tolerance": 1.5,  # Taken from model config.
          }))
  violation_metrics = compute_violation_metrics(
      batch=batch,
      atom14_pred_positions=batch["atom14_gt_positions"],
      violations=violations,
  )

  return violations, violation_metrics


def get_violation_metrics(prot: protein.Protein):
  """Computes violation and alignment metrics."""
  structural_violations, struct_metrics = find_violations(prot)
  violation_idx = np.flatnonzero(
      structural_violations["total_per_residue_violations_mask"])

  struct_metrics["residue_violations"] = violation_idx
  struct_metrics["num_residue_violations"] = len(violation_idx)
  struct_metrics["structural_violations"] = structural_violations
  return struct_metrics


def _run_one_iteration(
    *,
    pdb_string: str,
    max_iterations: int,
    tolerance: float,
    stiffness: float,
    restraint_set: str,
    max_attempts: int,
    use_gpu: bool,
    exclude_residues: Optional[Collection[int]] = None):
  """Runs the minimization pipeline.

  Args:
    pdb_string: A pdb string.
    max_iterations: An `int` specifying the maximum number of L-BFGS iterations.
    A value of 0 specifies no limit.
    tolerance: kcal/mol, the energy tolerance of L-BFGS.
    stiffness: kcal/mol A**2, spring constant of heavy atom restraining
      potential.
    restraint_set: The set of atoms to restrain.
    max_attempts: The maximum number of minimization attempts.
    use_gpu: Whether to run on GPU.
    exclude_residues: An optional list of zero-indexed residues to exclude from
        restraints.

  Returns:
    A `dict` of minimization info.
  """
  exclude_residues = exclude_residues or []

  # Assign physical dimensions.
  tolerance = tolerance * ENERGY
  stiffness = stiffness * ENERGY / (LENGTH**2)

  start = time.time()
  minimized = False
  attempts = 0
  while not minimized and attempts < max_attempts:
    attempts += 1
    try:
      logging.info("Minimizing protein, attempt %d of %d.",
                   attempts, max_attempts)
      ret = _openmm_minimize(
          pdb_string, max_iterations=max_iterations,
          tolerance=tolerance, stiffness=stiffness,
          restraint_set=restraint_set,
          exclude_residues=exclude_residues,
          use_gpu=use_gpu)
      minimized = True
    except Exception as e:  # pylint: disable=broad-except
      logging.info(e)
  if not minimized:
    raise ValueError(f"Minimization failed after {max_attempts} attempts.")
  ret["opt_time"] = time.time() - start
  ret["min_attempts"] = attempts
  return ret


def run_pipeline(
    prot: protein.Protein,
    stiffness: float,
    use_gpu: bool,
    max_outer_iterations: int = 1,
    place_hydrogens_every_iteration: bool = True,
    max_iterations: int = 0,
    tolerance: float = 2.39,
    restraint_set: str = "non_hydrogen",
    max_attempts: int = 100,
    checks: bool = True,
    exclude_residues: Optional[Sequence[int]] = None):
  """Run iterative amber relax.

  Successive relax iterations are performed until all violations have been
  resolved. Each iteration involves a restrained Amber minimization, with
  restraint exclusions determined by violation-participating residues.

  Args:
    prot: A protein to be relaxed.
    stiffness: kcal/mol A**2, the restraint stiffness.
    use_gpu: Whether to run on GPU.
    max_outer_iterations: The maximum number of iterative minimization.
    place_hydrogens_every_iteration: Whether hydrogens are re-initialized
        prior to every minimization.
    max_iterations: An `int` specifying the maximum number of L-BFGS steps
        per relax iteration. A value of 0 specifies no limit.
    tolerance: kcal/mol, the energy tolerance of L-BFGS.
        The default value is the OpenMM default.
    restraint_set: The set of atoms to restrain.
    max_attempts: The maximum number of minimization attempts per iteration.
    checks: Whether to perform cleaning checks.
    exclude_residues: An optional list of zero-indexed residues to exclude from
        restraints.

  Returns:
    out: A dictionary of output values.
  """

  # `protein.to_pdb` will strip any poorly-defined residues so we need to
  # perform this check before `clean_protein`.
  _check_residues_are_well_defined(prot)
  pdb_string = clean_protein(prot, checks=checks)

  exclude_residues = exclude_residues or []
  exclude_residues = set(exclude_residues)
  violations = np.inf
  iteration = 0
  # while violations > 0 and iteration < max_outer_iterations:
  while iteration < max_outer_iterations:
    ret = _run_one_iteration(
        pdb_string=pdb_string,
        exclude_residues=exclude_residues,
        max_iterations=max_iterations,
        tolerance=tolerance,
        stiffness=stiffness,
        restraint_set=restraint_set,
        max_attempts=max_attempts,
        use_gpu=use_gpu)
    prot = protein.from_pdb_string(ret["min_pdb"])
    if place_hydrogens_every_iteration:
      pdb_string = clean_protein(prot, checks=True)
    else:
      pdb_string = ret["min_pdb"]
    ret.update(get_violation_metrics(prot))
    ret.update({
        "num_exclusions": len(exclude_residues),
        "iteration": iteration,
    })
    violations = ret["violations_per_residue"]
    # print("violations:", violations)
    exclude_residues = exclude_residues.union(ret["residue_violations"])

    logging.info("Iteration completed: Einit %.2f Efinal %.2f Time %.2f s "
                 "num residue violations %d num residue exclusions %d ",
                 ret["einit"], ret["efinal"], ret["opt_time"],
                 ret["num_residue_violations"], ret["num_exclusions"])
    iteration += 1
  return ret


def get_initial_energies(pdb_strs: Sequence[str],
                         stiffness: float = 0.0,
                         restraint_set: str = "non_hydrogen",
                         exclude_residues: Optional[Sequence[int]] = None):
  """Returns initial potential energies for a sequence of PDBs.

  Assumes the input PDBs are ready for minimization, and all have the same
  topology.
  Allows time to be saved by not pdbfixing / rebuilding the system.

  Args:
    pdb_strs: List of PDB strings.
    stiffness: kcal/mol A**2, spring constant of heavy atom restraining
        potential.
    restraint_set: Which atom types to restrain.
    exclude_residues: An optional list of zero-indexed residues to exclude from
        restraints.

  Returns:
    A list of initial energies in the same order as pdb_strs.
  """
  exclude_residues = exclude_residues or []

  openmm_pdbs = [openmm_app.PDBFile(PdbStructure(io.StringIO(p)))
                 for p in pdb_strs]
  force_field = openmm_app.ForceField("amber99sb.xml")
  system = force_field.createSystem(openmm_pdbs[0].topology,
                                    constraints=openmm_app.HBonds)
  stiffness = stiffness * ENERGY / (LENGTH**2)
  if stiffness > 0 * ENERGY / (LENGTH**2):
    _add_restraints(system, openmm_pdbs[0], stiffness, restraint_set,
                    exclude_residues)
  simulation = openmm_app.Simulation(openmm_pdbs[0].topology,
                                     system,
                                     openmm.LangevinIntegrator(0, 0.01, 0.0),
                                     openmm.Platform.getPlatformByName("CPU"))
  energies = []
  for pdb in openmm_pdbs:
    try:
      simulation.context.setPositions(pdb.positions)
      state = simulation.context.getState(getEnergy=True)
      energies.append(state.getPotentialEnergy().value_in_unit(ENERGY))
    except Exception as e:  # pylint: disable=broad-except
      logging.error("Error getting initial energy, returning large value %s", e)
      energies.append(unit.Quantity(1e20, ENERGY))
  return energies
