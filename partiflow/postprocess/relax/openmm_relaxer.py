import io
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Any

import pdbfixer
import openmm
from openmm import app as openmm_app
from openmm import unit


ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "relax_assets")
_CUSTOM_RESIDUE_XML_PATH = os.path.join(_ASSET_DIR, "topology.xml")
_STAPLED_NTERM_TEMPLATE = {
    "LYZ": "NLYZ", "ADZ": "NADZ", "GLZ": "NGLZ",
    "MK8": "NMK8", "NLE": "NNLE", "PMK8": "NPMK8", "PNLE": "NPNLE",
}
_STAPLED_CTERM_TEMPLATE = {
    "LYZ": "CLYZ", "ADZ": "CADZ", "GLZ": "CGLZ",
    "MK8": "CMK8", "NLE": "CNLE", "PMK8": "CPMK8", "PNLE": "CPNLE",
}
_NONTERMINAL_TEMPLATE_RESIDUES = {"RTY", "MK8", "NLE", "PMK8", "PNLE"}
_STANDARD_INTERNAL_TEMPLATES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


@lru_cache(maxsize=None)
def _build_filtered_forcefield_xml(residue_names_key: tuple[str, ...]) -> str:
    """Write a filtered copy of topology.xml containing only the requested residues."""
    residue_names = set(residue_names_key)
    tree = ET.parse(_CUSTOM_RESIDUE_XML_PATH)
    root = tree.getroot()
    residues_node = root.find("Residues")
    if residues_node is None:
        raise ValueError(f"topology.xml at {_CUSTOM_RESIDUE_XML_PATH} is missing a Residues block.")
    for node in list(residues_node):
        if node.tag == "Residue" and node.get("name", "") not in residue_names:
            residues_node.remove(node)
    handle = tempfile.NamedTemporaryFile(prefix="partiflow_ff_", suffix=".xml", delete=False)
    handle.close()
    tree.write(handle.name, encoding="utf-8", xml_declaration=True)
    return handle.name


def run_topology_relax(
    fix_result: Any,
    *,
    platform: str = "CUDA",
    stiffness: float = 10.0,
    max_iterations: int = 0,
    tolerance: Any = 2.39 * unit.kilocalories_per_mole,
) -> tuple[str, dict[str, Any]]:
    """
    Lightweight relaxer that accepts a TopologyFixResult from TopologyFixer.

    All topology-aware chemistry (atom deletions, renames, bond additions) has
    already been applied to fix_result.topology.  This function only needs to:
      - load the correct force field (with custom XML when needed)
      - run energy minimization
      - return the minimized PDB string and energy info
    """
    from partiflow.postprocess.relax.topology import _write_pdb, _add_connects

    topology = fix_result.topology
    positions = fix_result.positions
    custom_residue_names = set(fix_result.custom_residue_names or set())
    template_overrides = dict(getattr(fix_result, "residue_template_overrides", {}) or {})
    custom_residue_names.update(str(name) for name in template_overrides.values())

    residue_templates: dict[Any, str] = {}
    # Collect SG-SG bonded residues to force CYX/NCYX/CCYX template (vs ambiguous CYM match).
    sg_sg_bonded: set[Any] = set()
    for atom_a, atom_b in topology.bonds():
        if atom_a.name == "SG" and atom_b.name == "SG":
            sg_sg_bonded.add(atom_a.residue)
            sg_sg_bonded.add(atom_b.residue)
    # Identify chain-terminal residues so we pick the right CYX variant.
    n_terminal_residues: set[Any] = set()
    c_terminal_residues: set[Any] = set()
    for chain in topology.chains():
        chain_res = list(chain.residues())
        if chain_res:
            n_terminal_residues.add(chain_res[0])
            c_terminal_residues.add(chain_res[-1])

    # Terminal stapled residues need dedicated templates because their backbone
    # atom sets differ from the internal LYZ/ADZ/GLZ residues.
    for residue in topology.residues():
        atom_names = {a.name for a in residue.atoms()}
        if residue.name in _STAPLED_NTERM_TEMPLATE and residue in n_terminal_residues and ("H2" in atom_names or "H3" in atom_names):
            custom_residue_names.add(_STAPLED_NTERM_TEMPLATE[residue.name])
        if residue.name in _STAPLED_CTERM_TEMPLATE and residue in c_terminal_residues and "OXT" in atom_names:
            custom_residue_names.add(_STAPLED_CTERM_TEMPLATE[residue.name])

    if custom_residue_names:
        extra_xml = _build_filtered_forcefield_xml(tuple(sorted(custom_residue_names)))
        force_field = openmm_app.ForceField("amber14-all.xml", extra_xml)
    else:
        force_field = openmm_app.ForceField("amber14-all.xml")

    for residue in topology.residues():
        atom_names = {a.name for a in residue.atoms()}
        override_template = template_overrides.get((str(residue.chain.id), str(residue.id)))
        if override_template:
            custom_residue_names.add(override_template)
            residue_templates[residue] = override_template
        elif residue.name in _STAPLED_NTERM_TEMPLATE and residue in n_terminal_residues and ("H2" in atom_names or "H3" in atom_names):
            residue_templates[residue] = _STAPLED_NTERM_TEMPLATE[residue.name]
        elif residue.name in _STAPLED_CTERM_TEMPLATE and residue in c_terminal_residues and "OXT" in atom_names:
            residue_templates[residue] = _STAPLED_CTERM_TEMPLATE[residue.name]
        elif residue.name in _NONTERMINAL_TEMPLATE_RESIDUES:
            residue_templates[residue] = residue.name
        elif residue.name in custom_residue_names:
            residue_templates[residue] = residue.name
        elif residue in sg_sg_bonded:
            if residue in n_terminal_residues and ("H2" in atom_names or "H3" in atom_names):
                residue_templates[residue] = "NCYX"
            elif residue in c_terminal_residues and "OXT" in atom_names:
                residue_templates[residue] = "CCYX"
            else:
                residue_templates[residue] = "CYX"
        elif (
            residue.name in _STANDARD_INTERNAL_TEMPLATES
            and residue not in n_terminal_residues
            and residue not in c_terminal_residues
        ):
            # Some passive custom templates intentionally share a standard
            # residue atom graph (for example DAR vs ARG). Pin ordinary
            # residues to their AMBER template so custom XML only applies to
            # residues explicitly replaced by the topology fixer.
            residue_templates[residue] = residue.name

    system = force_field.createSystem(
        topology,
        constraints=openmm_app.HBonds,
        ignoreExternalBonds=True,
        residueTemplates=residue_templates,
    )

    force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", float(stiffness))
    for param in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(param)
    for i, atom in enumerate(topology.atoms()):
        if atom.element.name != "hydrogen":
            force.addParticle(i, positions[i])
    system.addForce(force)

    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform_obj = openmm.Platform.getPlatformByName(platform)
    simulation = openmm_app.Simulation(topology, system, integrator, platform_obj)
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True)
    einit = state.getPotentialEnergy().value_in_unit(ENERGY)

    tol_val = tolerance.value_in_unit(unit.kilojoules_per_mole) if hasattr(tolerance, "unit") else float(tolerance)
    simulation.minimizeEnergy(
        maxIterations=int(max_iterations),
        tolerance=float(tol_val) * unit.kilojoule_per_mole / unit.nanometer,
    )

    state = simulation.context.getState(getEnergy=True, getPositions=True)
    efinal = state.getPotentialEnergy().value_in_unit(ENERGY)

    pdb_min = _write_pdb(simulation.topology, state.getPositions())
    if fix_result.connects:
        pdb_min = _add_connects(pdb_min, fix_result.connects)
    pdb_lines = pdb_min.splitlines()
    pdb_lines.insert(1, f"REMARK   1  FINAL ENERGY:   {efinal:.3f} KCAL/MOL")
    pdb_lines.insert(1, f"REMARK   1  INITIAL ENERGY: {einit:.3f} KCAL/MOL")
    pdb_min = "\n".join(pdb_lines) + "\n"

    return pdb_min, {"einit": einit, "efinal": efinal}



