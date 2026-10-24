import torch
#from graphein.protein.resi_atoms import ATOM_NUMBERING

from partiflow.openfold.np.residue_constants import atom_types

# PDB and OpenFold have different atom ordering, these utils convert between the two
# PDB ordering: https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf
# OpenFold ordering: https://github.com/aqlaboratory/openfold/blob/f6c875b3c8e3e873a932cbe3b31f94ae011f6fd4/openfold/np/residue_constants.py#L556

ATOM_NUMBERING = {
    'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG': 5, 'CG': 6, 
    'CD1': 7, 'CD2': 8, 'CE1': 9, 'CE2': 10, 'CZ': 11, 'OD1': 12, 'ND2': 13, 
    'CG1': 14, 'CG2': 15, 'CD': 16, 'CE': 17, 'NZ': 18, 'OD2': 19, 'OE1': 20, 
    'NE2': 21, 'OE2': 22, 'OH': 23, 'NE': 24, 'NH1': 25, 'NH2': 26, 'OG1': 27, 
    'SD': 28, 'ND1': 29, 'SG': 30, 'NE1': 31, 'CE3': 32, 'CZ2': 33, 'CZ3': 34, 
    'CH2': 35, 'OXT': 36
}

PDB_TO_OPENFOLD_INDEX_TENSOR = torch.tensor(
    [ATOM_NUMBERING[atom] for atom in atom_types]
)

OPENFOLD_TO_PDB_INDEX_TENSOR = torch.tensor(
    [atom_types.index(atom) for atom in ATOM_NUMBERING]
)

AA_CHARACTER_PROTORP = {
    "ALA": "A",
    "CYS": "P",
    "GLU": "C",
    "ASP": "C",
    "GLY": "A",
    "PHE": "A",
    "ILE": "A",
    "HIS": "P",
    "LYS": "C",
    "MET": "A",
    "LEU": "A",
    "ASN": "P",
    "GLN": "P",
    "PRO": "A",
    "SER": "P",
    "ARG": "C",
    "THR": "P",
    "TRP": "P",
    "VAL": "A",
    "TYR": "P",
}

SIDECHAIN_TIP_ATOMS = {
    "ALA": ["CA", "CB"],
    "ARG": ["CD", "CZ", "NE", "NH1", "NH2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "ASN": ["CB", "CG", "ND2", "OD1"],
    "CYS": ["CA", "CB", "SG"],
    "GLU": ["CG", "CD", "OE1", "OE2"],
    "GLN": ["CG", "CD", "NE2", "OE1"],
    "GLY": [],
    "HIS": ["CB", "CG", "CD2", "CE1", "ND1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CE", "NZ"],
    "MET": ["CG", "CE", "SD"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CA", "CB", "CG", "CD", "N"],
    "SER": ["CA", "CB", "OG"],
    "THR": ["CA", "CB", "CG2", "OG1"],
    "TRP": ["CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}
