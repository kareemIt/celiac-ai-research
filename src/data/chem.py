"""Shared molecule utilities: canonical SMILES, fingerprints, scaffold and group splits."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def canonical(smiles):
    """Canonical SMILES of the largest fragment. 18 TG2 entries are salts, whose counterions would
    otherwise set fingerprint bits and shift the Murcko scaffold the split groups on."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    return Chem.MolToSmiles(max(frags, key=lambda m: m.GetNumHeavyAtoms()))


def morgan(smiles_list, radius=2, n_bits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles_list]).astype(np.uint8)


def group_split(groups, frac=(0.8, 0.1, 0.1), seed=0):
    """Assign whole groups to train/valid/test so no group crosses splits. Returns index arrays.
    Groups are shuffled, then filled by row count (not group count), so split sizes stay near `frac`
    even when a few groups are large (e.g. a big analog series)."""
    groups = np.asarray(groups)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(uniq))
    start = np.empty(len(uniq))
    start[order] = np.cumsum(counts[order]) - counts[order]  # rows already placed before each group
    n = len(groups)
    part = np.where(start < frac[0] * n, 0, np.where(start < (frac[0] + frac[1]) * n, 1, 2))
    which = part[inv]
    return tuple(np.where(which == i)[0] for i in range(3))


def scaffold_split(smiles, frac=(0.8, 0.1, 0.1), seed=0):
    """Bemis-Murcko scaffold split: analog series stay together, so test measures generalization to new chemotypes."""
    return group_split([MurckoScaffold.MurckoScaffoldSmiles(s) for s in smiles], frac, seed)
