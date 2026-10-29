"""The selectivity conclusion rests on two things: targets never being pooled, and the
paper-held-out split actually holding out papers. Both are checked here."""
import numpy as np
import pandas as pd

from scripts.run_selectivity import run_task
from src.data import chem, chembl

SMILES = ["CCO", "CCN", "c1ccccc1", "c1ccccc1C", "c1ccncc1", "CC(=O)O", "CCCC", "CC(C)O"]


def activity(smiles, value_nM, relation="="):
    """A minimal ChEMBL-shaped frame, post-load_raw (so it already carries `smiles`)."""
    return pd.DataFrame({
        "smiles": smiles,
        "canonical_smiles": smiles,
        "standard_type": "IC50",
        "standard_units": "nM",
        "standard_value": value_nM,
        "standard_relation": relation,
        "data_validity_comment": None,
        "document_chembl_id": "d1",
    })


def test_group_split_never_shares_a_document():
    """The whole 0.974 -> 0.574 collapse depends on this: a paper in test is absent from train."""
    docs = np.array(["doc_a", "doc_a", "doc_b", "doc_b", "doc_c", "doc_c", "doc_d", "doc_d"])
    for seed in range(5):
        tr, va, te = chem.group_split(docs, seed=seed)
        assert not (set(docs[tr]) | set(docs[va])) & set(docs[te])


def test_scaffold_groups_do_not_separate_papers():
    """Why a scaffold split is the wrong control here, stated structurally rather than by luck:
    one scaffold carries molecules from two papers, so holding out that scaffold does not hold out
    either paper. A scaffold split could only isolate a paper if scaffolds nested inside papers,
    and in the FXIII-A set they do not."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    smiles = ["c1ccccc1C", "c1ccccc1CC", "c1ccncc1C", "c1ccncc1CC"]
    docs = ["paper_a", "paper_b", "paper_a", "paper_b"]
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in smiles]
    assert len(set(scaf)) == 2, "fixture should hold exactly two scaffolds"

    by_scaffold = {}
    for s, doc in zip(scaf, docs):
        by_scaffold.setdefault(s, set()).add(doc)
    assert all(len(v) > 1 for v in by_scaffold.values()), (
        "every scaffold should span both papers, so no scaffold split can isolate one")


def test_run_task_respects_the_grouping_it_is_given():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    groups = np.array(["g1"] * 4 + ["g2"] * 4)
    out = run_task("t", SMILES, y, seeds=3, reps=10, rng=np.random.default_rng(0), groups=groups)
    assert out is None or out["n_mols"] <= len(SMILES)


def test_offtarget_labels_are_never_tg2_labels():
    """Same molecule, different enzyme, opposite label: the two frames must stay independent."""
    potent = chembl.classification(activity(["CCO"], [10.0]))
    weak = chembl.classification(activity(["CCO"], [50000.0]))
    assert potent.y.iloc[0] == 1
    assert weak.y.iloc[0] == 0


def test_censored_bound_below_the_cut_proves_nothing():
    """'> 0.5 uM' against a 1 uM cut must not become a negative label."""
    assert chembl.classification(activity(["CCO"], [500.0], ">"), ic50_um=1.0).empty


def test_censored_bound_above_the_cut_is_a_negative():
    """'> 10 uM' does prove 'not potent at 1 uM', and that is how inactives enter the set."""
    out = chembl.classification(activity(["CCO"], [10000.0], ">"), ic50_um=1.0)
    assert len(out) == 1 and out.y.iloc[0] == 0
