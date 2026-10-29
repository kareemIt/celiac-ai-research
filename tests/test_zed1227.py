"""ZED1227 is in our training data, so the one thing that must never happen is a prediction for it
being presented as validation. These tests pin the structure, the membership finding, and the
refusal.
"""
import io
import os

import pytest
from rdkit import Chem

from scripts.analyse_zed1227 import ZED1227
from src.data import tg2

DOC = "docs/zed1227_analysis.md"


def test_structure_parses_and_matches_published_formula():
    m = Chem.MolFromSmiles(ZED1227)
    assert m is not None
    from rdkit.Chem import rdMolDescriptors
    assert rdMolDescriptors.CalcMolFormula(m) == "C26H36N6O6"


def test_warhead_is_an_unsaturated_ester_not_an_acrylamide():
    """The distinction matters: our kinetic data is overwhelmingly acrylamides."""
    m = Chem.MolFromSmiles(ZED1227)
    assert m.HasSubstructMatch(Chem.MolFromSmarts("C=CC(=O)O"))
    assert not m.HasSubstructMatch(Chem.MolFromSmarts("C=CC(=O)N"))


def test_zed1227_is_in_the_tg2_training_set():
    """If this ever becomes False the leakage warning in the report must be revisited."""
    want = Chem.MolToSmiles(Chem.MolFromSmiles(ZED1227))
    have = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in tg2.combined("classification").smiles}
    assert want in have


def test_report_refuses_to_predict_and_says_why():
    if not os.path.exists(DOC):
        pytest.skip("report not present")
    t = io.open(DOC, encoding="utf-8").read()
    assert "No prediction was made for ZED1227" in t
    assert "memorisation" in t


def test_report_separates_evidence_from_inference():
    if not os.path.exists(DOC):
        pytest.skip("report not present")
    t = io.open(DOC, encoding="utf-8").read()
    for tag in ("**[E]**", "**[I]**", "**[U]**"):
        assert tag in t, f"missing evidence tag {tag}"


def test_report_does_not_assert_the_phase2b_endpoint_failed():
    """We could not reach a primary source; the report must not overstate it."""
    if not os.path.exists(DOC):
        pytest.skip("report not present")
    t = io.open(DOC, encoding="utf-8").read()
    assert "cannot confirm the characterisation" in t
