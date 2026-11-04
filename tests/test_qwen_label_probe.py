"""Unit tests for the label-leakage probe's parsing and its shuffled-SMILES control.

The control is only a control if it is a different string for the same molecule, and the parser is
the only thing standing between a free-text answer and a reported correlation, so both are pinned.
"""
import pytest

from scripts.probe_qwen_labels import parse_nm, shuffled_smiles

ZED = "CC(=O)N1CCC(CC1)NC(=O)[C@@H](CC1=CC=CC=C1)NC(=O)OCC1=CC=CC=C1"


@pytest.mark.parametrize("text, nm", [
    ("IC50 = 250 nM", 250.0),
    ("IC50 = 1.5 uM", 1500.0),
    ("IC50 = 1.5 μM", 1500.0),
    ("IC50 = 0.25 mM", 250_000.0),
    ("IC50 = 500 pM", 0.5),
    ("The reported IC50 is 30 nM against TG2.", 30.0),
])
def test_parses_units(text, nm):
    assert parse_nm(text) == pytest.approx(nm)


@pytest.mark.parametrize("text", [
    "UNKNOWN", "unknown", "I do not know the published value.", None, "", "no data available",
    "IC50 = 0 nM",            # zero is not a concentration
    "Published in 2024.",     # a bare year must not become a measurement
    "IC50 = 2 M",             # no enzyme IC50 is molar; outside the plausible range
    "IC50 = 1e30 M",          # implausible, outside the enzyme range
])
def test_rejects_non_values(text):
    assert parse_nm(text) is None


def test_unknown_wins_over_a_number():
    # a hedged answer that also names a range must not be scored as a recalled value
    assert parse_nm("UNKNOWN, though such compounds are often near 100 nM") is None


def test_shuffled_smiles_is_a_different_string_for_the_same_molecule():
    from rdkit import Chem

    alt = shuffled_smiles(ZED)
    assert alt != ZED
    assert Chem.MolToSmiles(Chem.MolFromSmiles(alt)) == Chem.MolToSmiles(Chem.MolFromSmiles(ZED))


def test_shuffled_smiles_is_deterministic():
    assert shuffled_smiles(ZED) == shuffled_smiles(ZED)


@pytest.mark.parametrize("text, nm", [
    # scientific notation: the tail of the mantissa must not be read as the whole number
    ("IC50 = 1.413e+04 nM", 14130.0),
    ("IC50 = 1.5e-8 M", 15.0),
    ("IC50 = 2.5 x 10^-7 M", 250.0),
    ("IC50 = 2.5 \u00d7 10^-7 M", 250.0),
    ("IC50 = 1,000 nM", 1000.0),
    ("IC50 = 0.5 \u00b5M", 500.0),
])
def test_parses_hard_numbers(text, nm):
    assert parse_nm(text) == pytest.approx(nm)


def test_analysis_detects_leakage_when_it_is_there(tmp_path, monkeypatch, capsys):
    """Positive control: if the model returned every published value, the analysis must say so.

    A probe that cannot detect leakage in a case where leakage is total would tell us nothing when
    it reports none, so this pins the detector rather than the model.
    """
    import numpy as np
    import pandas as pd

    from scripts import probe_qwen_labels as probe
    from src.data import tg2

    truth = tg2.combined("regression").set_index("smiles")["y"]
    smiles = tg2.combined("classification")["smiles"].tolist()
    rows = []
    for s in smiles:
        t = truth.get(s, np.nan)
        # arm 1 recalls the exact published value; arm 2 always answers the same number; arm 3 abstains
        rows.append({"arm": "smiles", "smiles": s,
                     "answer": f"IC50 = {10 ** (9 - t):.6g} nM" if np.isfinite(t) else "UNKNOWN"})
        rows.append({"arm": "smiles+name", "smiles": s, "answer": "IC50 = 100 nM"})
        rows.append({"arm": "smiles_shuffled", "smiles": s, "answer": "UNKNOWN"})
    monkeypatch.setattr(probe, "RAW", str(tmp_path / "raw.csv"))
    monkeypatch.setattr(probe, "OUT", str(tmp_path / "out.csv"))
    pd.DataFrame(rows).to_csv(probe.RAW, index=False)

    probe.analyse()
    out = pd.read_csv(probe.OUT).set_index("arm")

    recalled = out.loc["smiles"]
    assert recalled["spearman"] == pytest.approx(1.0, abs=1e-6)
    assert recalled["near_exact_frac"] > 0.99
    assert recalled["near_exact_chance"] < 0.2          # the matching is not a chance artefact
    assert recalled["auroc_1um"] == pytest.approx(1.0)

    constant = out.loc["smiles+name"]
    assert constant["distinct_answers"] == 1            # degeneracy is reported, not hidden
    assert np.isnan(constant["spearman"])
    assert constant["auroc_1um"] == pytest.approx(0.5)

    assert out.loc["smiles_shuffled", "answered"] == 0
