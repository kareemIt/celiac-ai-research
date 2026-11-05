"""The aggregation analysis in scripts/analyse_protocol_gap.py.

The script's claim is that leave-one-publication-out and grouped K-fold never disagreed about the
models -- the 0.17 gap on CDK2 is the unit the AUROC is averaged over. Three things have to be
right for that: pooling must be refused where a molecule is tested more than once, a macro average
over small groups must actually come out below a pooled score when the groups are homogeneous, and
the matched comparison must pair publications rather than fold indices.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyse_protocol_gap import (one_row_per_molecule, per_publication_auroc,
                                          pooled_auroc, within_publication_macro)
from scripts.analyse_protocol_gap import tested_once as each_molecule_once


def preds(rows, model="rf"):
    """rows: (smiles, y, p, paper)."""
    return pd.DataFrame([{"model": model, "smiles": s, "y": y, "p": p, "paper": q}
                         for s, y, p, q in rows])


def series(paper, base, n=10):
    """One publication: a congeneric series whose activities sit in a narrow band around `base`,
    ranked correctly inside the series but offset from other series."""
    rows = []
    for i in range(n):
        y = int(i >= n // 2)
        rows.append((f"{paper}_{i}", y, base + 0.01 * i, paper))
    return rows


def test_tested_once_is_true_for_a_publication_arm():
    p = preds(series("P1", 0.2) + series("P2", 0.6))
    assert each_molecule_once(p)


def test_tested_once_is_false_when_a_molecule_repeats():
    """The scaffold arm's repeated seeds: pooling these would mix correlated predictions."""
    p = preds(series("P1", 0.2))
    assert not each_molecule_once(pd.concat([p, p.iloc[:1]], ignore_index=True))


def test_tested_once_is_false_on_an_empty_frame():
    assert not each_molecule_once(preds([]))


def test_macro_over_homogeneous_publications_falls_below_the_pooled_score():
    """The mechanism the script claims. Each publication is a congeneric series that is mostly
    potent or mostly inactive, and the model tracks that series-level difference. Globally the
    probabilities separate the labels well; inside a series they barely rank at all."""
    rows = []
    for k, (base, prevalence) in enumerate(((0.1, 0.1), (0.5, 0.5), (0.9, 0.9))):
        for i in range(10):
            y = int(i < prevalence * 10)
            # the probability says which SERIES this is, and almost nothing about the molecule
            rows.append((f"P{k}_{i}", y, base + 0.001 * (i % 2), f"P{k}"))
    p = preds(rows)
    pooled = pooled_auroc(p)["rf"]
    macro = within_publication_macro(p, min_per_paper=5)["rf"]["within_pub_macro"]
    assert macro < pooled - 0.1


def test_macro_equals_pooled_when_every_publication_spans_the_same_range():
    """No between-series offset to exploit, so the two aggregations agree."""
    rows = []
    for k in range(4):
        for i in range(10):
            rows.append((f"P{k}_{i}", int(i >= 5), 0.05 + 0.1 * i, f"P{k}"))
    p = preds(rows)
    pooled = pooled_auroc(p)["rf"]
    macro = within_publication_macro(p, min_per_paper=5)["rf"]["within_pub_macro"]
    assert np.isclose(pooled, 1.0) and np.isclose(macro, 1.0)


def test_single_class_publications_are_not_scored():
    one_class = [(f"Q_{i}", 1, 0.6, "Q") for i in range(10)]
    p = preds(series("P1", 0.2) + series("P2", 0.6) + one_class)
    assert per_publication_auroc(p, min_per_paper=5)["rf"].keys() == {"P1", "P2"}


def test_publications_below_the_size_floor_are_not_scored():
    p = preds(series("P1", 0.2, n=10) + series("P2", 0.6, n=4))
    assert set(per_publication_auroc(p, min_per_paper=10)["rf"]) == {"P1"}


def test_within_publication_macro_needs_three_publications():
    assert within_publication_macro(preds(series("P1", 0.2)), min_per_paper=5) == {}


def test_one_row_per_molecule_averages_repeated_tests():
    p = preds([("A", 1, 0.2, "P1"), ("A", 1, 0.8, "P1"), ("B", 0, 0.4, "P1")])
    out = one_row_per_molecule(p).set_index("smiles")
    assert np.isclose(out.loc["A", "p"], 0.5)
    assert out.loc["A", "paper"] == "P1"
    assert len(out) == 2


def test_pooled_auroc_refuses_a_single_class_model():
    p = preds([(f"A{i}", 1, 0.5, "P1") for i in range(20)])
    assert "rf" not in pooled_auroc(p)


def test_pooled_auroc_can_be_restricted_to_a_molecule_subset():
    p = preds(series("P1", 0.2) + series("P2", 0.6))
    keep = {s for s, *_ in series("P1", 0.2)}
    assert "rf" in pooled_auroc(p, keep=keep)
    assert pooled_auroc(p, keep=keep)["rf"] != pooled_auroc(p)["rf"]
