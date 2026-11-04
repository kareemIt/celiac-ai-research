"""The grouped publication K-fold proposed in scripts/audit_replication_feasibility.py.

The audit's recommendation rests on this protocol having three properties that leave-one-
publication-out lacks: no publication straddles a fold, every molecule is tested exactly once, and
the folds are close enough in size that none dominates a pooled estimate. Those are the claims
docs/replication_feasibility.md makes, so they are pinned here rather than inspected by eye.

The runtime projection is also checked for the one thing that could silently invalidate the whole
audit: a model missing from the TG2 timings would make `sum` return NaN and quietly understate the
cost as zero hours.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.audit_replication_feasibility import SWEEP, deal_papers, grouped_folds, project


def frame(sizes, pos_frac=0.5):
    """A synthetic labelled set: `sizes` maps a paper id to its molecule count."""
    rows = []
    for paper, n in sizes.items():
        for i in range(n):
            rows.append(dict(smiles=f"C{paper}{i}", y=int(i < round(n * pos_frac)), paper=paper))
    return pd.DataFrame(rows)


def test_every_molecule_tested_exactly_once():
    df = frame({f"P{i}": 10 + i for i in range(30)})
    folds = list(grouped_folds(df, 10))
    assert sum(n_test for _, n_test, _, _ in folds) == len(df)


def test_train_and_test_partition_the_set():
    df = frame({f"P{i}": 10 + i for i in range(30)})
    for n_train, n_test, _, _ in grouped_folds(df, 10):
        assert n_train + n_test == len(df)


def test_no_publication_straddles_a_fold():
    """The property the whole experiment depends on: a training fold never sees the test paper."""
    df = frame({f"P{i}": 10 + i for i in range(30)})
    fold = deal_papers(df, 10)
    # Every molecule of a paper must carry the same fold index, so train and test share no paper.
    assert df.groupby("paper").apply(lambda g: fold[g.index].nunique()).eq(1).all()


def test_fold_assignment_is_reproducible_for_a_seed():
    df = frame({f"P{i}": 10 + i for i in range(30)})
    assert deal_papers(df, 10, seed=3).equals(deal_papers(df, 10, seed=3))


def test_the_seed_only_moves_publications_of_equal_size():
    """Largest-first dealing fixes the order whenever sizes are distinct; the seed breaks ties.

    This is why fold-assignment variance is estimable on the real targets and not on this frame:
    CDK2 has 83 publications across 32 distinct sizes and MAO-B 209 across 47, so most papers are
    tied and re-dealing with another seed moves 35% and 57% of molecules between folds.
    """
    distinct = frame({f"P{i}": 10 + i for i in range(30)})
    assert deal_papers(distinct, 10, seed=3).equals(deal_papers(distinct, 10, seed=4))

    tied = frame({f"P{i}": 10 for i in range(30)})
    assert not deal_papers(tied, 10, seed=3).equals(deal_papers(tied, 10, seed=4))


def test_folds_are_balanced_in_size():
    """Largest-first dealing should keep folds within a wide but bounded band.

    With 30 papers of 10-39 molecules into 10 folds, perfectly equal folds are impossible; the
    audit only claims no fold dominates, which is what a 2x cap on the max/min ratio encodes.
    """
    df = frame({f"P{i}": 10 + i for i in range(30)})
    sizes = [n_test for _, n_test, _, _ in grouped_folds(df, 10)]
    assert max(sizes) / min(sizes) < 2.0


def test_one_huge_publication_cannot_be_balanced_away():
    """A single paper holding most of the data makes grouped folds uneven, and should show it.

    This is TG2's situation -- its largest seven papers held 394 of 460 molecules -- and the reason
    the audit does not propose grouped folds for TG2.
    """
    df = frame({"BIG": 400, **{f"P{i}": 10 for i in range(9)}})
    sizes = [n_test for _, n_test, _, _ in grouped_folds(df, 10)]
    assert max(sizes) >= 400  # the big paper lands whole in one fold


def test_single_class_folds_are_reported():
    """A paper reporting only potent compounds must be flagged, not silently scored."""
    df = pd.concat([frame({f"P{i}": 10 for i in range(9)}, pos_frac=0.5),
                    frame({"ALLPOS": 10}, pos_frac=1.0)])
    flags = [single for _, _, single, _ in grouped_folds(df, 10)]
    assert any(flags)


def test_projection_fails_loudly_on_a_missing_model():
    """A model absent from the TG2 timings must not project as zero hours."""
    partial = pd.Series({m: 10.0 for m in SWEEP[:-1]})
    assert np.isnan(project(1000, partial, 500, 10))


def test_projection_scales_linearly_with_training_size():
    seconds = pd.Series({m: 10.0 for m in SWEEP})
    one = project(500, seconds, 500, 10)
    two = project(1000, seconds, 500, 10)
    assert two == pytest.approx(2 * one)
