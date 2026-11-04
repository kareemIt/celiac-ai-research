"""The nesting and overlap measures in scripts/audit_assay_confound.py.

The script's conclusion -- that an assay-held-out control cannot be built on ChEMBL-derived data
because assays are nested inside publications -- rests entirely on these three counts. A sign error
or a groupby on the wrong key would turn "not identifiable" into a fabricated control, so each
measure is pinned against a frame whose structure is known by construction.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.audit_assay_confound import nesting, within_paper_contrast


def frame(rows):
    """rows: (paper, assay, n_molecules)."""
    out = []
    for paper, assay, n in rows:
        for i in range(n):
            out.append(dict(smiles=f"C{paper}{assay}{i}", y=i % 2, paper=paper, assay=assay))
    return pd.DataFrame(out)


def test_perfect_nesting_is_reported_as_zero_crossing_assays():
    """One assay per publication: the real CDK2 and MAO-B structure."""
    m = frame([("P1", "A1", 10), ("P2", "A2", 10), ("P3", "A3", 10)])
    f = nesting(m)
    assert f["assays_spanning_multiple_papers"] == 0
    assert f["papers_with_multiple_assays"] == 0


def test_an_assay_shared_between_publications_is_detected():
    """If this ever became non-zero in real data, the control would become identifiable."""
    m = frame([("P1", "SHARED", 10), ("P2", "SHARED", 10), ("P3", "A3", 10)])
    assert nesting(m)["assays_spanning_multiple_papers"] == 1


def test_a_publication_running_two_assays_is_detected():
    m = frame([("P1", "A1", 7), ("P1", "A2", 8), ("P2", "A3", 10)])
    f = nesting(m)
    assert f["papers_with_multiple_assays"] == 1
    assert f["molecules_in_multi_assay_papers"] == 15


def test_within_paper_contrast_requires_two_assays_above_the_floor():
    """A publication whose second assay holds one molecule gives no usable contrast."""
    m = frame([("P1", "A1", 20), ("P1", "A2", 1), ("P2", "A3", 10)])
    assert within_paper_contrast(m, 5)["papers_with_2plus_assays_of_min_size"] == 0

    m2 = frame([("P1", "A1", 20), ("P1", "A2", 6), ("P2", "A3", 10)])
    f = within_paper_contrast(m2, 5)
    assert f["papers_with_2plus_assays_of_min_size"] == 1
    assert f["molecules_in_those_papers"] == 26


def test_min_per_assay_is_an_inclusive_floor():
    m = frame([("P1", "A1", 5), ("P1", "A2", 5)])
    assert within_paper_contrast(m, 5)["papers_with_2plus_assays_of_min_size"] == 1
    assert within_paper_contrast(m, 6)["papers_with_2plus_assays_of_min_size"] == 0
