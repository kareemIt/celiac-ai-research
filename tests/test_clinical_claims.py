"""Claims about the clinical status of TG2 inhibitors must stay accurate.

The paper asserted for several drafts that no TG2 inhibitor had reached the clinic. ZED1227 met
its primary endpoint against placebo in 163 patients (Schuppan et al., NEJM 2021;385:35-45), so
the claim was false and would have been caught by any reviewer in the field. These tests stop it
coming back and keep the surviving nuance -- none is approved -- intact.
"""
import glob
import io

import pytest

DOCS = sorted(glob.glob("docs/**/*.md", recursive=True))


def read(p):
    return io.open(p, encoding="utf-8").read()


@pytest.mark.parametrize("path", DOCS)
def test_no_document_claims_the_clinic_was_never_reached(path):
    t = read(path).lower()
    for bad in ("no tg2 inhibitor has reached the clinic",
                "no transglutaminase inhibitor has reached the clinic"):
        assert bad not in t, f"{path} repeats the corrected claim"


def test_paper_cites_the_trial_where_it_states_clinical_status():
    t = read("docs/paper.md")
    assert "ZED1227" in t
    assert "10.1056/NEJMoa2032441" in t
    assert "163 patients" in t


def test_the_approval_nuance_is_kept():
    """ZED1227 is not approved; saying the target is validated must not become saying it is a drug."""
    t = read("docs/paper.md")
    assert "has yet been approved" in t or "not approved" in t


def test_trial_numbers_match_the_abstract():
    """163 randomized, and the two effect sizes quoted from the primary endpoint."""
    t = read("docs/paper.md")
    for s in ("163", "0.44", "0.48", "0.15 to 0.73", "0.20 to 0.77"):
        assert s in t, f"missing {s}"
