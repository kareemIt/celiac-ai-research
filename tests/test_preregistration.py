"""A preregistered design is only preregistered if it cannot drift.

These tests pin the thresholds and disclosures that the collaboration package commits to, so that
editing the success criterion after seeing data breaks the suite rather than passing quietly. They
also check that the package keeps saying what it must: that no compounds were selected, that the
chemotype mismatch is disclosed, and that nothing claims a therapeutic effect.
"""
import io
import os

import pytest

DOCS = "docs/collaboration"
PLAN = "docs/experimental_plan.md"
FILES = ["research_summary.md", "collaboration_proposal.md", "outreach_email.md"]


def read(name):
    p = os.path.join(DOCS, name) if name in FILES else name
    if not os.path.exists(p):
        pytest.skip(f"{p} not present")
    return io.open(p, encoding="utf-8").read()


def test_package_files_exist():
    for f in FILES:
        assert os.path.exists(os.path.join(DOCS, f)), f


@pytest.mark.parametrize("doc", [PLAN, "collaboration_proposal.md"])
def test_success_threshold_is_unchanged(doc):
    """Success was preregistered at Spearman >= 0.5 and failure below 0.3."""
    t = read(doc)
    assert "0.5" in t and "0.3" in t
    assert "Spearman" in t or "spearman" in t


@pytest.mark.parametrize("doc", [PLAN, "collaboration_proposal.md"])
def test_sampling_spans_the_range_rather_than_the_top(doc):
    """The anti-cherry-picking design is the point; 4 bins of 4 across the full range."""
    t = read(doc).lower()
    assert "4 compounds" in t or "4 equal-width bins" in t or "four" in t
    assert "span" in t


@pytest.mark.parametrize("doc", FILES)
def test_no_candidate_compounds_are_named(doc):
    """The package must not contain a shortlist. ChEMBL molecule ids would be the giveaway."""
    import re
    t = read(doc)
    # Document and assay ids are fine; molecule ids are not used anywhere legitimately here.
    mol_ids = re.findall(r"CHEMBL\d{6,}", t)
    allowed = {"CHEMBL2086255", "CHEMBL5729393", "CHEMBL3351998", "CHEMBL5104118",
               "CHEMBL5154739", "CHEMBL5230175", "CHEMBL2730", "CHEMBL4530"}
    assert set(mol_ids) <= allowed, f"unexpected ChEMBL ids in {doc}: {set(mol_ids) - allowed}"


@pytest.mark.parametrize("doc", FILES)
def test_limitations_are_disclosed(doc):
    """Every outward-facing document must carry the three disclosures."""
    t = read(doc).lower()
    assert "retrospective" in t, f"{doc} must say the evidence is retrospective"
    assert "one series" in t or "second series" in t, f"{doc} must disclose the single series"
    assert "huntington" in t, f"{doc} must disclose the chemotype mismatch"


@pytest.mark.parametrize("doc", FILES)
def test_no_therapeutic_claim(doc):
    """The package must never imply a treatment."""
    t = read(doc).lower()
    for phrase in ("drug candidate for celiac", "cure", "will treat", "therapeutic candidate"):
        assert phrase not in t, f"{doc} contains a therapeutic claim: {phrase}"


def test_outreach_is_marked_unsent():
    t = read("outreach_email.md")
    assert "NOT SENT" in t
    assert "no laboratory has been contacted" in t.lower()


def test_no_invented_personal_email_addresses():
    """Only institutional/general addresses may appear; a guessed personal address would be
    both wrong and discourteous."""
    import re
    t = read("collaboration_proposal.md")
    addrs = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", t))
    assert addrs <= {"chmchair@uOttawa.ca"}, f"unverified addresses present: {addrs}"
