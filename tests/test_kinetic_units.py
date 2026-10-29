"""Unit recovery writes units that were never in the source data, so it is the part of this
project most able to corrupt results silently. A wrong factor of 1000 would be invisible."""
import pandas as pd
import pytest
from rdkit import Chem

from scripts.audit_covalent import WARHEADS
from scripts.run_covalent import TO_PER_M_S
from src.data import kinetic_units


def rows(doc, units=None, value=10.0):
    return pd.DataFrame({
        "document_chembl_id": [doc],
        "standard_units": [units],
        "standard_value": [value],
        "standard_type": ["Ratio"],
    })


def test_recovered_units_are_convertible():
    """A recovered unit that the converter does not know would silently drop the rows again."""
    for doc, (unit, _, _) in kinetic_units.RECOVERED.items():
        assert unit in TO_PER_M_S, f"{doc} recovered to {unit}, which has no conversion"


def test_recovery_fills_only_missing_units():
    out = kinetic_units.apply(rows("CHEMBL3351998", units=None))
    assert out.standard_units.iloc[0] == "10^3/M/min"


def test_recovery_never_overwrites_a_stated_unit():
    """ChEMBL's own unit always wins; recovery is for absence, not disagreement."""
    out = kinetic_units.apply(rows("CHEMBL3351998", units="/M/s"))
    assert out.standard_units.iloc[0] == "/M/s"


def test_unlisted_documents_are_left_alone():
    """Only documents whose unit was read from the paper are touched."""
    out = kinetic_units.apply(rows("CHEMBL2086255", units=None))
    assert out.standard_units.isna().all()


def test_the_rejected_document_is_not_recovered():
    """Prime 2012 is the largest block and deliberately excluded; a later edit must not quietly
    add it without reading the paper."""
    assert "CHEMBL2086255" in kinetic_units.REJECTED
    assert "CHEMBL2086255" not in kinetic_units.RECOVERED


def test_every_entry_carries_a_citation_and_evidence():
    for doc, (unit, citation, evidence) in kinetic_units.RECOVERED.items():
        assert "doi" in citation.lower() or "J Med Chem" in citation, doc
        assert len(evidence) > 100, f"{doc} needs evidence a reader can check"


def test_mM_per_min_is_a_thousand_M_per_min():
    """The published unit is mM^-1 min^-1; the mapping treats it as 10^3 M^-1 min^-1. Getting this
    wrong by 1000x would shift one paper's intercept invisibly."""
    assert TO_PER_M_S["10^3/M/min"] == pytest.approx(1000 / 60)


def test_dihydroisoxazole_pattern_matches_a_real_one():
    """The original SMARTS was a four-membered ring and matched none of the 48 Klock compounds."""
    erw_like = "O=C(NC[C@@H]1CC(Br)=NO1)C"  # 3-bromo-4,5-dihydroisoxazole fragment
    q = Chem.MolFromSmarts(WARHEADS["dihydroisoxazole"])
    assert q is not None
    assert Chem.MolFromSmiles(erw_like).HasSubstructMatch(q)
    # and does not fire on an acrylamide
    assert not Chem.MolFromSmiles("C=CC(=O)NC").HasSubstructMatch(q)
