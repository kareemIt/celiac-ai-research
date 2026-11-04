"""The assay-format classifier and the power calculation in scripts/audit_selectivity_data.py.

The classifier decides which of the 147 pairs are commensurable, so a wrong label here changes the
feasibility verdict. Its first version matched the whole description field and labelled a 1,643-
character isopeptidase methods blob "plasma_clot", because 'plasma' appears inside a citation title
and a reagent source far below the assay title. These cases pin the title-only rule.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.audit_selectivity_data import assay_format, names_subject, power_spearman, title_of

# The real CHEMBL5729393 description, shortened but keeping the structure that broke the first rule:
# an isopeptidase title, then 'plasma' deep in a citation and a reagent source.
ZEDIRA = (
    "Isopeptidase Assay for Estimating Inhibitor Potency: FXIIIa activity has been determined "
    "using substrate A101 (Zedira GmbH, Darmstadt, Germany), which is based on the N-terminal "
    "dodecapeptide of alpha2-antipiasmin. (Oertel K et al. A highly sensitive fluorometric assay "
    "for determination of human coagulation factor XIII in plasma. Anal Biochem 2007; 367:152-8.)"
    "Briefly, 12 uL recombinant cFXIII or FXIII-A2B2 derived from human plasma (T007) was used."
)
TG2_TRANSAMIDATION = "Inhibition of human recombinant TG2 by fluorescent transamidation assay"
F13_CLOT = ("Inhibition of thrombin-activated F13-A (unknown origin) in plasma assessed as "
            "inhibition of fibrin clot formation after 7 mins by biotin incorporation assay")
TG2_BARE = "Inhibition of transglutaminase 2 (unknown origin)"


def test_isopeptidase_blob_is_not_called_plasma():
    assert assay_format(ZEDIRA) == "isopeptidase"


def test_plasma_clot_assay_is_recognised_from_its_title():
    assert assay_format(F13_CLOT) == "plasma_clot"


def test_transamidation_assay_is_amine_incorporation():
    assert assay_format(TG2_TRANSAMIDATION) == "amine_incorporation"


def test_a_bare_description_is_unspecified():
    assert assay_format(TG2_BARE) == "unspecified"


def test_a_title_naming_two_formats_is_ambiguous_not_arbitrary():
    assert assay_format("Isopeptidase and transamidation assay in plasma: methods follow") \
        == "ambiguous"


def test_title_stops_at_the_first_colon():
    assert title_of(ZEDIRA) == "Isopeptidase Assay for Estimating Inhibitor Potency"


def test_title_of_a_colonless_description_is_its_opening():
    assert title_of(TG2_BARE) == TG2_BARE


def test_a_colon_past_the_limit_does_not_truncate_the_title():
    # 'Anal Biochem 2007; 367:152' sits well past the title; the head must not be cut there
    text = "Inhibition of human TG2 by transamidation assay " + "x" * 200 + ": tail"
    assert assay_format(text) == "amine_incorporation"


@pytest.mark.parametrize("text,which,expected", [
    (TG2_TRANSAMIDATION, "tg2", True),
    # The Zedira title names the ASSAY but no enzyme, so neither target is confirmed by it. The
    # enzyme appears only in the methods text, which is the part shared by both assay records and
    # therefore cannot attribute a number to a target.
    (ZEDIRA, "tg2", False),
    (ZEDIRA, "f13", False),
    (F13_CLOT, "f13", True),
    (TG2_BARE, "tg2", True),
])
def test_subject_check_reads_the_title_only(text, which, expected):
    assert names_subject(text, which) is expected


def test_power_falls_as_the_sample_grows():
    big, small = power_spearman(69), power_spearman(500)
    assert 0 < small < big < 1
    assert np.isclose(big, 0.335, atol=0.02)      # n=69 cannot resolve a weak correlation


def test_power_at_the_textbook_case():
    # n=30, alpha=0.05, 80% power needs roughly |rho| = 0.5
    assert np.isclose(power_spearman(30), 0.5, atol=0.03)
