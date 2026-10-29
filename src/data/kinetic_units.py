"""Units recovered from primary publications for ChEMBL kinetic rows that carry none.

docs/covalent.md dropped 118 kinact/Ki rows because ChEMBL stores them without a unit, and a
second-order rate constant cannot be placed on a common scale without one. Guessing the convention
from the magnitude would be fabricating data, so those rows stayed out.

This module records units read from the primary literature instead. A document is listed only when
the paper states its unit explicitly AND the stored values fall on that scale; both conditions are
recorded per entry so a reader can check the claim. Anything else stays excluded.

Nothing here converts a value. The entries supply the missing `standard_units` string using
ChEMBL's own vocabulary, after which the normal conversion in scripts/run_covalent.py applies.
"""

# document -> (ChEMBL unit string, citation, evidence for the assignment)
RECOVERED = {
    "CHEMBL3351998": (
        "10^3/M/min",
        "Klock C, Herrera Z, Albertelli M, Khosla C. Discovery of potent and specific "
        "dihydroisoxazole inhibitors of human transglutaminase 2. J Med Chem 2014;57(21):9042-9064. "
        "doi:10.1021/jm501145a (open access, PMC4234452).",
        "The paper reports k_inh/K_i in mM^-1 min^-1, stating ERW1041E's potency against human TG2 "
        "as 17 mM^-1 min^-1. 1 mM^-1 min^-1 = 10^3 M^-1 min^-1, which is ChEMBL's '10^3/M/min'. "
        "The 45 stored values for this document span 1.2 to 108 with a median of 10.4, matching "
        "that scale and the published figure; rows under the explicit '10'3/M/min' convention "
        "elsewhere in the same file span 13.2 to 130. Assignment is at document level from the "
        "stated unit plus the scale match, not verified row by row -- the paper's tables are "
        "embedded as images.",
    ),
}

# Documents examined and deliberately NOT recovered, so the decision is visible rather than an
# omission a later reader has to rediscover.
REJECTED = {
    "CHEMBL2086255": (
        "Prime ME, Andersen OA, Barker JJ, et al. Discovery and structure-activity relationship of "
        "potent and selective covalent inhibitors of transglutaminase 2 for Huntington's disease. "
        "J Med Chem 2012;55(3):1021-1046. doi:10.1021/jm201310y (PMID 22224594).",
        "55 molecules, the largest single block of unitless rows. The unit could not be confirmed: "
        "the article is not open access, and a later paper from the same group quotes k_inact/K_I "
        "in M^-1 min^-1 while the stored values here (0.29 to 1460, median 80.5) would fit either "
        "M^-1 min^-1 or mM^-1 min^-1 depending on the series. A three-order-of-magnitude error "
        "would be invisible in aggregate and corrupt every model trained on it, so these rows stay "
        "excluded until someone reads the paper. All 55 molecules already carry an IC50 label, so "
        "the cost is kinetic labels for known chemistry rather than lost molecules.",
    ),
}


def apply(df):
    """Fill `standard_units` for rows whose document has a recovered unit. Returns a copy."""
    out = df.copy()
    for doc, (unit, _, _) in RECOVERED.items():
        fill = (out["document_chembl_id"] == doc) & out["standard_units"].isna()
        out.loc[fill, "standard_units"] = unit
    return out


def provenance():
    """One row per decision, for the audit table."""
    rows = [{"document": d, "status": "recovered", "unit": u, "citation": c, "evidence": e}
            for d, (u, c, e) in RECOVERED.items()]
    rows += [{"document": d, "status": "rejected", "unit": None, "citation": c, "evidence": e}
             for d, (c, e) in REJECTED.items()]
    return rows
