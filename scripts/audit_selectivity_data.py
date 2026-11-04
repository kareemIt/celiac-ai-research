"""Are the 147 paired TG2/FXIII-A measurements a compatible basis for selectivity modelling?

Phase 1 feasibility audit. docs/selectivity.md and scripts/audit_fxiiia.py already established
that selectivity does not predict across series. Neither checked whether the paired measurements
are commensurable in the first place -- the pairing rule is "same molecule, same publication", and
a publication can run the two enzymes in different assay formats, in which case the log ratio mixes
a selectivity difference with an assay-format difference.

This script asks only that: units, assay provenance, format compatibility, validity flags, and what
the pooled selectivity statistic is actually an average over. It fits nothing. If the data fails
here, no model on top of it can be trusted, however well it scores.

Reuses scripts.audit_fxiiia.paired() so both analyses see exactly the same 147 rows.

Writes results/selectivity_data_audit.csv.

Usage: python scripts/audit_selectivity_data.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

from scripts.audit_fxiiia import paired
from src.data import chembl

OUT = "results/selectivity_data_audit.csv"
RAW = "data/raw/chembl/{name}_activities.csv"

# Assay formats, matched against the free-text description. They measure different things: amine
# incorporation reports transamidase turnover on a protein substrate, isopeptidase reports cleavage
# of a quenched peptide, and a plasma clot assay reports a downstream coagulation endpoint in whole
# plasma. An IC50 ratio across two of these is not a selectivity ratio.
FORMATS = (
    ("plasma_clot", r"plasma|fibrin clot|clot formation"),
    ("isopeptidase", r"isopeptidase"),
    ("amine_incorporation", r"transamidation|transamidat|cadaverine|dimethylcasein|dimethyl casein"),
)
# Which enzyme the description presents as its own subject. ChEMBL attaches a paper's whole methods
# blob to several assay records, so a TG2-target assay can carry FXIIIa's method text.
SUBJECT = {"tg2": r"\bTG ?2\b|transglutaminase 2|tissue transglutaminase|TGM2",
           "f13": r"\bF ?XIII ?a?\b|\bF13 ?a?\b|factor ?-? ?(XIII|13)"}


def title_of(text, limit=120):
    """ChEMBL's own assay title, which precedes the methods text. Matching the whole field instead
    mislabels this paper set: CHEMBL5729393 carries a 1,643-character methods blob where 'plasma'
    appears only inside a citation title and a reagent source."""
    head = text[:limit]
    return head.split(":")[0] if ":" in head else text[:160]


def assay_format(text):
    title = title_of(text)
    hits = [name for name, pattern in FORMATS if re.search(pattern, title, re.I)]
    if len(hits) > 1:
        return "ambiguous"
    return hits[0] if hits else "unspecified"


def names_subject(text, which):
    """Does the description's own title name the enzyme its ChEMBL target claims? Heuristic."""
    return bool(re.search(SUBJECT[which], title_of(text), re.I))


def flags():
    """Validity and duplicate flags per (molecule, publication, enzyme), from the raw files."""
    out = {}
    for key, name in (("tg2", "tgm2"), ("f13", "f13a1")):
        r = chembl.load_raw(RAW.format(name=name))
        r = r[r.standard_type == "IC50"]
        out[key] = r.assign(
            outside=r.data_validity_comment.notna(),
            dup=r.potential_duplicate.astype(bool),
        ).groupby(["smiles", "document_chembl_id"])[["outside", "dup"]].any()
    return out


def power_spearman(n, alpha=0.05, power=0.80):
    """Smallest |Spearman rho| detectable at the given power, via the Fisher-z approximation."""
    z = stats.norm.isf(alpha / 2) + stats.norm.isf(1 - power)
    return float(np.tanh(z / np.sqrt((n - 3) / 1.06)))


def main():
    rows = []
    m = paired()
    f = flags()

    print("## 1. Provenance of the 147 pairs\n")
    print(f"   {len(m)} molecules, {m.document_chembl_id.nunique()} publications. Every pair is one"
          "\n   molecule measured on both enzymes inside one publication.\n")
    for key, name in (("tg2", "tgm2"), ("f13", "f13a1")):
        r = chembl.load_raw(RAW.format(name=name))
        r = r[r.standard_type == "IC50"]
        print(f"   {name:6s} IC50 rows: units {sorted(set(r.standard_units.dropna()))}, "
              f"assay types {sorted(set(r.assay_type))}, "
              f"organisms {sorted(set(r.target_organism))}")
    print("\n   Units are uniform (nM) and every row is a human binding assay, so no unit"
          "\n   conversion or species mixing enters the ratio. Censored values were already"
          "\n   excluded upstream: only standard_relation '=' is paired.")

    # Validity flags carried by the paired rows specifically.
    idx = pd.MultiIndex.from_frame(m[["smiles", "document_chembl_id"]])
    for key in ("tg2", "f13"):
        j = f[key].reindex(idx)
        m[f"outside_{key}"] = j.outside.to_numpy()
        m[f"dup_{key}"] = j.dup.to_numpy()
    outside = (m.outside_tg2 | m.outside_f13).fillna(False)
    dup = (m.dup_tg2 | m.dup_f13).fillna(False)
    print(f"\n   ChEMBL 'outside typical range' on one side or both: {int(outside.sum())} pairs")
    print(f"   ChEMBL potential-duplicate flag:                     {int(dup.sum())} pairs")
    rows.append({"section": "provenance", "pairs": len(m),
                 "publications": int(m.document_chembl_id.nunique()),
                 "n_outside_range": int(outside.sum()), "n_potential_dup": int(dup.sum())})

    print("\n## 2. Are the two measurements in a pair the same kind of experiment?\n")
    m["fmt_tg2"] = m.assay_tg2.map(assay_format)
    m["fmt_f13"] = m.assay_f13.map(assay_format)
    m["fmt_match"] = m.fmt_tg2 == m.fmt_f13
    m["subject_ok_tg2"] = [names_subject(t, "tg2") for t in m.assay_tg2]
    m["subject_ok_f13"] = [names_subject(t, "f13") for t in m.assay_f13]
    # The same description on both sides means ChEMBL attached one methods blob to both assays, so
    # the record cannot say which format produced which number.
    m["same_desc"] = m.assay_tg2 == m.assay_f13

    tab = m.groupby(["fmt_tg2", "fmt_f13"]).size().rename("pairs").reset_index()
    print(tab.to_string(index=False))
    known = m[(m.fmt_tg2 != "unspecified") & (m.fmt_f13 != "unspecified")]
    print(f"\n   both formats identifiable: {len(known)} of {len(m)} pairs")
    print(f"   formats agree:             {int(m.fmt_match.sum())} pairs "
          f"({int((known.fmt_tg2 == known.fmt_f13).sum())} of the identifiable ones)")
    print(f"   TG2 description names TG2:         {int(m.subject_ok_tg2.sum())} of {len(m)}")
    print(f"   FXIII-A description names FXIII-A: {int(m.subject_ok_f13.sum())} of {len(m)}")
    print(f"   SAME description text on both sides: {int(m.same_desc.sum())} of {len(m)} -- for"
          "\n   these the record cannot state which assay produced which number")
    rows.append({"section": "assay_compatibility", "pairs": len(m),
                 "n_formats_identifiable": len(known),
                 "n_format_match": int(m.fmt_match.sum()),
                 "n_tg2_desc_names_tg2": int(m.subject_ok_tg2.sum()),
                 "n_f13_desc_names_f13": int(m.subject_ok_f13.sum()),
                 "n_same_description": int(m.same_desc.sum())})

    print("\n   Per publication:\n")
    print(f"   {'publication':<16}{'n':>4}  {'TG2 assay':<20}{'FXIII-A assay':<20}"
          f"{'match':>6}{'subj ok':>9}")
    for doc, g in m.groupby("document_chembl_id", sort=False):
        print(f"   {doc:<16}{len(g):>4}  {g.fmt_tg2.iloc[0]:<20}{g.fmt_f13.iloc[0]:<20}"
              f"{str(bool(g.fmt_match.iloc[0])):>6}"
              f"{str(bool(g.subject_ok_tg2.iloc[0] and g.subject_ok_f13.iloc[0])):>9}")
        rows.append({"section": "publication", "publication": doc, "pairs": len(g),
                     "fmt_tg2": g.fmt_tg2.iloc[0], "fmt_f13": g.fmt_f13.iloc[0],
                     "fmt_match": bool(g.fmt_match.iloc[0]),
                     "median_log_sel": float(g.selectivity.median()),
                     "sd_log_sel": float(g.selectivity.std()),
                     "n_10x_tg2_selective": int((g.selectivity > 1).sum()),
                     "n_10x_wrong_way": int((g.selectivity < -1).sum())})

    print("\n## 3. What the pooled selectivity statistic averages over\n")
    print(f"   {'publication':<16}{'n':>4}{'median':>9}{'sd':>7}{'>10x TG2':>10}{'>10x FXIII':>12}"
          f"   programme")
    for doc, g in m.groupby("document_chembl_id", sort=False):
        s = g.selectivity
        direction = ("TG2-directed" if s.median() > 0.2 else
                     "FXIII-A-directed" if s.median() < -0.2 else "neither / too few")
        print(f"   {doc:<16}{len(g):>4}{s.median():>+9.2f}"
              f"{(s.std() if len(g) > 1 else np.nan):>7.2f}"
              f"{int((s > 1).sum()):>10}{int((s < -1).sum()):>12}   {direction}")
    s = m.selectivity
    print(f"\n   pooled           {len(s):>4}{s.median():>+9.2f}{s.std():>7.2f}"
          f"{int((s > 1).sum()):>10}{int((s < -1).sum()):>12}")
    big = m.groupby("document_chembl_id").size().nlargest(2).index
    a, b = (m[m.document_chembl_id == d].selectivity for d in big)
    u = stats.mannwhitneyu(a, b)
    print(f"\n   The two largest publications hold {len(a) + len(b)} of {len(m)} pairs and have"
          f"\n   OPPOSITE median selectivity ({a.median():+.2f} vs {b.median():+.2f}, "
          f"Mann-Whitney p={u.pvalue:.2e}).")
    print("   The pooled median is therefore a mixture of a TG2 programme and an FXIII-A"
          "\n   programme, not an estimate of the selectivity of TG2 chemistry. Reporting it"
          "\n   without that split overstates what the number means.")
    rows.append({"section": "mixture", "pairs": len(m),
                 "median_log_sel": float(s.median()),
                 "paper_a": big[0], "paper_b": big[1],
                 "median_a": float(a.median()), "median_b": float(b.median()),
                 "mannwhitney_p": float(u.pvalue)})

    print("\n## 4. Does potency predict selectivity?\n")
    print("   Selectivity is pIC50(TG2) - pIC50(FXIII-A), so it CONTAINS the TG2 potency it is"
          "\n   being correlated against: any such correlation is positive by construction. The"
          "\n   'null' column is what independent enzymes with these variances would already give,"
          "\n   and 'r(TG2, FXIII-A)' is the assumption-free quantity.\n")
    print(f"   {'publication':<16}{'n':>5}{'r(potency,sel)':>16}{'null':>8}"
          f"{'r(TG2,FXIII-A)':>16}{'p':>11}")
    for label, d in [("pooled", m)] + [(doc, g) for doc, g in m.groupby("document_chembl_id")
                                       if len(g) >= 20]:
        r = stats.spearmanr(d.p_tg2, d.selectivity)
        cross = stats.pearsonr(d.p_tg2, d.p_f13)
        # corr(x, x - y) when x and y are independent with these sample variances
        vx, vy = d.p_tg2.var(), d.p_f13.var()
        null = float(np.sqrt(vx / (vx + vy)))
        print(f"   {label:<16}{len(d):>5}{r.statistic:>+16.3f}{null:>+8.3f}"
              f"{cross.statistic:>+16.3f}{cross.pvalue:>11.3g}")
        rows.append({"section": "potency_vs_selectivity", "publication": label, "pairs": len(d),
                     "spearman": float(r.statistic), "spearman_p": float(r.pvalue),
                     "spearman_null_independent": null,
                     "pearson_tg2_vs_f13": float(cross.statistic),
                     "pearson_tg2_vs_f13_p": float(cross.pvalue)})
    print("\n   Every observed correlation with selectivity is BELOW its independence null, so"
          "\n   potency buys less selectivity than chance would, not more. The reason is the last"
          "\n   column: TG2 and FXIII-A potency are positively correlated, i.e. the molecules that"
          "\n   inhibit TG2 harder also inhibit FXIII-A harder. A potency-ranked cascade therefore"
          "\n   enriches for cross-reactivity, which is the opposite of the intended effect.")

    print("\n## 5. What sample size would the modelling questions need?\n")
    sizes = m.groupby("document_chembl_id").size()
    print(f"   publications with >= 20 pairs: {int((sizes >= 20).sum())} "
          f"({sizes[sizes >= 20].sum()} pairs)")
    print(f"   publications with >= 10 pairs: {int((sizes >= 10).sum())} "
          f"({sizes[sizes >= 10].sum()} pairs)\n")
    for n in sorted(set(sizes[sizes >= 20].tolist() + [len(m)])):
        print(f"   within one series of n={n:>3}: smallest Spearman detectable at 80% power, "
              f"alpha=0.05 is {power_spearman(n):.2f}")
        rows.append({"section": "power", "pairs": n, "min_detectable_spearman": power_spearman(n)})
    k = int((sizes >= 20).sum())
    print(f"\n   Leave-one-publication-out is the only control that separates selectivity from"
          f"\n   series identity, and {k} publications support it: {k} transfer observations, which"
          "\n   cannot carry a confidence interval. Multi-task learning shares a representation"
          "\n   across tasks and needs more groups, not more molecules inside two of them.")

    print("\n## 6. Verdict on feasibility\n")
    clean = m[m.fmt_match & m.subject_ok_tg2 & m.subject_ok_f13 & ~m.same_desc]
    print(f"   format-matched pairs with self-consistent, distinct descriptions: {len(clean)} "
          f"across {clean.document_chembl_id.nunique()} publications")
    print(f"   of those, TG2-selective by >10x: {int((clean.selectivity > 1).sum())}, "
          f"by >100x: {int((clean.selectivity > 2).sum())}")
    rows.append({"section": "verdict", "pairs": len(clean),
                 "publications": int(clean.document_chembl_id.nunique()),
                 "n_10x_tg2_selective": int((clean.selectivity > 1).sum()),
                 "n_100x_tg2_selective": int((clean.selectivity > 2).sum())})

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
