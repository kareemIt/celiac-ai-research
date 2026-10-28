"""Do ChEMBL and the PubChem-derived set agree? Decides whether merging them is defensible."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data import chembl, tg2

c_raw, p_raw = chembl.load_raw(), tg2.load_raw()
cr, pr = chembl.regression(c_raw), tg2.regression(p_raw)
cc, pc = chembl.classification(c_raw), tg2.classification(p_raw)

print(f"ChEMBL : {len(cr)} molecules with exact pIC50, {len(cc)} with a class label")
print(f"PubChem: {len(pr)} molecules with exact pIC50, {len(pc)} with a class label")
print(f"\npIC50 range  ChEMBL {cr.y.min():.2f}-{cr.y.max():.2f}   PubChem {pr.y.min():.2f}-{pr.y.max():.2f}")
print(f"class balance ChEMBL {cc.y.mean():.3f} positive   PubChem {pc.y.mean():.3f} positive")

m = cr.merge(pr, on="smiles", suffixes=("_chembl", "_pubchem"))
print(f"\n== Overlap on exact pIC50: {len(m)} molecules ==")
if len(m) > 3:
    d = m.y_chembl - m.y_pubchem
    print(f"  pearson {np.corrcoef(m.y_chembl, m.y_pubchem)[0,1]:.3f} | spearman "
          f"{pd.Series(m.y_chembl).corr(pd.Series(m.y_pubchem), method='spearman'):.3f}")
    print(f"  mean difference {d.mean():+.3f} log units (sd {d.std():.3f}), median |diff| {d.abs().median():.3f}")
    print(f"  agree within 0.5 log: {(d.abs() <= 0.5).mean():.1%} | within 1.0 log: {(d.abs() <= 1).mean():.1%}")
    print(f"  worst disagreement: {d.abs().max():.2f} log units")
    print("\n  largest disagreements:")
    print(m.assign(diff=d).reindex(d.abs().sort_values(ascending=False).index)
          .head(5)[["y_chembl", "y_pubchem", "diff", "n_meas"]].round(2).to_string(index=False))

lab = cc.merge(pc, on="smiles", suffixes=("_chembl", "_pubchem"))
if len(lab) > 3:
    agree = (lab.y_chembl == lab.y_pubchem).mean()
    print(f"\n== Class-label overlap: {len(lab)} molecules, labels agree {agree:.1%} ==")
    print(pd.crosstab(lab.y_chembl, lab.y_pubchem))

new = set(cc.smiles) - set(pc.smiles)
print(f"\n== What ChEMBL adds ==")
print(f"  molecules not already in the PubChem set: {len(new)}")
print(f"  union size: {len(set(cc.smiles) | set(pc.smiles))} (PubChem alone: {len(pc)})")
addl = cc[cc.smiles.isin(new)]
print(f"  of those, positive (potent): {int(addl.y.sum())}, negative: {int((1-addl.y).sum())}")
print(f"\n  measurement support in ChEMBL: median {cr.n_meas.median():.0f} measurements, "
      f"{cr.n_docs.median():.0f} documents per molecule")
print(f"  molecules whose own replicates span >1 log unit: {int((cr.spread_log > 1).sum())} of {len(cr)}")
