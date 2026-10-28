import os

import numpy as np
import pandas as pd
import pytest

from src.data import chem, tg2


def test_group_split_keeps_groups_disjoint():
    groups = np.repeat(np.arange(50), 3)
    tr, va, te = chem.group_split(groups)
    sets = [set(groups[i]) for i in (tr, va, te)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(tr) + len(va) + len(te) == len(groups)


def test_group_split_sizes_follow_rows_not_groups():
    groups = np.concatenate([np.zeros(60), np.arange(1, 41)])  # one group holds 60% of rows
    for seed in range(10):
        tr, va, te = chem.group_split(groups, seed=seed)
        assert len(te) <= 10  # test only gets groups that start past the 90% mark, so it never swallows the big group


def test_scaffold_split_puts_analogs_together():
    smi = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1O", "C1CCNCC1C", "C1CCNCC1O", "CCO"]
    parts = chem.scaffold_split(smi, frac=(0.34, 0.33, 0.33))
    where = {i: k for k, idx in enumerate(parts) for i in idx}
    assert where[0] == where[1] == where[2]  # benzene scaffold
    assert where[3] == where[4]  # piperidine scaffold


def test_tg2_labels():
    df = pd.DataFrame({
        "Smiles": ["CCO", "OCC", "CCN", "CCC", "CCCl"],
        "Type": ["IC50", "IC50", "IC50", "IC50", "IC50"],
        "Value": [1.0, 100.0, 10.0, 0.1, 50.0],
        "Acqual": ["=", "=", ">", "=", "="],
        "Activity": ["Active", "Inactive", "Inactive", "Active", "Unspecified"],
    })
    d = df.assign(smiles=df.Smiles.map(chem.canonical))
    cls = dict(tg2.classification(d).values)
    assert cls["CCO"] == 0  # CCO/OCC are one molecule: median pIC50 of 6 and 4 is 5 (10 uM), not potent
    assert cls["CCC"] == 1 and cls["CCCl"] == 0  # 0.1 uM potent, 50 uM not
    assert cls["CCN"] == 0  # '>10 uM' has no pIC50 but still proves "not potent at 1 uM"
    assert "CCN" not in dict(tg2.classification(d, ic50_um=100.0).values)  # bound below the cut proves nothing
    assert dict(tg2.classification(d.assign(Value=[1.0] * 5)).values)["CCC"] == 1  # exactly 1 uM counts as potent
    reg = dict(tg2.regression(d).values)
    assert reg["CCC"] == pytest.approx(7.0)  # 0.1 uM -> pIC50 7
    assert "CCN" not in reg  # censored '>' excluded


def test_morgan_shape():
    assert chem.morgan(["CCO", "c1ccccc1"], n_bits=256).shape == (2, 256)


@pytest.mark.skipif(not os.path.exists(tg2.RAW), reason="run scripts/download_data.sh first")
def test_tg2_real_data_loads():
    d = tg2.load_raw()
    c, r = tg2.classification(d), tg2.regression(d)
    assert len(c) > len(r) > 300  # classification also keeps censored molecules, which have no pIC50
    assert set(r["smiles"]) < set(c["smiles"])
    assert c.loc[~c["smiles"].isin(r["smiles"]), "y"].eq(0).all()  # every censored addition is a negative
    assert 0.3 < c["y"].mean() < 0.7  # threshold label should be roughly balanced
    assert c["smiles"].is_unique and r["smiles"].is_unique
    assert not c["smiles"].str.contains(r"[.]").any()  # salts stripped to the parent molecule


@pytest.mark.skipif(not os.path.exists("data/raw/tdc/gdsc2.pkl"), reason="run scripts/download_data.sh first")
def test_gdsc2_cold_drug_split():
    from src.data import gdsc2

    df = gdsc2.load()
    assert not df.duplicated(["Drug_ID", "Cell Line_ID"]).any()
    tr, _, te = gdsc2.split(df, "cold_drug")
    assert not set(df["Drug_ID"].iloc[tr]) & set(df["Drug_ID"].iloc[te])
    cells, x = gdsc2.expression_matrix(df)
    assert x.shape == (805, 17737)
