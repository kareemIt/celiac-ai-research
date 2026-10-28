"""GDSC2 drug response (TDC DrugRes): drug SMILES + cell-line expression (17,737 genes) -> log-normalized IC50."""
import numpy as np

from src.data.chem import group_split

RAW = "data/raw/tdc"


def load(path=RAW):
    from tdc.multi_pred import DrugRes

    df = DrugRes(name="GDSC2", path=path).get_data()
    # 674 drug/cell pairs are measured more than once; average them so a pair can't land in train and test.
    y = df.groupby(["Drug_ID", "Cell Line_ID"], as_index=False)["Y"].mean()
    first = df.drop_duplicates(["Drug_ID", "Cell Line_ID"]).drop(columns="Y")
    return first.merge(y, on=["Drug_ID", "Cell Line_ID"])


def expression_matrix(df):
    """One row per cell line: (cell_ids, float32 matrix [n_cells, 17737])."""
    cells = df.drop_duplicates("Cell Line_ID")
    return cells["Cell Line_ID"].to_numpy(), np.stack(cells["Cell Line"].to_numpy()).astype(np.float32)


def features(df, train_idx, drug_vec, seed=0, n_pcs=256):
    """Row features = [drug vector ‖ expression PCA]. PCA is fit on training cell lines only,
    so cold_cell test lines don't leak into the projection. drug_vec: {Drug_ID: 1-D array}."""
    from sklearn.decomposition import PCA

    cells, expr = expression_matrix(df)
    row_cell = df["Cell Line_ID"].map({c: i for i, c in enumerate(cells)}).to_numpy()
    z = PCA(n_components=n_pcs, random_state=seed).fit(expr[np.unique(row_cell[train_idx])]).transform(expr)
    row_drug = np.stack([drug_vec[d] for d in df["Drug_ID"]]).astype(np.float32)
    return np.hstack([row_drug, z[row_cell].astype(np.float32)])


def split(df, kind="random", seed=0):
    """random: pairs mixed (TDC default, optimistic). cold_drug / cold_cell: unseen drugs / cell lines in test."""
    if kind == "random":
        return group_split(np.arange(len(df)), seed=seed)
    col = {"cold_drug": "Drug_ID", "cold_cell": "Cell Line_ID"}[kind]
    return group_split(df[col].to_numpy(), seed=seed)
