"""The exported splits must be the same folds the existing TG2 results were computed on.

If this drifts, every comparison in docs/state_of_the_art_comparison.md is between models scored on
different molecules, which is the failure the export exists to prevent.
"""
import pytest

from scripts.export_splits import (GROUPED_K, chembl_dataset, dataset, grouped, lopo,
                                   publication_subset, splits)
from src.data import chem, tg2

SEEDS = range(3)


@pytest.fixture(scope="module")
def df():
    return dataset()


@pytest.fixture(scope="module")
def exported(df):
    return splits(df, SEEDS)


def test_dataset_matches_the_published_combined_set(df):
    base = tg2.combined("classification")
    assert len(df) == 631
    assert df["smiles"].tolist() == base["smiles"].tolist()
    assert df["y"].tolist() == base["y"].tolist()


def test_scaffold_folds_are_byte_identical_to_a_fresh_split(df, exported):
    smiles = df["smiles"].tolist()
    for seed in SEEDS:
        fresh = chem.scaffold_split(smiles, seed=seed)
        got = exported[(exported.split_kind == "scaffold") & (exported.seed == seed)]
        for fold, idx in zip(("train", "valid", "test"), fresh):
            assert got[got.fold == fold]["smiles"].tolist() == df.iloc[idx]["smiles"].tolist()


def test_lopo_folds_are_byte_identical_to_a_fresh_derivation(df, exported):
    sub = publication_subset(df)
    for i, paper, idx in lopo(sub):
        got = exported[(exported.split_kind == "paper_lopo") & (exported.seed == i)]
        assert set(got["held_out_paper"]) == {paper}
        for fold, part in zip(("train", "valid", "test"), idx):
            assert got[got.fold == fold]["smiles"].tolist() == sub.loc[part, "smiles"].tolist()


def test_every_molecule_lands_in_exactly_one_fold_per_seed(exported):
    assert exported.groupby(["split_kind", "seed", "smiles"]).size().max() == 1


def test_labels_are_consistent_for_a_molecule_across_every_row(exported):
    assert (exported.groupby("smiles")["y"].nunique() == 1).all()


def test_scaffold_splits_share_no_murcko_core_between_train_and_test(df, exported):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaf = {s: MurckoScaffold.MurckoScaffoldSmiles(s) for s in df["smiles"]}
    for kind in ("scaffold", "scaffold_pub"):
        for seed in SEEDS:
            g = exported[(exported.split_kind == kind) & (exported.seed == seed)]
            train = {scaf[s] for s in g[g.fold == "train"]["smiles"]}
            assert not train & {scaf[s] for s in g[g.fold == "test"]["smiles"]}


def test_lopo_leaks_neither_the_held_out_paper_nor_the_test_molecules(exported):
    g = exported[exported.split_kind == "paper_lopo"]
    for seed, fold in g.groupby("seed"):
        held = fold["held_out_paper"].iloc[0]
        fit = fold[fold.fold.isin(["train", "valid"])]
        assert held not in set(fit["paper"]), f"fold {seed} trains on the held-out paper"
        assert not set(fit["smiles"]) & set(fold[fold.fold == "test"]["smiles"])


def test_lopo_tests_every_subset_molecule_exactly_once(df, exported):
    sub = publication_subset(df)
    test = exported[(exported.split_kind == "paper_lopo") & (exported.fold == "test")]
    assert sorted(test["smiles"]) == sorted(sub["smiles"])


def test_both_publication_arms_cover_the_same_molecules(exported):
    a = set(exported[exported.split_kind == "scaffold_pub"]["smiles"])
    b = set(exported[exported.split_kind == "paper_lopo"]["smiles"])
    assert a == b, "the splitting-strategy comparison is not matched on molecules"


def test_export_is_deterministic(df):
    assert splits(df, SEEDS).equals(splits(df, SEEDS))


def test_seeded_test_folds_are_usable(exported):
    """AUROC is undefined on a single-class fold. The three seeded protocols must never produce one;
    paper_lopo is exempt and audited by the next test instead, because its folds are whole papers.
    """
    seeded = exported[exported.split_kind != "paper_lopo"]
    for (kind, seed), g in seeded[seeded.fold == "test"].groupby(["split_kind", "seed"]):
        assert len(g) >= 19, f"{kind} seed {seed} has only {len(g)} test molecules"
        assert g["y"].nunique() == 2, f"{kind} seed {seed} test fold has one class"


def test_lopo_single_class_folds_are_the_known_ones(exported):
    """One of the 11 TG2 publications reports only potent molecules, so its fold yields no AUROC.
    Pinned so that a data change making this worse fails loudly instead of quietly dropping folds.
    """
    t = exported[(exported.split_kind == "paper_lopo") & (exported.fold == "test")]
    one_class = t.groupby("held_out_paper")["y"].nunique().eq(1)
    assert one_class.sum() <= 1, f"too many unusable LOPO folds: {list(one_class[one_class].index)}"
    assert t.groupby("seed").size().min() >= 10


# ---------------------------------------------------------------- the grouped publication protocol
#
# paper_grouped is what docs/replication_feasibility.md puts in place of leave-one-publication-out
# on CDK2 and MAO-B. Its whole value rests on holding out *whole* publications, so these cases pin
# that property on real CDK2 data rather than on a synthetic frame.

@pytest.fixture(scope="module")
def cdk2():
    return publication_subset(chembl_dataset("cdk2"))


@pytest.fixture(scope="module")
def cdk2_grouped(cdk2):
    return list(grouped(cdk2, GROUPED_K))


def test_grouped_folds_hold_out_whole_publications(cdk2, cdk2_grouped):
    """No publication may appear in both the training and the test side of a fold.

    This is the only thing separating the protocol from a scaffold split: docs/second_target.md
    measured 69 publications straddling CDK2's scaffold folds, and that leak is the mechanism
    behind the entire effect being replicated.
    """
    for _, (tr, va, te) in cdk2_grouped:
        test_papers = set(cdk2.loc[te, "paper"])
        assert not test_papers & set(cdk2.loc[tr, "paper"])
        assert not test_papers & set(cdk2.loc[va, "paper"])


def test_grouped_folds_test_every_molecule_exactly_once(cdk2, cdk2_grouped):
    """Unlike leave-one-publication-out, the folds must not overlap -- that overlap is the pooling
    artefact which narrowed bootstrap intervals by 15-18% in docs/second_target.md."""
    tested = [i for _, (_, _, te) in cdk2_grouped for i in te]
    assert sorted(tested) == sorted(cdk2.index)


def test_grouped_folds_partition_each_fold(cdk2, cdk2_grouped):
    for _, (tr, va, te) in cdk2_grouped:
        assert len(set(tr) | set(va) | set(te)) == len(cdk2)
        assert not set(tr) & set(va)


def test_grouped_test_folds_are_usable(cdk2, cdk2_grouped):
    """The protocol was chosen partly because it yields no single-class folds on CDK2."""
    for _, (_, _, te) in cdk2_grouped:
        assert cdk2.loc[te, "y"].nunique() == 2


def test_fold_sampling_is_reproducible_and_a_true_subset():
    from scripts.run_sota import sample_folds
    pool = list(range(209))
    assert sample_folds(pool, 50, seed=0) == sample_folds(pool, 50, seed=0)
    assert sample_folds(pool, 50, seed=0) != sample_folds(pool, 50, seed=1)
    got = sample_folds(pool, 50, seed=0)
    assert len(got) == len(set(got)) == 50
    assert set(got) <= set(pool)
    assert got == sorted(got)


def test_fold_sampling_is_not_a_prefix():
    """A prefix of leave-one-out folds would be the oldest publications, since folds are indexed
    by sorted ChEMBL document id. The draw must spread across the range."""
    from scripts.run_sota import sample_folds
    got = sample_folds(list(range(209)), 50, seed=0)
    assert got != list(range(50))
    assert max(got) > 150                      # reaches the recent publications
    assert sum(s >= 104 for s in got) >= 15    # a fair share from the back half


def test_fold_sampling_returns_everything_when_asked_for_too_many():
    from scripts.run_sota import sample_folds
    assert sample_folds([3, 1, 2], 10) == [3, 1, 2]
