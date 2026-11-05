"""The novelty-share validation in scripts/validate_novelty_share.py.

The script exists to put an interval and a sensitivity check on a number -- "51-78% of the
publication penalty is chemical novelty" -- that a previous commit reported as a bare point
estimate. Three things would make it lie: a bootstrap that resamples molecules instead of
publications (too narrow an interval), a share formula that reads high when novelty explains
nothing, and band logic that silently scores a band present in only one arm.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.validate_novelty_share import PaperBootstrap, banded_share, continuous_share

EDGES = (0.0, 0.5, 1.01)


def mols(sim, y, p, paper="P1", model="rf"):
    return pd.DataFrame({"model": model, "smiles": [f"C{i}" for i in range(len(y))],
                         "y": y, "p": p, "paper": paper, "max_tanimoto": sim})


def separable(n, sim, offset=0.0, paper="P1"):
    """n molecules at one novelty level, with probabilities that rank the labels perfectly when
    offset is 0 and worse as offset grows."""
    y = ([0] * (n // 2)) + ([1] * (n - n // 2))
    p = [0.3 + offset * (i % 2) for i in range(n // 2)] + \
        [0.7 - offset * (i % 2) for i in range(n - n // 2)]
    return mols([sim] * n, y, p, paper=paper)


def test_share_is_near_one_when_only_the_novelty_mix_differs():
    """Same accuracy inside each band; the arms just draw different proportions from them."""
    pub = pd.concat([separable(10, 0.8), separable(90, 0.2, offset=0.45)])   # mostly novel
    sca = pd.concat([separable(90, 0.8), separable(10, 0.2, offset=0.45)])   # mostly familiar
    r = banded_share(pub, sca, EDGES)
    assert r["penalty_raw"] < -0.05                 # a visible raw penalty
    assert r["share"] > 0.9                         # essentially all composition


def test_share_is_near_zero_when_the_arms_have_the_same_novelty_mix():
    """Equal composition, but the publication arm is worse inside every band."""
    pub = pd.concat([separable(50, 0.8, offset=0.45), separable(50, 0.2, offset=0.45)])
    sca = pd.concat([separable(50, 0.8), separable(50, 0.2)])
    r = banded_share(pub, sca, EDGES)
    assert r["penalty_raw"] < -0.05
    assert abs(r["share"]) < 0.1                    # nothing is composition


def test_a_band_present_in_only_one_arm_is_not_scored():
    pub = pd.concat([separable(40, 0.8), separable(40, 0.2)])
    sca = separable(40, 0.8)                        # no novel molecules at all
    assert banded_share(pub, sca, EDGES)["bands_used"] == 1


def test_a_band_with_one_class_is_not_scored():
    one_class = mols([0.2] * 20, [1] * 20, [0.6] * 20)
    pub = pd.concat([separable(40, 0.8), one_class])
    sca = pd.concat([separable(40, 0.8), one_class])
    assert banded_share(pub, sca, EDGES)["bands_used"] == 1


def test_no_shared_band_returns_nothing():
    assert banded_share(separable(40, 0.8), separable(40, 0.2), EDGES) is None


def test_bootstrap_resamples_whole_publications():
    """Every drawn row must belong to a publication, and a drawn publication comes whole."""
    pub = pd.concat([separable(10, 0.5, paper=f"P{i}") for i in range(8)], ignore_index=True)
    sca = pub.copy()
    boot = PaperBootstrap(pub, sca)
    rng = np.random.default_rng(0)
    for _ in range(10):
        a, _ = boot.draw(rng)
        sizes = a.groupby("paper").size()
        assert set(sizes.index) <= set(pub["paper"])
        assert all(n % 10 == 0 for n in sizes)      # whole blocks of 10, never a part
        assert len(a) == len(pub)                   # same total, publications resampled


def test_bootstrap_applies_the_same_publications_to_both_arms():
    pub = pd.concat([separable(10, 0.5, paper=f"P{i}") for i in range(8)], ignore_index=True)
    sca = pub.copy()
    boot = PaperBootstrap(pub, sca)
    rng = np.random.default_rng(1)
    a, b = boot.draw(rng)
    assert list(a.groupby("paper").size().items()) == list(b.groupby("paper").size().items())


def test_bootstrap_varies_between_draws_and_is_seed_reproducible():
    pub = pd.concat([separable(10, 0.5, paper=f"P{i}") for i in range(8)], ignore_index=True)
    boot = PaperBootstrap(pub, pub.copy())
    first = boot.draw(np.random.default_rng(0))[0]["paper"].tolist()
    again = boot.draw(np.random.default_rng(0))[0]["paper"].tolist()
    other = boot.draw(np.random.default_rng(7))[0]["paper"].tolist()
    assert first == again
    assert first != other


def test_bootstrap_skips_a_publication_missing_from_one_arm():
    pub = pd.concat([separable(10, 0.5, paper=f"P{i}") for i in range(4)], ignore_index=True)
    sca = pub[pub["paper"] != "P3"].copy()
    boot = PaperBootstrap(pub, sca)
    _, b = boot.draw(np.random.default_rng(0))
    assert "P3" not in set(b["paper"])


def test_continuous_share_detects_a_purely_novelty_driven_gap():
    """Accuracy falls with novelty; the arms differ only in the novelty they were given."""
    rng = np.random.default_rng(0)
    def arm(sims):
        y = rng.integers(0, 2, len(sims))
        # a correct prediction becomes likelier as similarity rises
        correct = rng.random(len(sims)) < 0.4 + 0.5 * sims
        p = np.where(correct, np.where(y == 1, 0.9, 0.1), np.where(y == 1, 0.1, 0.9))
        return pd.DataFrame({"y": y, "p": p, "max_tanimoto": sims, "paper": "P1"})
    pub = arm(rng.uniform(0.0, 0.5, 3000))
    sca = arm(rng.uniform(0.5, 1.0, 3000))
    c = continuous_share(pub, sca)
    assert c["share"] > 0.7            # similarity accounts for most of the arm coefficient


def test_publication_contrast_returns_nothing_without_an_unseen_group(tmp_path, monkeypatch):
    """CDK2 and TG2's real situation: every publication straddles the scaffold split."""
    import scripts.validate_novelty_share as mod
    sp = pd.DataFrame({
        "split_kind": "scaffold_pub", "seed": [0] * 6,
        "fold": ["train", "train", "test", "test", "train", "test"],
        "paper": ["P1", "P2", "P1", "P2", "P3", "P3"],
        "smiles": [f"C{i}" for i in range(6)], "y": [0, 1, 0, 1, 0, 1]})
    sp.to_csv(tmp_path / "splits.csv", index=False)
    monkeypatch.setattr(mod, "per_molecule", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("scripts.run_sota.paths",
                        lambda t: (str(tmp_path / "splits.csv"), "", ""))
    assert mod.publication_contrast("cdk2") is None


def test_overlap_is_full_when_both_arms_cover_the_same_bands():
    from scripts.validate_novelty_share import overlap
    a = pd.concat([separable(40, 0.8), separable(40, 0.2)]).assign(
        band=lambda d: pd.cut(d["max_tanimoto"], [0.0, 0.5, 1.01], right=False))
    o = overlap(a, a.copy(), min_per_band=20)
    assert o["support_pub"] == 1.0 and o["support_scaffold"] == 1.0
    assert o["min_band_n"] == 40


def test_overlap_flags_a_band_one_arm_barely_reaches():
    """TG2's situation: the publication arm lives where the scaffold arm has almost nothing."""
    from scripts.validate_novelty_share import overlap
    cut = lambda d: pd.cut(d["max_tanimoto"], [0.0, 0.5, 1.01], right=False)
    pub = pd.concat([separable(40, 0.8), separable(360, 0.2)])     # mostly novel
    sca = pd.concat([separable(390, 0.8), separable(10, 0.2)])     # 10 novel molecules
    pub, sca = pub.assign(band=cut(pub)), sca.assign(band=cut(sca))
    o = overlap(pub, sca, min_per_band=20)
    assert o["support_pub"] < 0.15        # 90% of the publication arm has no counterpart
    assert o["support_scaffold"] > 0.9


def test_overlap_is_zero_when_no_band_clears_the_floor_in_both_arms():
    from scripts.validate_novelty_share import overlap
    cut = lambda d: pd.cut(d["max_tanimoto"], [0.0, 0.5, 1.01], right=False)
    pub = separable(40, 0.2).pipe(lambda d: d.assign(band=cut(d)))
    sca = separable(40, 0.8).pipe(lambda d: d.assign(band=cut(d)))
    o = overlap(pub, sca, min_per_band=20)
    assert o["support_pub"] == 0.0 and o["min_band_n"] == 0
