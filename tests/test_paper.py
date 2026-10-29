"""The paper caches numbers computed elsewhere, and one such cache already went stale in this
project. This keeps docs/paper.md honest against results/."""
import pytest

from scripts.check_paper import claims, PAPER


def paper_text():
    import io
    import os
    if not os.path.exists(PAPER):
        pytest.skip("paper not present")
    return io.open(PAPER, encoding="utf-8").read().replace("−", "-")


@pytest.mark.parametrize("label,value,dp", claims())
def test_paper_value_matches_results(label, value, dp):
    assert f"{float(value):.{dp}f}".lstrip("-") in paper_text(), (
        f"{label} = {value:.{dp}f} is in results/ but not in the paper")
