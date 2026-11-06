"""The paper caches numbers computed elsewhere, and one such cache already went stale in this
project. This keeps docs/paper.md honest against results/.

The matching rule lives in scripts/check_paper.locate and is shared with the CLI, so the test and
the command cannot drift apart. A claim that names a section is checked only inside that section:
before that existed, every one of 5.14's matched penalties passed here against numbers belonging to
other experiments while the section quoted the wrong rows of its own table.
"""
import io
import os

import pytest

from scripts.check_paper import PAPER, claims, locate


def paper_text():
    if not os.path.exists(PAPER):
        pytest.skip("paper not present")
    return io.open(PAPER, encoding="utf-8").read()


@pytest.mark.parametrize("claim", claims(), ids=lambda c: c[0])
def test_paper_value_matches_results(claim):
    label, value, dp = claim[:3]
    needle, hits, scope = locate(claim, paper_text())
    where = f"section {scope}" if scope else "the paper"
    assert hits, f"{label} = {needle} is in results/ but not in {where}"


def test_an_anchored_claim_is_not_satisfied_from_another_section():
    """The regression that section anchoring exists to catch: 0.110 appears in 5.8 as a ridge
    RMSE, so an unanchored check passes while a 5.14-anchored one does not."""
    text = paper_text()
    assert locate(("ridge value", 0.110, 3), text)[1] >= 1
    assert locate(("ridge value", 0.110, 3, "5.8"), text)[1] >= 1
    assert locate(("ridge value", 0.110, 3, "5.6"), text)[1] == 0


def test_a_claim_naming_a_missing_section_is_an_error():
    with pytest.raises(LookupError):
        locate(("nonsense", 1.0, 1, "99.9 Not A Section"), paper_text())
