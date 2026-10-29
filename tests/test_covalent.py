"""The covalent result rests on one unit conversion and one exclusion rule. Both are checked here:
a wrong factor of 60 would silently shift two papers by 1.8 log units."""
import numpy as np
import pandas as pd
import pytest

from scripts.run_covalent import TO_PER_M_S, kinetic_table


def ratio_row(smiles, value, units, relation="=", doc="d1"):
    return {
        "smiles": smiles, "canonical_smiles": smiles,
        "standard_type": "Ratio", "standard_relation": relation,
        "standard_value": value, "standard_units": units,
        "data_validity_comment": None, "document_chembl_id": doc,
    }


def table(rows, tmp_path):
    p = tmp_path / "acts.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return kinetic_table(str(p))


def test_per_minute_converts_to_per_second():
    """1e3/M/min is 1000/60 /M.s. Getting this backwards would move those papers by 1.8 log."""
    assert TO_PER_M_S["/M/s"] == 1.0
    assert TO_PER_M_S["10^3/M/min"] == pytest.approx(1000 / 60)
    assert TO_PER_M_S["10'3/M/min"] == pytest.approx(1000 / 60)


def test_the_two_unit_conventions_agree_after_conversion(tmp_path):
    """The same physical rate written both ways must land on the same log value."""
    t = table([ratio_row("CCO", 1000.0, "/M/s"),
               ratio_row("CCN", 60.0, "10^3/M/min")], tmp_path)
    assert len(t) == 2
    assert t.logk.iloc[0] == pytest.approx(t.logk.iloc[1])
    assert t.logk.iloc[0] == pytest.approx(3.0)


def test_unitless_rows_are_dropped_not_guessed(tmp_path):
    """118 real rows carry no unit; inferring one from magnitude would be inventing data."""
    t = table([ratio_row("CCO", 1000.0, None)], tmp_path)
    assert t.empty


def test_censored_rows_are_dropped(tmp_path):
    """A bounded rate constant cannot be regressed on."""
    t = table([ratio_row("CCO", 1000.0, "/M/s", relation=">")], tmp_path)
    assert t.empty


def test_replicates_collapse_to_the_median_and_keep_their_paper(tmp_path):
    t = table([ratio_row("CCO", 100.0, "/M/s", doc="p1"),
               ratio_row("CCO", 10000.0, "/M/s", doc="p1"),
               ratio_row("CCO", 1000.0, "/M/s", doc="p1")], tmp_path)
    assert len(t) == 1
    assert t.logk.iloc[0] == pytest.approx(3.0)
    assert t.n_meas.iloc[0] == 3
    assert t.paper.iloc[0] == "p1"
