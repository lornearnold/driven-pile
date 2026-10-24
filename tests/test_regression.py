"""Reproduce the research code's results on the benchmark piles."""

import numpy as np
import pytest
from conftest import paper_inputs

import driven_pile as dp


def test_target_matches_research_within_rounding(fixture):
    log, soil = paper_inputs(fixture)
    target = dp.prepare_target(log, soil)
    ref = fixture["target"]
    n_top = target.n_top
    np.testing.assert_allclose(target.elevation[n_top:], ref["elevation"])
    np.testing.assert_allclose(target.bpf[n_top:], ref["bpf"])
    np.testing.assert_allclose(target.capacity[n_top:], ref["capacity"], atol=1.5)


def test_characterization_reproduces_research_on_identical_target(fixture):
    log, soil = paper_inputs(fixture)
    target = dp.prepare_target(log, soil)
    research_target = target.with_capacity(
        np.concatenate([np.full(target.n_top, np.nan), fixture["target"]["capacity"]]))
    ref = fixture["reference"]
    res = dp.run_drive(research_target, top_phi=ref["top_phi"])
    np.testing.assert_allclose(res.elevation, ref["elevation"])
    np.testing.assert_allclose(res.phi, ref["phi"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(res.ru, ref["ru"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(res.ru_toe, ref["ru_toe"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(res.felt, ref["felt"])
    np.testing.assert_allclose(res.predicted, ref["predicted"], rtol=1e-6, atol=1e-3)
    assert res.reason.tolist() == ref["reason"]
    assert res.ltc_shaft == pytest.approx(ref["shaft"], rel=1e-6)
    assert res.toe_eod == pytest.approx(ref["toe_eod"], rel=1e-6, abs=1e-3)
    assert res.rms_kips == pytest.approx(ref["rms_kips"], rel=1e-6, abs=1e-6)


def test_capacity_from_log_matches_research_within_one_percent(fixture):
    log, soil = paper_inputs(fixture)
    target = dp.prepare_target(log, soil)
    ref = fixture["reference"]
    res = dp.run_drive(target, top_phi=ref["top_phi"])
    assert res.ltc == pytest.approx(ref["ltc"], rel=0.01)


@pytest.mark.slow
def test_top_band_matches_frozen_sweep(fixture):
    log, soil = paper_inputs(fixture)
    target = dp.prepare_target(log, soil)
    band = dp.sweep_top_phi(target)
    assert (band.lo, band.hi) == (fixture["band"]["lo"], fixture["band"]["hi"])
