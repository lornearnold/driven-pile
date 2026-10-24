"""End-to-end behavior of analyze and monte_carlo on one benchmark pile."""

import numpy as np
import pytest
from conftest import load_fixture, paper_inputs

import driven_pile as dp


@pytest.fixture(scope="module")
def inputs():
    return load_fixture("B-29.5"), *paper_inputs(load_fixture("B-29.5"))


def test_analyze_uses_lowest_admissible_top_phi(inputs):
    fx, log, soil = inputs
    out = dp.analyze(log, soil)
    assert out.mc is None
    assert out.top_band.lo == fx["band"]["lo"]
    assert out.drive.top_phi == out.top_band.lo
    assert out.drive.ltc == pytest.approx(fx["reference"]["ltc"], rel=0.01)
    assert out.drive.ltc == pytest.approx(out.drive.ltc_shaft + out.drive.toe_eod)
    assert out.drive.reason_counts()["ok"] > 0
    assert 0.0 <= out.drive.fraction_liquefied() <= 1.0


def test_long_term_capacity_matches_result_fields(inputs):
    fx, log, soil = inputs
    target = dp.prepare_target(log, soil)
    res = dp.run_drive(target, top_phi=fx["band"]["lo"])
    shaft, toe = dp.long_term_capacity(res)
    assert shaft == pytest.approx(res.ltc_shaft)
    assert toe == pytest.approx(res.toe_eod)


def test_result_carries_target_capacities(inputs):
    fx, log, soil = inputs
    target = dp.prepare_target(log, soil)
    res = dp.run_drive(target, top_phi=fx["band"]["lo"])
    np.testing.assert_array_equal(res.target, target.capacity)
    np.testing.assert_array_equal(res.raw_capacity, target.raw_capacity)
    assert np.all(np.isnan(res.raw_capacity[: res.n_top]))


def test_monte_carlo_is_seeded_and_shaped(inputs):
    fx, log, soil = inputs
    target = dp.prepare_target(log, soil)
    band = dp.TopPhiBand(phi=np.array([]), rms=np.array([]),
                         lo=fx["band"]["lo"], hi=fx["band"]["hi"])
    opts = dp.MonteCarloOptions(chunk=5, max_realizations=10, seed=3)
    a = dp.monte_carlo(log, soil, mc_options=opts, target=target, top_band=band)
    b = dp.monte_carlo(log, soil, mc_options=opts, target=target, top_band=band)
    assert a.n == 10
    np.testing.assert_allclose(a.ltc, b.ltc)
    assert a.phi_quantiles.shape == (len(target.elevation), 5)
    assert np.all((a.top_phi >= band.lo) & (a.top_phi <= band.hi))
    assert np.all(np.abs(a.dphi) <= opts.dphi_bound)
    np.testing.assert_allclose(a.ltc, a.shaft + a.toe_eod)
    assert a.quantiles(0.5) == pytest.approx(np.median(a.ltc))
