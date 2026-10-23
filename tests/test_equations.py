"""Anchor values for the equation-level functions."""

import numpy as np
import pytest

from driven_pile import bearing, formulas, shaft, spt
from driven_pile import liquefaction as lq
from driven_pile.inputs import Pile, bpm_to_stroke
from driven_pile.options import ModelOptions


def close(x, y, rtol=0.05):
    assert np.isclose(x, y, rtol=rtol), f"{x} !~ {y}"


def test_danish_formula_matches_hand_calculation():
    pile = Pile(area=477 / 144, perimeter=6.627, width=2.0, length=150.0, modulus=864_000.0)
    s0 = np.sqrt(20.0 * 6.0 * 150.0 / (2.0 * pile.area * pile.modulus))
    q = 0.45 * 20.0 * 6.0 / (1.0 / 20.0 + s0)
    close(formulas.danish_capacity(20.0, 6.0, 20.0, pile), q, rtol=1e-12)


def test_bpm_to_stroke_free_fall():
    close(bpm_to_stroke(40.0), 32.2 / 2.0 * (0.75) ** 2, rtol=1e-12)


def test_wsdot_correlation_round_trip():
    for n in (5.0, 15.0, 30.0):
        close(spt.n160_from_phi(spt.phi_from_n160(n)), n, rtol=1e-6)
    assert spt.phi_from_n160(200.0) == 45.0
    assert spt.n160_from_phi(45.0) == 100.0


def test_boulanger_idriss_anchors():
    assert lq.delta_n_fines(0.0) < 0.05
    close(lq.delta_n_fines(35.0), 5.6, rtol=0.10)
    close(lq.crr_m75(10.0), 0.11, rtol=0.15)
    close(lq.crr_m75(20.0), 0.21, rtol=0.15)
    close(lq.k_sigma(15.0, 2.116), 1.0, rtol=0.01)
    close(lq.k_sigma(15.0, 2 * 2.116), 0.923, rtol=0.02)
    assert lq.k_sigma(15.0, 0.1 * 2.116) == 1.1


def test_csr_and_cycles():
    close(lq.csr_driving(33.0), 0.294, rtol=0.02)
    close(lq.cycles_to_liquefaction(0.2, 0.2), 15.0, rtol=1e-9)
    close(lq.cycles_to_liquefaction(0.24, 0.2), 15.0 * (0.2 / 0.24) ** (1 / 0.34), rtol=1e-9)
    assert lq.cycles_to_liquefaction(0.0, 0.2) == 1e9
    assert lq.ru_linear(5.0, 10.0) == 0.5
    assert lq.ru_linear(20.0, 10.0) == 1.0


def test_kramer_wang_residual_strength():
    sr = lq.residual_strength(10.0, 2.116)
    close(sr * 1000.0, 2116.0 * np.exp(-8.444 + 1.09 + 5.379), rtol=1e-9)


def test_vesic_factor_grows_with_phi_and_rigidity():
    assert bearing.nq_vesic(35.0, 100.0) > bearing.nq_vesic(30.0, 100.0)
    assert bearing.nq_vesic(35.0, 200.0) > bearing.nq_vesic(35.0, 100.0)


def test_rigidity_index_decays_with_stress():
    opts = ModelOptions()
    assert bearing.rigidity_index(35.0, 1.0, opts) > bearing.rigidity_index(35.0, 4.0, opts)
    assert bearing.rigidity_index(45.0, 2.0, opts) > bearing.rigidity_index(30.0, 2.0, opts)


def test_phi_inversion_round_trip_and_bounds():
    opts = ModelOptions()
    q = float(bearing.toe_bearing(3.0, 36.0, opts))
    phi, reason = bearing.phi_from_toe_bearing(q, 3.0, opts)
    close(phi, 36.0, rtol=1e-6)
    assert reason == "ok"
    assert bearing.phi_from_toe_bearing(0.01, 3.0, opts) == (25.0, "below_floor")
    assert bearing.phi_from_toe_bearing(1e6, 3.0, opts) == (45.0, "above_ceiling")
    assert np.isnan(bearing.phi_from_toe_bearing(np.nan, 3.0, opts)[0])
    close(bearing.phi_from_toe_bearing_grid(q, 3.0, opts), 36.0, rtol=1e-3)


def test_coyle_castello_table_and_extension():
    close(shaft.coyle_castello_k(33.0, 2.0), 2.17, rtol=1e-9)
    assert shaft.coyle_castello_k(40.0, 2.0) > shaft.coyle_castello_k(36.0, 2.0)
    fs = shaft.fs_coyle_castello(2.0, 33.0, 0.0, 2.0)
    close(fs, 2.0 * 2.17 * np.tan(np.radians(0.8 * 33.0)), rtol=1e-9)


def test_partial_liquefaction_law():
    assert shaft.fs_partially_liquefied(1.0, 0.0, 0.2, 1.0) == 1.0
    assert shaft.fs_partially_liquefied(1.0, 1.0, 0.2, 0.5) == pytest.approx(0.1)
    assert shaft.fs_partially_liquefied(1.0, 0.5, 2.0, 1.0) == 1.0


def test_interface_alpha_branches():
    assert shaft.interface_alpha(0.22, 1.0) == pytest.approx(1.0)
    assert shaft.interface_alpha(0.88, 1.0) == pytest.approx(0.5)
    assert shaft.interface_alpha(4.0, 1.0) == pytest.approx(np.sqrt(0.22) * 4.0 ** -0.25)
    assert shaft.interface_alpha(0.0, 1.0) == 1.0
