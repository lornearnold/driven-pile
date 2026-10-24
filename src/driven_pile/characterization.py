"""Sequential implicit characterization: step down the pile one logged foot
at a time, solve each foot's friction angle and toe pore-pressure ratio to
self-consistency from the capacity target, and keep updating the pore
pressure of the feet above as later blows arrive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import liquefaction as lq
from .bearing import phi_from_toe_bearing, phi_from_toe_bearing_grid, toe_bearing
from .capacity import shaft_long_term, toe_end_of_drive
from .constants import (
    FOOT_INCREMENT,
    LIQ_AFTER,
    LIQ_BEFORE,
    LIQ_DURING,
    LIQ_NEVER,
    NONFINITE,
    OK,
    TOP_SEGMENT,
)
from .options import ModelOptions
from .results import DriveResult
from .shaft import fs_coyle_castello, fs_partially_liquefied, interface_alpha
from .spt import n160_from_phi

_RU_GRID = 25


@dataclass
class Foot:
    """State of one foot of soil during the characterization pass."""

    index: int
    elevation: float
    sigma_v0: float
    z: float
    bpf: float
    fines: float
    phi: float = np.nan
    n160: float = np.nan
    csr: float = np.nan
    crr15: float = np.nan
    nl: float = np.nan
    felt_arrival: float = np.nan
    felt: float = 0.0
    ru: float = 0.0
    ru_toe: float = 0.0
    liquefied: bool = False
    liq_when: str = LIQ_NEVER
    reason: str = OK


def triggering_state(phi, sigma_v0, fines, options):
    """Blow count, cyclic stress and resistance, and cycles to liquefaction.

    Parameters
    ----------
    phi : float or ndarray
        Friction angle, deg.
    sigma_v0 : float
        Static vertical effective stress, ksf.
    fines : float
        Fines content, percent.
    options : ModelOptions

    Returns
    -------
    n160, csr, crr15, nl
        Each a float or ndarray matching ``phi``.
    """
    n160 = n160_from_phi(np.asarray(phi) - options.dphi, cap=options.phi_hi)
    csr = lq.csr_driving(phi)
    crr15 = lq.crr_15(n160, fines, sigma_v0) * options.crr_multiplier
    nl = lq.cycles_to_liquefaction(csr, crr15, b=options.b_exponent)
    return n160, csr, crr15, nl


def fs_unliquefied(foot, pile, options):
    """Unit shaft resistance of a foot with its pore pressure dissipated, ksf."""
    if not np.isfinite(foot.phi):
        return 0.0
    return float(fs_coyle_castello(foot.sigma_v0, foot.phi, foot.z, pile.width,
                                   options.cc_delta_ratio))


def fs_current(foot, pile, options):
    """Unit shaft resistance of a foot in its current state, ksf."""
    if not np.isfinite(foot.phi):
        return 0.0
    fu = fs_unliquefied(foot, pile, options)
    if foot.ru <= 0.0:
        return fu
    sr = float(lq.residual_strength(min(foot.n160, lq.N160_RESIDUAL_MAX), foot.sigma_v0))
    alpha = interface_alpha(sr, foot.sigma_v0, options.sr_psi_nc)
    return fs_partially_liquefied(fu, foot.ru, sr, alpha)


def shaft_resistance(feet, pile, options):
    """Shaft resistance of a list of feet in their current states, kips."""
    return sum(fs_current(f, pile, options) * pile.perimeter * FOOT_INCREMENT for f in feet)


def _ru_implied(ru, q_needed, foot, felt_end, options):
    """Pore-pressure ratio implied by characterizing the toe at reduction ``ru``."""
    sigma = foot.sigma_v0 * (1.0 - np.asarray(ru, dtype=float))
    phi = phi_from_toe_bearing_grid(q_needed, sigma, options)
    _, _, _, nl = triggering_state(phi, foot.sigma_v0, foot.fines, options)
    return np.minimum(lq.ru_linear(felt_end, nl), options.ru_cap_toe)


def _ru_roots(q_needed, foot, felt_end, options):
    """All self-consistent toe pore-pressure ratios in ``[0, ru_cap_toe]``, ascending."""
    cap = options.ru_cap_toe
    grid = np.linspace(0.0, cap, _RU_GRID)
    h = grid - _ru_implied(grid, q_needed, foot, felt_end, options)
    roots = []
    for k in range(_RU_GRID - 1):
        if h[k] == 0.0:
            roots.append(float(grid[k]))
        elif h[k] * h[k + 1] < 0:
            roots.append(float(brentq(
                lambda r: r - float(_ru_implied(r, q_needed, foot, felt_end, options)),
                grid[k], grid[k + 1], xtol=1e-5)))
    if h[-1] <= 0.0:
        roots.append(cap)
    if not roots:
        roots = [0.0]
    return sorted({round(r, 6) for r in roots})


def characterize_foot(foot, q_needed, felt_arrival, options):
    """Characterize the soil at the toe from the required toe resistance.

    Parameters
    ----------
    foot : Foot
        The foot the toe has just reached; updated in place.
    q_needed : float
        Unit toe resistance the target demands after subtracting the shaft, ksf.
    felt_arrival : float
        Blows the foot felt before the toe arrived.
    options : ModelOptions
    """
    felt_end = felt_arrival + foot.bpf
    foot.felt_arrival = felt_arrival

    phi, reason = phi_from_toe_bearing(q_needed, foot.sigma_v0, options)
    if not np.isfinite(phi):
        foot.reason = reason
        return
    state = triggering_state(phi, foot.sigma_v0, foot.fines, options)

    ru_toe = _ru_roots(q_needed, foot, felt_end, options)[0]
    if ru_toe > 0.0:
        phi, reason = phi_from_toe_bearing(q_needed, foot.sigma_v0 * (1.0 - ru_toe), options)
        state = triggering_state(phi, foot.sigma_v0, foot.fines, options)
    n160, csr, crr15, nl = (float(x) for x in state)

    if felt_arrival >= nl:
        foot.liq_when, foot.liquefied = LIQ_BEFORE, True
    elif felt_end >= nl:
        foot.liq_when, foot.liquefied = LIQ_DURING, True
    else:
        foot.liq_when, foot.liquefied = LIQ_NEVER, False

    foot.phi, foot.reason, foot.ru_toe = float(phi), reason, float(ru_toe)
    foot.n160, foot.csr, foot.crr15, foot.nl = n160, csr, crr15, nl
    foot.felt = felt_end
    foot.ru = 1.0 if foot.liquefied else float(lq.ru_linear(felt_end, nl))


def run_drive(target, options=None, top_phi=None):
    """Characterize every foot sequentially from the capacity target, top down.

    Parameters
    ----------
    target : Target
        Capacity target, pile and static soil state on the analysis grid.
    options : ModelOptions, optional
    top_phi : float, optional
        Friction angle assigned to the unlogged top segment, deg. Required
        when the target has a top segment.

    Returns
    -------
    DriveResult
    """
    options = options or ModelOptions()
    pile = target.pile
    elev = target.elevation
    n, n_top = len(elev), target.n_top
    if n_top > 0 and top_phi is None:
        raise ValueError("top_phi is required when the target has an unlogged top segment")
    ground = max(float(target.mudline), float(elev.max()))
    zone_h = options.zone_factor * pile.width

    feet = [Foot(index=i, elevation=float(elev[i]), sigma_v0=float(target.sigma_v0[i]),
                 z=max(ground - float(elev[i]), 0.0),
                 bpf=float(target.bpf[i]) if np.isfinite(target.bpf[i]) else 0.0,
                 fines=float(target.fines[i]))
            for i in range(n)]
    for f in feet[:n_top]:
        f.reason = TOP_SEGMENT
        f.phi = float(top_phi)
        f.n160, f.csr, f.crr15, f.nl = (float(x) for x in
                                        triggering_state(f.phi, f.sigma_v0, f.fines, options))

    predicted = np.full(n, np.nan)
    felt = np.zeros(n)
    for i in range(n_top, n):
        f = feet[i]
        liq_before = np.array([g.liquefied for g in feet])
        tgt = target.capacity[i]
        if not np.isfinite(tgt):
            f.reason = NONFINITE
        else:
            qs = shaft_resistance(feet[:i], pile, options)
            characterize_foot(f, (float(tgt) - qs) / pile.area, felt[i], options)
            if np.isfinite(f.phi):
                qp = float(toe_bearing(f.sigma_v0 * (1.0 - f.ru_toe), f.phi, options))
                predicted[i] = qp * pile.area + qs
        if f.bpf > 0:
            reach = (elev - elev[i] >= -zone_h) & ~liq_before
            felt[reach] += f.bpf
        for j in range(i + 1):
            g = feet[j]
            if g.liquefied or not np.isfinite(g.nl) or felt[j] <= g.felt:
                continue
            g.felt = felt[j]
            g.ru = float(lq.ru_linear(g.felt, g.nl))
            if g.felt >= g.nl:
                g.liquefied = True
                g.ru = 1.0
                if g.liq_when == LIQ_NEVER:
                    g.liq_when = LIQ_AFTER

    def arr(name, dtype=float):
        return np.array([getattr(f, name) for f in feet], dtype=dtype)

    fs_eod = np.array([fs_current(f, pile, options) for f in feet])
    fs_ltc = np.array([fs_unliquefied(f, pile, options) for f in feet])
    res = predicted - target.capacity
    res = res[np.isfinite(res)]
    rms = float(np.sqrt(np.mean(res ** 2))) if res.size else float("nan")
    finite_pred = predicted[np.isfinite(predicted)]
    eod = float(finite_pred[-1]) if finite_pred.size else float("nan")

    m = np.isfinite(target.capacity) & np.isfinite(predicted)
    residual = float(predicted[m][-1] - target.capacity[m][-1]) if m.any() else 0.0
    last = feet[-1]
    shaft = shaft_long_term(fs_ltc, pile)
    toe = toe_end_of_drive(last.phi, last.sigma_v0, last.ru_toe, residual, pile, options)

    return DriveResult(
        pile_id=target.pile_id,
        elevation=elev.copy(),
        depth=arr("z"),
        sigma_v0=arr("sigma_v0"),
        bpf=arr("bpf"),
        fines=arr("fines"),
        phi=arr("phi"),
        n160=arr("n160"),
        csr=arr("csr"),
        crr15=arr("crr15"),
        cycles_to_liq=arr("nl"),
        felt_arrival=arr("felt_arrival"),
        felt=arr("felt"),
        ru=arr("ru"),
        ru_toe=arr("ru_toe"),
        liquefied=arr("liquefied", bool),
        liq_when=arr("liq_when", object),
        reason=arr("reason", object),
        fs_eod=fs_eod,
        fs_ltc=fs_ltc,
        target=target.capacity.copy(),
        predicted=predicted,
        top_phi=float(top_phi) if n_top > 0 else float("nan"),
        n_top=n_top,
        eod_capacity=eod,
        ltc=shaft + toe,
        ltc_shaft=shaft,
        toe_eod=toe,
        rms_kips=rms,
        pile=pile,
        options=options,
    )
