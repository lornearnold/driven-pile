"""From the driving log to the capacity target on the analysis grid."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import savgol_filter

from .constants import GAMMA_WATER
from .formulas import danish_capacity
from .inputs import Pile
from .options import ModelOptions


@dataclass
class Target:
    """Per-foot capacity target and static soil state on the analysis grid.

    The grid runs from one foot below the mudline to the final logged foot,
    top down. The first ``n_top`` entries are the unlogged top segment.

    Attributes
    ----------
    pile_id : str
        Identifier of the pile.
    pile : Pile
        The pile that was driven.
    elevation : ndarray
        Elevation of each foot, ft, descending.
    capacity : ndarray
        Smoothed capacity target, kips; NaN over the top segment.
    raw_capacity : ndarray
        Unsmoothed formula capacity, kips; NaN over the top segment.
    bpf : ndarray
        Blows per foot; 0 over the top segment.
    fines : ndarray
        Fines content, percent.
    sigma_v0 : ndarray
        Static vertical effective stress, ksf.
    mudline : float
        Elevation of the soil surface, ft.
    n_top : int
        Number of unlogged feet above the first logged foot.
    """

    pile_id: str
    pile: Pile
    elevation: np.ndarray
    capacity: np.ndarray
    raw_capacity: np.ndarray
    bpf: np.ndarray
    fines: np.ndarray
    sigma_v0: np.ndarray
    mudline: float
    n_top: int

    @property
    def n_logged(self) -> int:
        """Number of logged feet."""
        return len(self.elevation) - self.n_top

    def with_capacity(self, capacity) -> Target:
        """Copy of the target with a different capacity array."""
        return replace(self, capacity=np.asarray(capacity, dtype=float))


def smooth_capacity(capacity, window, order):
    """Savitzky-Golay smoothing of a capacity sequence.

    Parameters
    ----------
    capacity : array_like
        Capacity per foot in grid order, kips.
    window : int
        Window length in feet; reduced to an odd number no longer than the
        sequence when the sequence is shorter.
    order : int
        Polynomial order of the local fit.

    Returns
    -------
    ndarray
        Smoothed capacity; the input unchanged when the window does not
        exceed the polynomial order.
    """
    cap = np.asarray(capacity, dtype=float)
    n = len(cap)
    if n < window:
        window = n if n % 2 == 1 else n - 1
    if window <= order:
        return cap.copy()
    return savgol_filter(cap, window, order)


def sigma_v0_profile(elevation, surface, gamma_total, water_elevation=0.0,
                     gamma_water=GAMMA_WATER):
    """Static vertical effective stress at each elevation.

    Parameters
    ----------
    elevation : array_like
        Elevations, ft.
    surface : float
        Ground surface elevation, ft.
    gamma_total : float
        Total unit weight of the soil, ksf/ft.
    water_elevation : float
        Water table elevation, ft. Water standing above the surface adds to
        the total stress.
    gamma_water : float
        Unit weight of water, ksf/ft.

    Returns
    -------
    ndarray
        Effective stress, ksf, floored at zero.
    """
    elev = np.asarray(elevation, dtype=float)
    head_on_surface = max(water_elevation - surface, 0.0)
    total = gamma_water * head_on_surface + gamma_total * np.maximum(surface - elev, 0.0)
    pore = gamma_water * np.maximum(water_elevation - elev, 0.0)
    return np.maximum(total - pore, 0.0)


def extend_to_mudline(elevation, mudline):
    """Prepend the unlogged feet between the mudline and the first logged foot.

    Parameters
    ----------
    elevation : array_like
        Logged elevations, ft, descending.
    mudline : float
        Soil surface elevation, ft.

    Returns
    -------
    elevation : ndarray
        Grid elevations, one-foot steps from ``mudline - 1`` down to the
        first logged foot, followed by the logged elevations.
    n_top : int
        Number of prepended feet.
    """
    elev = np.asarray(elevation, dtype=float)
    first = float(elev.max())
    if first >= mudline:
        return elev, 0
    top = np.arange(mudline - 1.0, first, -1.0)
    return np.concatenate([top, elev]), len(top)


def prepare_target(log, soil, options=None, min_feet=20):
    """Build the capacity target from a driving log.

    Parameters
    ----------
    log : DriveLog
        The driving record, with its pile and hammer.
    soil : SoilProfile
        Mudline, water table, unit weight and fines content.
    options : ModelOptions, optional
        Formula coefficients and smoothing settings.
    min_feet : int
        Smallest number of valid logged feet accepted.

    Returns
    -------
    Target

    Raises
    ------
    ValueError
        If the log has fewer than ``min_feet`` valid feet.
    """
    options = options or ModelOptions()
    m = log.valid
    if m.sum() < min_feet:
        raise ValueError(f"{log.pile_id}: only {int(m.sum())} valid feet; need {min_feet}")
    elev = log.elevation[m]
    order = np.argsort(-elev)
    elev = elev[order]
    bpf = log.bpf[m][order]
    raw = danish_capacity(bpf, log.stroke[m][order], log.hammer.ram_weight, log.pile,
                          options.c1, options.c2)
    cap = smooth_capacity(raw, options.smoothing_window, options.smoothing_order)
    if options.tip_anchor:
        cap[-1] = raw[-1]

    grid, n_top = extend_to_mudline(elev, soil.mudline)
    pad = np.full(n_top, np.nan)
    surface = max(soil.mudline, float(grid.max()))
    return Target(
        pile_id=log.pile_id,
        pile=log.pile,
        elevation=grid,
        capacity=np.concatenate([pad, cap]),
        raw_capacity=np.concatenate([pad, raw]),
        bpf=np.concatenate([np.zeros(n_top), bpf]),
        fines=soil.fines_at(grid),
        sigma_v0=sigma_v0_profile(grid, surface, soil.gamma_total, soil.water_elevation),
        mudline=float(soil.mudline),
        n_top=n_top,
    )


def residual_scale(target):
    """Robust standard deviation of the raw-minus-smoothed capacity residuals.

    Parameters
    ----------
    target : Target

    Returns
    -------
    float
        ``1.4826 * median(|r - median(r)|)`` over the logged feet, kips.
    """
    m = np.isfinite(target.raw_capacity) & np.isfinite(target.capacity)
    res = target.raw_capacity[m] - target.capacity[m]
    return float(1.4826 * np.median(np.abs(res - np.median(res))))
