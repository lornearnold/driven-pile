"""Toe bearing in sand: Vesic cavity expansion with a stress-dependent rigidity
index (Eqs. 2 and 3), and its inversion for the friction angle."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .constants import ABOVE_CEILING, BELOW_FLOOR, NONFINITE, OK, P_ATM

_GRID_POINTS = 512


def nq_vesic(phi, irr, k0=None):
    """Vesic bearing capacity factor on vertical effective stress.

    Parameters
    ----------
    phi : array_like
        Friction angle, deg.
    irr : array_like
        Reduced rigidity index.
    k0 : array_like, optional
        At-rest earth pressure coefficient; ``1 - sin(phi)`` by default.

    Returns
    -------
    ndarray
        ``N_q* = N_sigma (1 + 2 K0) / 3`` with
        ``N_sigma = 3 / (3 - sin phi) exp((pi/2 - phi) tan phi)
        tan^2(45 + phi/2) Irr^(4 sin phi / (3 (1 + sin phi)))``.

    References
    ----------
    Vesic (1977); Eq. 2 of Arnold (2027).
    """
    p = np.radians(np.asarray(phi, dtype=float))
    sin_p = np.sin(p)
    if k0 is None:
        k0 = 1.0 - sin_p
    n_sigma = (
        (3.0 / (3.0 - sin_p))
        * np.exp((np.pi / 2.0 - p) * np.tan(p))
        * np.tan(np.pi / 4.0 + p / 2.0) ** 2
        * np.asarray(irr, dtype=float) ** (4.0 * sin_p / (3.0 * (1.0 + sin_p)))
    )
    return n_sigma * (1.0 + 2.0 * k0) / 3.0


def rigidity_index(phi, sigma_v, options):
    """Rigidity index from a stress-dependent shear modulus.

    Parameters
    ----------
    phi : array_like
        Friction angle, deg.
    sigma_v : array_like
        Vertical effective stress, ksf.
    options : ModelOptions
        Supplies ``g_ratio``, ``k2_lo``, ``k2_hi``, ``phi_lo``, ``phi_hi``
        and ``irr_bounds``.

    Returns
    -------
    ndarray
        ``I_r = g_ratio G_max / (sigma'_m tan phi)`` clamped to
        ``irr_bounds``, with ``sigma'_m = (1 + 2 K0) / 3 sigma'_v`` and
        ``G_max = 21.7 K2 Pa sqrt(sigma'_m / Pa)``.

    References
    ----------
    Seed and Idriss (1970); Eq. 3 of Arnold (2027).
    """
    phi = np.asarray(phi, dtype=float)
    p = np.radians(phi)
    k0 = 1.0 - np.sin(p)
    sig_m = np.maximum((1.0 + 2.0 * k0) / 3.0 * np.asarray(sigma_v, dtype=float), 1e-3)
    k2 = np.interp(phi, [options.phi_lo, options.phi_hi], [options.k2_lo, options.k2_hi])
    g_max = 21.7 * k2 * P_ATM * np.sqrt(sig_m / P_ATM)
    ir = options.g_ratio * g_max / (sig_m * np.tan(p))
    return np.clip(ir, *options.irr_bounds)


def toe_bearing(sigma_v, phi, options):
    """Unit toe resistance ``q_p = N_q*(phi, I_r) sigma'_v``, ksf."""
    irr = rigidity_index(phi, sigma_v, options)
    return np.asarray(sigma_v, dtype=float) * nq_vesic(phi, irr)


def phi_from_toe_bearing(q_needed, sigma_v, options):
    """Friction angle whose toe bearing on ``sigma_v`` equals ``q_needed``.

    Parameters
    ----------
    q_needed : float
        Required unit toe resistance, ksf.
    sigma_v : float
        Vertical effective stress at the toe, ksf.
    options : ModelOptions
        Supplies the friction angle bounds and rigidity parameters.

    Returns
    -------
    phi : float
        Friction angle, deg, held at a bound when the requirement lies
        outside the attainable range; NaN when ``q_needed`` is not finite.
    reason : str
        One of the reason codes in `driven_pile.constants`.
    """
    lo, hi = options.phi_lo, options.phi_hi
    if not np.isfinite(q_needed):
        return np.nan, NONFINITE
    if sigma_v <= 0:
        return lo, BELOW_FLOOR
    if q_needed <= toe_bearing(sigma_v, lo, options):
        return lo, BELOW_FLOOR
    if q_needed >= toe_bearing(sigma_v, hi, options):
        return hi, ABOVE_CEILING
    phi = brentq(lambda p: float(toe_bearing(sigma_v, p, options)) - q_needed, lo, hi)
    return phi, OK


def phi_from_toe_bearing_grid(q_needed, sigma_v, options):
    """Friction angle from toe bearing by interpolation on a fixed grid.

    Parameters
    ----------
    q_needed : float or array_like
        Required unit toe resistance, ksf.
    sigma_v : float or array_like
        Vertical effective stress at the toe, ksf.
    options : ModelOptions
        Supplies the friction angle bounds and rigidity parameters.

    Returns
    -------
    float or ndarray
        Friction angle, deg, clamped to the bounds. Faster than
        `phi_from_toe_bearing` when evaluated many times.
    """
    grid = np.linspace(options.phi_lo, options.phi_hi, _GRID_POINTS)
    sig = np.atleast_1d(np.asarray(sigma_v, dtype=float))
    q = np.broadcast_to(np.atleast_1d(np.asarray(q_needed, dtype=float)), sig.shape)
    out = np.empty_like(sig)
    for k, (s_k, q_k) in enumerate(zip(sig, q)):
        qp = toe_bearing(max(s_k, 1e-9), grid, options)
        out[k] = np.interp(q_k, qp, grid)
    return out if np.ndim(sigma_v) else float(out[0])
