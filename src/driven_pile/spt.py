"""Friction angle versus overburden-corrected blow count."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def phi_from_n160(n160, cap=45.0):
    """Friction angle from the corrected SPT blow count.

    Parameters
    ----------
    n160 : array_like
        Overburden- and energy-corrected blow count N1,60.
    cap : float, default 45.0
        Upper limit on the friction angle, deg.

    Returns
    -------
    ndarray
        ``min(cap, 26 + 5.5 log10(N1,60 + 0.01) + 0.11 N1,60)``, deg.

    References
    ----------
    WSDOT Geotechnical Design Manual (2022), Table 5-1 band midpoints.
    """
    n160 = np.asarray(n160, dtype=float)
    return np.minimum(26.0 + 5.5 * np.log10(n160 + 0.01) + 0.11 * n160, cap)


def n160_from_phi(phi, bounds=(0.1, 100.0), cap=45.0):
    """Corrected blow count whose correlated friction angle equals ``phi``.

    Parameters
    ----------
    phi : array_like
        Friction angle, deg.
    bounds : tuple of float
        Search bracket on N1,60. Values of ``phi`` at or beyond the
        correlation's range return the bracket edge.
    cap : float
        Cap passed to `phi_from_n160`.

    Returns
    -------
    float or ndarray
        N1,60, NaN where ``phi`` is not finite.
    """
    phi = np.asarray(phi, dtype=float)
    lo, hi = bounds
    phi_lo, phi_hi = phi_from_n160(lo, cap), phi_from_n160(hi, cap)

    def solve_one(p):
        if not np.isfinite(p):
            return np.nan
        if p <= phi_lo:
            return lo
        if p >= phi_hi:
            return hi
        return brentq(lambda n: float(phi_from_n160(n, cap)) - p, lo, hi)

    out = np.array([solve_one(p) for p in phi.ravel()]).reshape(phi.shape)
    return float(out) if out.ndim == 0 else out
