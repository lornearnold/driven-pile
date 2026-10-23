"""Shaft resistance: Coyle and Castello earth pressure (Eq. 5) and the
partially liquefied law (Eq. 10)."""

from __future__ import annotations

import numpy as np

_CC_PHI = np.array([30.0, 33.0, 36.0])
_CC_KT = np.array([1.33, 2.17, 4.49])

_ND_PHI = np.array([25.0, 30.0, 35.0, 40.0])
_ND_KD = np.array([[0.70, 0.85, 1.00],
                   [0.85, 1.15, 1.45],
                   [1.15, 1.75, 2.35],
                   [1.70, 3.00, 4.30]])
_ND_LOGV = np.log10(np.array([0.0093, 0.093, 0.93]))

_FT2_TO_M3_PER_M = 0.09290304


def nordlund_k_delta(phi, displaced_volume):
    """Nordlund lateral coefficient for a non-tapered displacement pile.

    Parameters
    ----------
    phi : array_like
        Friction angle, deg.
    displaced_volume : float
        Soil volume displaced per unit length, m^3/m.

    Returns
    -------
    ndarray
        K_delta, interpolated in log10 of the displaced volume and
        log-linearly in friction angle; the 35 to 40 deg slope is continued
        above 40 deg.

    References
    ----------
    Nordlund (1963, 1979) as tabulated in FHWA GEC-12 (Hannigan et al., 2016).
    """
    kd_v = np.array([np.interp(np.log10(displaced_volume), _ND_LOGV, row) for row in _ND_KD])
    lnk = np.log(kd_v)
    phi = np.asarray(phi, dtype=float)
    out = np.interp(phi, _ND_PHI, lnk)
    slope = (lnk[-1] - lnk[-2]) / (_ND_PHI[-1] - _ND_PHI[-2])
    out = np.where(phi > _ND_PHI[-1], lnk[-1] + slope * (phi - _ND_PHI[-1]), out)
    return np.exp(out)


def coyle_castello_k(phi, width):
    """Coyle and Castello earth pressure coefficient at the ground surface.

    Parameters
    ----------
    phi : array_like
        Friction angle, deg.
    width : float
        Pile width, ft.

    Returns
    -------
    ndarray
        K_t interpolated between the tabulated values at 30, 33 and 36 deg
        and extended above 36 deg in proportion to `nordlund_k_delta` at
        the pile's displaced volume.

    References
    ----------
    Coyle and Castello (1981).
    """
    phi = np.asarray(phi, dtype=float)
    kt = np.interp(phi, _CC_PHI, _CC_KT)
    v = width * width * _FT2_TO_M3_PER_M
    kt = np.where(phi > _CC_PHI[-1],
                  _CC_KT[-1] * nordlund_k_delta(phi, v) / nordlund_k_delta(_CC_PHI[-1], v),
                  kt)
    return kt


def fs_coyle_castello(sigma_v0, phi, z, width, delta_ratio=0.8):
    """Unit shaft resistance of a driven pile in sand.

    Parameters
    ----------
    sigma_v0 : array_like
        Vertical effective stress, ksf.
    phi : array_like
        Friction angle, deg.
    z : array_like
        Depth below the ground surface, ft.
    width : float
        Pile width, ft.
    delta_ratio : float
        Interface friction angle divided by ``phi``.

    Returns
    -------
    ndarray
        ``sigma'v0 K_t(phi) 10**(-0.017 z / B) tan(delta_ratio phi)``, ksf.

    References
    ----------
    Coyle and Castello (1981); Eq. 5 of Arnold (2027).
    """
    phi = np.asarray(phi, dtype=float)
    delta = np.radians(delta_ratio * phi)
    kt = coyle_castello_k(phi, width)
    z = np.asarray(z, dtype=float)
    return np.asarray(sigma_v0, dtype=float) * kt * 10.0 ** (-0.017 * z / width) * np.tan(delta)


def interface_alpha(sr, sigma_v0, psi_nc=0.22):
    """Interface factor on the residual strength of liquefied soil.

    Parameters
    ----------
    sr : float
        Residual strength, ksf.
    sigma_v0 : float
        Vertical effective stress, ksf.
    psi_nc : float
        Normally consolidated undrained strength ratio.

    Returns
    -------
    float
        ``min(1, sqrt(psi_nc / psi))`` for ``psi = sr / sigma'v0 <= 1`` and
        ``sqrt(psi_nc) psi**-0.25`` above; 1 when the inputs are not positive.

    References
    ----------
    Randolph and Murphy (1985), as adopted in API RP 2A.
    """
    if not (np.isfinite(sr) and sr > 0 and sigma_v0 > 0):
        return 1.0
    psi = sr / sigma_v0
    if psi <= 1.0:
        return float(min(1.0, np.sqrt(psi_nc / psi)))
    return float(np.sqrt(psi_nc) * psi ** -0.25)


def fs_partially_liquefied(fs_unliq, ru, sr, alpha):
    """Unit shaft resistance through soil at pore-pressure ratio ``ru``.

    Parameters
    ----------
    fs_unliq : float
        Unliquefied unit shaft resistance, ksf.
    ru : float
        Pore-pressure ratio.
    sr : float
        Residual strength of the liquefied soil, ksf.
    alpha : float
        Interface factor on ``sr``.

    Returns
    -------
    float
        ``max((1 - ru) fs_unliq, ru alpha sr)``.

    References
    ----------
    Boulanger and Brandenberg (2004); Eq. 10 of Arnold (2027).
    """
    return float(max((1.0 - ru) * fs_unliq, ru * alpha * sr))
