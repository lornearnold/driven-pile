"""Liquefaction triggering under hammer blows and post-liquefaction strength.

Equations 6 through 9 of Arnold (2027). Stresses in ksf.
"""

from __future__ import annotations

import numpy as np

from .constants import P_ATM

N_REF_CYCLES = 15.0
"""Equivalent uniform cycles at the magnitude 7.5 reference of the CRR curve."""

N160_RESIDUAL_MAX = 15.0
"""Upper limit of the Kramer and Wang (2015) calibration on N1,60."""


def delta_n_fines(fines):
    """Fines-content adjustment to the clean-sand equivalent blow count.

    Parameters
    ----------
    fines : array_like
        Fines content, percent.

    Returns
    -------
    ndarray
        ``exp(1.63 + 9.7 / (FC + 0.01) - (15.7 / (FC + 0.01))**2)``.

    References
    ----------
    Boulanger and Idriss (2014), Eq. 2.23.
    """
    x = np.asarray(fines, dtype=float) + 0.01
    return np.exp(1.63 + 9.7 / x - (15.7 / x) ** 2)


def n160_clean_sand(n160, fines):
    """Clean-sand equivalent blow count ``N1,60cs = N1,60 + dN(FC)``."""
    return np.asarray(n160, dtype=float) + delta_n_fines(fines)


def crr_m75(n160cs):
    """Cyclic resistance ratio at the magnitude 7.5 reference and 1 atm.

    Parameters
    ----------
    n160cs : array_like
        Clean-sand equivalent blow count.

    Returns
    -------
    ndarray
        ``exp(N/14.1 + (N/126)**2 - (N/23.6)**3 + (N/25.4)**4 - 2.8)``.

    References
    ----------
    Boulanger and Idriss (2014), Eq. 2.25.
    """
    n = np.asarray(n160cs, dtype=float)
    return np.exp(n / 14.1 + (n / 126.0) ** 2 - (n / 23.6) ** 3 + (n / 25.4) ** 4 - 2.8)


def k_sigma(n160cs, sigma_v0, pa=P_ATM):
    """Overburden correction factor on the cyclic resistance ratio.

    Parameters
    ----------
    n160cs : array_like
        Clean-sand equivalent blow count, clamped to 37.
    sigma_v0 : array_like
        Vertical effective stress, ksf.
    pa : float
        Atmospheric pressure, ksf.

    Returns
    -------
    ndarray
        ``min(1 - C ln(sigma'v0 / pa), 1.1)`` with
        ``C = min(1 / (18.9 - 2.55 sqrt(N)), 0.3)``.

    References
    ----------
    Boulanger and Idriss (2014), Eqs. 2.16 and 2.16b.
    """
    n = np.minimum(np.asarray(n160cs, dtype=float), 37.0)
    c = np.minimum(1.0 / (18.9 - 2.55 * np.sqrt(n)), 0.3)
    sig = np.maximum(np.asarray(sigma_v0, dtype=float), 1e-4)
    return np.minimum(1.0 - c * np.log(sig / pa), 1.1)


def crr_15(n160, fines, sigma_v0):
    """In-situ cyclic resistance ratio at the 15-cycle reference.

    Parameters
    ----------
    n160 : array_like
        Corrected blow count N1,60.
    fines : array_like
        Fines content, percent.
    sigma_v0 : array_like
        Vertical effective stress, ksf.

    Returns
    -------
    ndarray
        ``CRR_M7.5(N1,60cs) * K_sigma``.
    """
    ncs = n160_clean_sand(n160, fines)
    return crr_m75(ncs) * k_sigma(ncs, sigma_v0)


def csr_driving(phi):
    """Cyclic stress ratio applied by a hammer blow that fails the soil.

    Parameters
    ----------
    phi : array_like
        Friction angle, deg.

    Returns
    -------
    ndarray
        ``tau_max / sigma'v0 = (1 - sin phi) tan phi``.

    References
    ----------
    Eq. 6 of Arnold (2027).
    """
    p = np.radians(np.asarray(phi, dtype=float))
    return (1.0 - np.sin(p)) * np.tan(p)


def cycles_to_liquefaction(csr, crr15, b=0.34, n_ref=N_REF_CYCLES, nl_max=1e9):
    """Uniform cycles at ``csr`` needed to trigger liquefaction.

    Parameters
    ----------
    csr : array_like
        Applied cyclic stress ratio.
    crr15 : array_like
        Cyclic resistance ratio at ``n_ref`` cycles.
    b : float
        Exponent of the cyclic strength curve.
    n_ref : float
        Reference cycle count of ``crr15``.
    nl_max : float
        Cap returned where the result is not finite or ``csr`` is not
        positive.

    Returns
    -------
    ndarray
        ``n_ref * (crr15 / csr)**(1 / b)``, capped at ``nl_max``.

    References
    ----------
    Idriss and Boulanger (2008); Eq. 7 of Arnold (2027).
    """
    csr = np.asarray(csr, dtype=float)
    crr15 = np.asarray(crr15, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        nl = n_ref * (crr15 / csr) ** (1.0 / b)
    nl = np.where(np.isfinite(nl) & (csr > 0), nl, nl_max)
    return np.minimum(nl, nl_max)


def ru_linear(n_felt, nl):
    """Pore-pressure ratio growing linearly with the cycle ratio.

    Parameters
    ----------
    n_felt : array_like
        Cycles (blows) felt.
    nl : array_like
        Cycles to liquefaction.

    Returns
    -------
    ndarray
        ``min(n_felt / nl, 1)``.

    References
    ----------
    Eq. 8 of Arnold (2027).
    """
    nl = np.maximum(np.asarray(nl, dtype=float), 1e-12)
    return np.minimum(np.asarray(n_felt, dtype=float) / nl, 1.0)


def residual_strength(n160, sigma_v0):
    """Residual strength of liquefied soil.

    Parameters
    ----------
    n160 : array_like
        Corrected blow count N1,60.
    sigma_v0 : array_like
        Vertical effective stress, ksf.

    Returns
    -------
    ndarray
        ``Pa exp(-8.444 + 0.109 N1,60 + 5.379 (sigma'v0 / Pa)**0.1)``, ksf.

    References
    ----------
    Kramer and Wang (2015); Eq. 9 of Arnold (2027).
    """
    sig_psf = np.asarray(sigma_v0, dtype=float) * 1000.0
    sr_psf = 2116.0 * np.exp(
        -8.444 + 0.109 * np.asarray(n160, dtype=float) + 5.379 * (sig_psf / 2116.0) ** 0.1
    )
    return sr_psf / 1000.0
