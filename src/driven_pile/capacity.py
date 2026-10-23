"""Long-term capacity from the end-of-drive characterization."""

from __future__ import annotations

import numpy as np

from .bearing import toe_bearing
from .constants import FOOT_INCREMENT


def shaft_long_term(fs_ltc, pile):
    """Long-term shaft resistance, kips.

    Parameters
    ----------
    fs_ltc : array_like
        Unit shaft resistance per foot after pore-pressure dissipation, ksf.
    pile : Pile
        Supplies the perimeter.
    """
    fs = np.asarray(fs_ltc, dtype=float)
    return float(np.nansum(fs) * pile.perimeter * FOOT_INCREMENT)


def toe_end_of_drive(phi, sigma_v0, ru_toe, residual, pile, options):
    """Toe resistance at the end of the drive, kips.

    Parameters
    ----------
    phi : float
        Friction angle of the final foot, deg.
    sigma_v0 : float
        Static vertical effective stress at the toe, ksf.
    ru_toe : float
        Pore-pressure ratio applied to the toe stress.
    residual : float
        Modeled minus target capacity at the final reconciled foot, kips;
        subtracted from the toe so the end-of-drive total matches the log.
    pile : Pile
        Supplies the toe area.
    options : ModelOptions
        Rigidity parameters.

    Returns
    -------
    float
        Vesic bearing on the reduced stress ``(1 - ru_toe) sigma'v0``, less
        the residual, floored at zero. No toe setup is modeled.
    """
    if not np.isfinite(phi):
        return 0.0
    qp = float(toe_bearing(sigma_v0 * (1.0 - ru_toe), phi, options)) * pile.area
    return max(qp - residual, 0.0)


def long_term_capacity(result):
    """Long-term shaft and toe resistance of a `DriveResult`.

    Parameters
    ----------
    result : DriveResult

    Returns
    -------
    shaft, toe : float
        Kips. The toe is its end-of-drive value.
    """
    shaft = shaft_long_term(result.fs_ltc, result.pile)
    m = np.isfinite(result.target) & np.isfinite(result.predicted)
    residual = float(result.predicted[m][-1] - result.target[m][-1]) if m.any() else 0.0
    toe = toe_end_of_drive(result.phi[-1], result.sigma_v0[-1], result.ru_toe[-1],
                           residual, result.pile, result.options)
    return shaft, toe
