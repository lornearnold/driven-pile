"""Dynamic capacity formula applied to the driving log."""

from __future__ import annotations

import numpy as np


def elastic_set(stroke, ram_weight, pile):
    """Elastic compression term of the Danish formula.

    Parameters
    ----------
    stroke : array_like
        Hammer stroke, ft.
    ram_weight : float
        Ram weight, kips.
    pile : Pile
        Pile geometry and modulus.

    Returns
    -------
    ndarray
        ``s0 = sqrt(W h L / (2 A E))``, ft.
    """
    stroke = np.asarray(stroke, dtype=float)
    return np.sqrt(ram_weight * stroke * pile.length / (2.0 * pile.area * pile.modulus))


def danish_capacity(bpf, stroke, ram_weight, pile, c1=0.45, c2=0.0):
    """Capacity at the time of driving from the recalibrated Danish formula.

    Parameters
    ----------
    bpf : array_like
        Blows per foot.
    stroke : array_like
        Hammer stroke, ft.
    ram_weight : float
        Ram weight, kips.
    pile : Pile
        Pile geometry and modulus.
    c1, c2 : float
        Multiplicative and additive (kips) recalibration coefficients.

    Returns
    -------
    ndarray
        ``c1 * W h / (s + s0) + c2``, kips, with the set per blow
        ``s = 1 / bpf`` ft.

    References
    ----------
    Sorensen and Hansen (1957); Reddy and Stuedlein (2013), Puget Sound
    recalibration with unadjusted hammer energy.
    """
    bpf = np.asarray(bpf, dtype=float)
    stroke = np.asarray(stroke, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1.0 / bpf
        q = ram_weight * stroke / (s + elastic_set(stroke, ram_weight, pile))
    return c1 * q + c2
