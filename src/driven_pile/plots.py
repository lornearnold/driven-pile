"""Matplotlib helpers. matplotlib is imported on first use."""

from __future__ import annotations

import numpy as np


def _axes(ax, ncols=1, figsize=None):
    import matplotlib.pyplot as plt

    if ax is not None:
        return np.atleast_1d(ax)
    _, axes = plt.subplots(1, ncols, figsize=figsize, sharey=True)
    return np.atleast_1d(axes)


def plot_profile(result, ax=None):
    """Friction angle, blow count and pore-pressure ratio versus elevation.

    Parameters
    ----------
    result : DriveResult
    ax : sequence of Axes, optional
        Three axes; created when omitted.

    Returns
    -------
    ndarray of Axes
    """
    axes = _axes(ax, ncols=3, figsize=(9, 6))
    e = result.elevation
    for a, y, label in zip(axes, (result.phi, result.n160, result.ru),
                           ("friction angle (deg)", "N1,60", "r_u at end of drive")):
        a.plot(y, e, drawstyle="steps-mid")
        a.set_xlabel(label)
        a.grid(True, alpha=0.3)
    axes[0].set_ylabel("elevation (ft)")
    axes[0].axhline(result.elevation[result.n_top] if result.n_top else e[0],
                    color="gray", lw=0.8, ls="--")
    return axes


def plot_fit(result, ax=None):
    """Capacity target and modeled capacity versus elevation.

    Parameters
    ----------
    result : DriveResult
    ax : Axes, optional

    Returns
    -------
    Axes
    """
    a = _axes(ax, figsize=(4.5, 6))[0]
    a.plot(result.target, result.elevation, label="target")
    a.plot(result.predicted, result.elevation, "--", label="modeled")
    a.set_xlabel("capacity at time of driving (kips)")
    a.set_ylabel("elevation (ft)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    a.grid(True, alpha=0.3)
    return a


def plot_ltc(mc, ax=None):
    """Histogram of the long-term capacity realizations.

    Parameters
    ----------
    mc : McResult
    ax : Axes, optional

    Returns
    -------
    Axes
    """
    a = _axes(ax, figsize=(5, 3.5))[0]
    a.hist(mc.ltc, bins=30, color="#4c72b0", alpha=0.8)
    for q in mc.quantiles([0.05, 0.5, 0.95]):
        a.axvline(q, color="k", lw=0.8, ls="--")
    a.set_xlabel("long-term capacity (kips)")
    a.set_ylabel("realizations")
    return a
