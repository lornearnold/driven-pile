"""Admissible friction angles for the unlogged top segment."""

from __future__ import annotations

import numpy as np

from .march import run_drive
from .options import ModelOptions
from .results import TopPhiBand


def sweep_top_phi(target, site, options=None):
    """Sweep a constant top-segment friction angle and find the admissible band.

    Parameters
    ----------
    target : Target
    site : Site
    options : ModelOptions, optional
        Supplies the sweep range, step and tolerances.

    Returns
    -------
    TopPhiBand
        The band is the contiguous set of swept values around the best fit
        whose log-fit RMS is within
        ``max(top_sweep_tol_frac * rms_min, top_sweep_tol_kips)`` of the
        minimum. Values with equal RMS tie in favor of the lower angle.
    """
    options = options or ModelOptions()
    step = options.top_sweep_step
    phis = np.arange(options.phi_lo, options.phi_hi + 0.5 * step, step)
    rms = np.array([run_drive(target, site, options, top_phi=float(p)).rms_kips
                    for p in phis])
    k = int(np.argmin(rms))
    ok = rms <= rms[k] + max(options.top_sweep_tol_frac * rms[k], options.top_sweep_tol_kips)
    lo = k
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = k
    while hi < len(phis) - 1 and ok[hi + 1]:
        hi += 1
    return TopPhiBand(phi=phis, rms=rms, lo=float(phis[lo]), hi=float(phis[hi]))
