"""Monte Carlo propagation of the method's uncertainty sources."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .march import run_drive
from .options import McOptions, ModelOptions
from .preprocess import prepare_target, residual_scale
from .results import McResult
from .top_segment import sweep_top_phi


def monte_carlo(log, soil, options=None, mc_options=None, *,
                target=None, top_band=None):
    """Realizations of the long-term capacity under the five uncertainty sources.

    Parameters
    ----------
    log : DriveLog
    soil : SoilProfile
    options : ModelOptions, optional
    mc_options : McOptions, optional
    target : Target, optional
        Precomputed capacity target; built from the log when omitted.
    top_band : TopPhiBand, optional
        Precomputed admissible top-segment band; swept when omitted.

    Returns
    -------
    McResult

    Notes
    -----
    Each realization draws a capacity-formula bias (lognormal), a multiplier
    on the cyclic resistance ratio (lognormal), a shift of the friction angle
    correlation line (uniform), independent normal noise on each foot's
    capacity target, and a top-segment friction angle uniform over the
    admissible band. Realizations are added in chunks until the 95 percent
    confidence half-width of the mean long-term capacity is below
    ``mc_options.ci_target`` of the mean or ``mc_options.max_realizations``
    is reached.
    """
    options = options or ModelOptions()
    mc = mc_options or McOptions()
    if target is None:
        target = prepare_target(log, soil, options)
    if top_band is None:
        top_band = sweep_top_phi(target, options)
    rng = np.random.default_rng(mc.seed)

    noise_sd = residual_scale(target) if mc.target_noise else 0.0
    logged = np.isfinite(target.capacity)
    sln_bias = float(np.sqrt(np.log(1.0 + mc.bias_cov ** 2)))
    mu_bias = -0.5 * sln_bias ** 2

    draws = {k: [] for k in ("ltc", "shaft", "toe_eod", "bias", "dphi", "crr", "top_phi")}
    phis = []
    converged = False
    while len(draws["ltc"]) < mc.max_realizations:
        for _ in range(mc.chunk):
            dphi = float(rng.uniform(-mc.dphi_bound, mc.dphi_bound))
            crr = float(np.exp(rng.normal(0.0, mc.crr_sigma_ln)))
            bias = float(np.exp(rng.normal(mu_bias, sln_bias)))
            top_phi = float(rng.uniform(top_band.lo, top_band.hi))
            cap = target.capacity.copy()
            cap[logged] = np.maximum(
                bias * (cap[logged] + rng.normal(0.0, noise_sd, int(logged.sum()))), 1.0)
            opts = replace(options, dphi=dphi, crr_multiplier=crr)
            res = run_drive(target.with_capacity(cap), opts, top_phi=top_phi)
            draws["ltc"].append(res.ltc)
            draws["shaft"].append(res.ltc_shaft)
            draws["toe_eod"].append(res.toe_eod)
            draws["bias"].append(bias)
            draws["dphi"].append(dphi)
            draws["crr"].append(crr)
            draws["top_phi"].append(top_phi)
            phis.append(res.phi)
        v = np.asarray(draws["ltc"])
        half_width = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) / v.mean()
        if half_width < mc.ci_target:
            converged = True
            break

    phi_matrix = np.vstack(phis)
    pct = (5, 25, 50, 75, 95)
    phi_q = np.full((phi_matrix.shape[1], len(pct)), np.nan)
    for k in range(phi_matrix.shape[1]):
        col = phi_matrix[:, k]
        col = col[np.isfinite(col)]
        if col.size:
            phi_q[k] = np.percentile(col, pct)

    return McResult(
        ltc=np.asarray(draws["ltc"]),
        shaft=np.asarray(draws["shaft"]),
        toe_eod=np.asarray(draws["toe_eod"]),
        bias=np.asarray(draws["bias"]),
        dphi=np.asarray(draws["dphi"]),
        crr_multiplier=np.asarray(draws["crr"]),
        top_phi=np.asarray(draws["top_phi"]),
        elevation=target.elevation.copy(),
        phi_quantiles=phi_q,
        converged=converged,
        percentiles=pct,
    )
