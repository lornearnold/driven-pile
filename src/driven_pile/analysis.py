"""One-call analysis of a driving log."""

from __future__ import annotations

from .march import run_drive
from .montecarlo import monte_carlo
from .options import McOptions, ModelOptions
from .preprocess import prepare_target
from .results import Analysis
from .top_segment import sweep_top_phi


def analyze(log, site, options=None, mc_options=None, run_mc=False):
    """Characterize the soil from a driving log and estimate long-term capacity.

    Parameters
    ----------
    log : DriveLog
        The driving record, with its pile and hammer.
    site : Site
        Mudline, water table, unit weight and fines content.
    options : ModelOptions, optional
        Model parameters; defaults to the published values.
    mc_options : McOptions, optional
        Monte Carlo settings, used when ``run_mc`` is True.
    run_mc : bool, default False
        Also run the Monte Carlo simulation.

    Returns
    -------
    Analysis
        The deterministic run uses the lowest admissible top-segment
        friction angle.
    """
    options = options or ModelOptions()
    target = prepare_target(log, site, options)
    band = sweep_top_phi(target, site, options)
    drive = run_drive(target, site, options, top_phi=band.lo)
    mc = None
    if run_mc:
        mc = monte_carlo(log, site, options, mc_options or McOptions(),
                         target=target, top_band=band)
    return Analysis(drive=drive, top_band=band, mc=mc)
