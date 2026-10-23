"""Model and Monte Carlo options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOptions:
    """Parameters of the drive model. Defaults are the values of Arnold (2027).

    Attributes
    ----------
    c1, c2 : float
        Multiplicative and additive (kips) recalibration coefficients of the
        Danish formula, Eq. 1 (Reddy and Stuedlein, 2013).
    phi_lo, phi_hi : float
        Bounds on the friction angle, deg (WSDOT GDM correlation range).
    g_ratio : float
        Ratio of the operative shear modulus to the small-strain modulus in
        the rigidity index, Eq. 3.
    k2_lo, k2_hi : float
        Seed and Idriss (1970) modulus coefficient K2,max at ``phi_lo`` and
        ``phi_hi``; interpolated linearly in friction angle.
    irr_bounds : tuple of float
        Clamp on the rigidity index.
    cc_delta_ratio : float
        Interface friction angle divided by the soil friction angle in the
        Coyle and Castello (1981) shaft resistance, Eq. 5.
    sr_psi_nc : float
        Normally consolidated strength ratio in the Randolph and Murphy (1985)
        interface factor applied to the residual strength, Eq. 10.
    zone_factor : float
        A foot starts accumulating blows when the toe is within
        ``zone_factor * width`` above it.
    b_exponent : float
        Exponent of the cyclic strength curve, Eq. 7.
    ru_cap_toe : float
        Largest pore-pressure ratio applied to the toe stress.
    dphi : float
        Shift of the friction angle correlation line, deg, Eq. 4.
    crr_multiplier : float
        Multiplier on the cyclic resistance ratio.
    smoothing_window, smoothing_order : int
        Savitzky-Golay window (feet) and polynomial order for the capacity
        target.
    tip_anchor : bool
        Replace the smoothed capacity at the final foot with the unsmoothed
        value.
    top_sweep_step : float
        Step of the top-segment friction angle sweep, deg.
    top_sweep_tol_frac, top_sweep_tol_kips : float
        A swept value is admissible when its log-fit RMS is within
        ``max(top_sweep_tol_frac * rms_min, top_sweep_tol_kips)`` of the
        minimum.
    max_iter : int
        Iteration limit of the per-foot solves.
    ru_tol : float
        Tolerance of the pore-pressure ratio fixed point.
    """

    c1: float = 0.45
    c2: float = 0.0
    phi_lo: float = 25.0
    phi_hi: float = 45.0
    g_ratio: float = 0.05
    k2_lo: float = 30.0
    k2_hi: float = 75.0
    irr_bounds: tuple = (10.0, 500.0)
    cc_delta_ratio: float = 0.8
    sr_psi_nc: float = 0.22
    zone_factor: float = 1.5
    b_exponent: float = 0.34
    ru_cap_toe: float = 0.9
    dphi: float = 0.0
    crr_multiplier: float = 1.0
    smoothing_window: int = 13
    smoothing_order: int = 2
    tip_anchor: bool = True
    top_sweep_step: float = 1.0
    top_sweep_tol_frac: float = 0.05
    top_sweep_tol_kips: float = 1.0
    max_iter: int = 60
    ru_tol: float = 1e-4


@dataclass(frozen=True)
class MonteCarloOptions:
    """Monte Carlo settings.

    Attributes
    ----------
    bias_cov : float
        Coefficient of variation of the lognormal capacity-formula bias
        (mean 1), drawn per realization.
    crr_sigma_ln : float
        Log standard deviation of the lognormal multiplier on the cyclic
        resistance ratio (median 1), drawn per realization.
    dphi_bound : float
        Half-width of the uniform shift of the friction angle correlation
        line, deg, drawn per realization.
    target_noise : bool
        Add independent normal noise to each foot's capacity target with the
        robust standard deviation of the raw-minus-smoothed residuals.
    ci_target : float
        Stop when the 95 percent confidence half-width of the mean long-term
        capacity is below this fraction of the mean.
    chunk : int
        Realizations between convergence checks.
    max_realizations : int
        Upper limit on realizations.
    seed : int or None
        Seed of the random number generator.
    """

    bias_cov: float = 0.35
    crr_sigma_ln: float = 0.20
    dphi_bound: float = 2.5
    target_noise: bool = True
    ci_target: float = 0.02
    chunk: int = 100
    max_realizations: int = 1200
    seed: int | None = None
