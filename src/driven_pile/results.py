"""Result records."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import TOP_SEGMENT
from .options import ModelOptions


@dataclass
class DriveResult:
    """Soil characterization and capacities from one drive-model run.

    Per-foot arrays run over the analysis grid, top down, including the
    unlogged top segment.

    Attributes
    ----------
    pile_id : str
        Identifier of the pile.
    elevation : ndarray
        Elevation of each foot, ft.
    depth : ndarray
        Depth of each foot below the ground surface, ft.
    sigma_v0 : ndarray
        Static vertical effective stress, ksf.
    bpf : ndarray
        Blows per foot.
    fines : ndarray
        Fines content, percent.
    phi : ndarray
        Friction angle, deg; NaN where the foot was not characterized.
    n160 : ndarray
        Equivalent corrected blow count.
    csr : ndarray
        Cyclic stress ratio of a hammer blow.
    crr15 : ndarray
        Cyclic resistance ratio at 15 cycles.
    cycles_to_liq : ndarray
        Blows to liquefaction.
    felt_arrival : ndarray
        Blows felt before the toe reached the foot.
    felt : ndarray
        Blows felt by the end of the drive.
    ru : ndarray
        Pore-pressure ratio of each foot at the end of the drive.
    ru_toe : ndarray
        Pore-pressure ratio applied to the toe stress when the foot was
        characterized.
    liquefied : ndarray
        True where the foot liquefied.
    liq_when : ndarray
        "before", "during" or "after" the toe's arrival, or "" if never.
    reason : ndarray
        Outcome code of each foot's characterization.
    fs_eod : ndarray
        Unit shaft resistance at the end of the drive, ksf.
    fs_ltc : ndarray
        Unit shaft resistance after pore-pressure dissipation, ksf.
    target : ndarray
        Capacity target, kips.
    predicted : ndarray
        Modeled capacity at the time of driving, kips.
    top_phi : float
        Friction angle assigned to the top segment, deg; NaN if there is no
        top segment.
    n_top : int
        Number of feet in the top segment.
    eod_capacity : float
        Modeled capacity at the final foot, kips.
    ltc : float
        Long-term capacity, kips.
    ltc_shaft : float
        Long-term shaft resistance, kips.
    toe_eod : float
        Toe resistance at the end of the drive, kips.
    rms_kips : float
        Root-mean-square of ``predicted - target`` over the logged feet.
    pile : Pile
        Pile used for the run.
    options : ModelOptions
        Options used for the run.
    """

    pile_id: str
    elevation: np.ndarray
    depth: np.ndarray
    sigma_v0: np.ndarray
    bpf: np.ndarray
    fines: np.ndarray
    phi: np.ndarray
    n160: np.ndarray
    csr: np.ndarray
    crr15: np.ndarray
    cycles_to_liq: np.ndarray
    felt_arrival: np.ndarray
    felt: np.ndarray
    ru: np.ndarray
    ru_toe: np.ndarray
    liquefied: np.ndarray
    liq_when: np.ndarray
    reason: np.ndarray
    fs_eod: np.ndarray
    fs_ltc: np.ndarray
    target: np.ndarray
    predicted: np.ndarray
    top_phi: float
    n_top: int
    eod_capacity: float
    ltc: float
    ltc_shaft: float
    toe_eod: float
    rms_kips: float
    pile: object
    options: ModelOptions

    @property
    def n_logged(self) -> int:
        """Number of logged feet."""
        return len(self.elevation) - self.n_top

    @property
    def logged(self) -> np.ndarray:
        """Boolean mask of the logged feet."""
        return self.reason != TOP_SEGMENT

    def reason_counts(self) -> dict:
        """Count of each outcome code over the logged feet."""
        codes, counts = np.unique(self.reason[self.logged], return_counts=True)
        return dict(zip(codes.tolist(), counts.tolist()))

    def fraction_liquefied(self) -> float:
        """Fraction of logged feet that liquefied during the drive."""
        m = self.logged
        return float(np.mean(self.liquefied[m])) if m.any() else float("nan")

    def plot_profile(self, ax=None):
        """Plot friction angle, blow count and pore-pressure ratio versus elevation."""
        from . import plots

        return plots.plot_profile(self, ax=ax)

    def plot_fit(self, ax=None):
        """Plot the capacity target and the modeled capacity versus elevation."""
        from . import plots

        return plots.plot_fit(self, ax=ax)


@dataclass
class TopPhiBand:
    """Admissible friction angles for the unlogged top segment.

    Attributes
    ----------
    phi : ndarray
        Swept friction angles, deg.
    rms : ndarray
        Root-mean-square log-fit residual at each swept value, kips.
    lo, hi : float
        Lowest and highest admissible friction angles, deg.
    """

    phi: np.ndarray
    rms: np.ndarray
    lo: float
    hi: float


@dataclass
class McResult:
    """Monte Carlo realizations of the long-term capacity.

    Attributes
    ----------
    ltc : ndarray
        Long-term capacity per realization, kips.
    shaft : ndarray
        Long-term shaft resistance per realization, kips.
    toe_eod : ndarray
        End-of-drive toe resistance per realization, kips.
    bias : ndarray
        Capacity-formula bias drawn per realization.
    dphi : ndarray
        Shift of the friction angle correlation line drawn per realization, deg.
    crr_multiplier : ndarray
        Multiplier on the cyclic resistance ratio drawn per realization.
    top_phi : ndarray
        Top-segment friction angle drawn per realization, deg.
    elevation : ndarray
        Elevation of each foot of the analysis grid, ft.
    phi_quantiles : ndarray
        Friction angle percentiles 5, 25, 50, 75 and 95 per foot, shape
        ``(n_feet, 5)``.
    converged : bool
        True if the stopping criterion was met before the realization limit.
    """

    ltc: np.ndarray
    shaft: np.ndarray
    toe_eod: np.ndarray
    bias: np.ndarray
    dphi: np.ndarray
    crr_multiplier: np.ndarray
    top_phi: np.ndarray
    elevation: np.ndarray
    phi_quantiles: np.ndarray
    converged: bool
    percentiles: tuple = field(default=(5, 25, 50, 75, 95))

    @property
    def n(self) -> int:
        """Number of realizations."""
        return len(self.ltc)

    def quantiles(self, q):
        """Quantiles of the long-term capacity.

        Parameters
        ----------
        q : float or array_like
            Probabilities in [0, 1].

        Returns
        -------
        ndarray
            Capacity at each probability, kips.
        """
        return np.quantile(self.ltc, q)

    def mean_ci(self, z=1.96):
        """Mean long-term capacity and its confidence half-width, kips."""
        v = self.ltc
        return float(v.mean()), float(z * v.std(ddof=1) / np.sqrt(len(v)))

    def plot_ltc(self, ax=None):
        """Plot the distribution of long-term capacity."""
        from . import plots

        return plots.plot_ltc(self, ax=ax)


@dataclass
class Analysis:
    """Everything `analyze` produces.

    Attributes
    ----------
    drive : DriveResult
        Deterministic run at the lowest admissible top-segment friction angle.
    top_band : TopPhiBand
        Admissible top-segment friction angles.
    mc : McResult or None
        Monte Carlo realizations, when requested.
    """

    drive: DriveResult
    top_band: TopPhiBand
    mc: McResult | None = None
