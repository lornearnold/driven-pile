"""Input records: the driving log, the pile, the hammer, and the site."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import G_FT_S2


def bpm_to_stroke(bpm):
    """Hammer stroke from the blow rate of a free-falling ram.

    Parameters
    ----------
    bpm : array_like
        Hammer blows per minute.

    Returns
    -------
    ndarray
        Stroke (drop height), ft.

    Notes
    -----
    The ram is in free flight for the time between blows, rising for half of
    it and falling for the other half, so ``h = g / 2 * (t / 2)**2`` with
    ``t = 60 / bpm`` seconds.
    """
    bpm = np.asarray(bpm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = 60.0 / bpm
    return G_FT_S2 / 2.0 * (t / 2.0) ** 2


def _to_float(values):
    """Parse a sequence of cells to floats, NaN where a cell is not a number."""
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        try:
            x = float(str(v).strip())
        except ValueError:
            continue
        if np.isfinite(x):
            out[i] = x
    return out


def _forward_fill(x):
    """Replace NaN after the first finite value with the last finite value."""
    x = np.array(x, dtype=float)
    finite = np.where(np.isfinite(x))[0]
    if finite.size == 0:
        return x
    last = np.nan
    for i in range(finite[0], len(x)):
        if np.isfinite(x[i]):
            last = x[i]
        else:
            x[i] = last
    return x


@dataclass
class DriveLog:
    """A pile-driving record, one entry per logged foot.

    Attributes
    ----------
    pile_id : str
        Identifier of the pile.
    elevation : ndarray
        Toe elevation at the end of each logged foot, ft.
    bpf : ndarray
        Blows per foot.
    stroke : ndarray
        Hammer stroke (drop height), ft.
    bpm : ndarray or None
        Hammer blows per minute as recorded, when the stroke was derived
        from it; None when the stroke was recorded directly.
    """

    pile_id: str
    elevation: np.ndarray
    bpf: np.ndarray
    stroke: np.ndarray
    bpm: np.ndarray | None = None

    def __post_init__(self):
        self.elevation = np.asarray(self.elevation, dtype=float)
        self.bpf = np.asarray(self.bpf, dtype=float)
        self.stroke = np.asarray(self.stroke, dtype=float)
        n = len(self.elevation)
        if len(self.bpf) != n or len(self.stroke) != n:
            raise ValueError("elevation, bpf and stroke must have the same length")
        if self.bpm is not None:
            self.bpm = np.asarray(self.bpm, dtype=float)
            if len(self.bpm) != n:
                raise ValueError("bpm must have the same length as elevation")

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask of feet with finite elevation, stroke and blow count."""
        return np.isfinite(self.elevation) & np.isfinite(self.stroke) & np.isfinite(self.bpf)

    @property
    def depth(self) -> np.ndarray:
        """Depth below the first logged foot, ft."""
        return np.nanmax(self.elevation) - self.elevation

    @classmethod
    def from_arrays(cls, pile_id, bpf, *, stroke=None, bpm=None, elevation=None,
                    depth=None, tip_elevation=None, forward_fill_stroke=False):
        """Build a log from per-foot arrays.

        Parameters
        ----------
        pile_id : str
            Identifier of the pile.
        bpf : array_like
            Blows per foot. Non-numeric entries become NaN.
        stroke, bpm : array_like, optional
            Hammer stroke (ft) or blow rate (blows per minute). Exactly one
            must be given; the stroke is derived from ``bpm`` with
            `bpm_to_stroke`.
        elevation : array_like, optional
            Toe elevation at the end of each foot, ft.
        depth, tip_elevation : array_like and float, optional
            Depth of each foot (ft) and the final toe elevation (ft), used
            instead of ``elevation``. The elevation of each foot is
            ``tip_elevation + depth[-1] - depth`` after trimming.
        forward_fill_stroke : bool, default False
            Carry the last recorded stroke (or blow rate) forward over
            unrecorded feet.

        Returns
        -------
        DriveLog
            The log trimmed to the span from the first to the last foot where
            both the stroke and the blow count are recorded.
        """
        if (stroke is None) == (bpm is None):
            raise ValueError("give exactly one of stroke or bpm")
        if (elevation is None) == (depth is None):
            raise ValueError("give exactly one of elevation or depth")
        if depth is not None and tip_elevation is None:
            raise ValueError("tip_elevation is required with depth")

        bpf = _to_float(bpf)
        source = _to_float(stroke if stroke is not None else bpm)
        if forward_fill_stroke:
            source = _forward_fill(source)
        n = len(bpf)
        if len(source) != n:
            raise ValueError("bpf and stroke/bpm must have the same length")

        both = np.isfinite(bpf) & np.isfinite(source)
        idx = np.where(both)[0]
        if idx.size == 0:
            raise ValueError("no foot has both a stroke and a blow count")
        sl = slice(idx[0], idx[-1] + 1)
        bpf, source = bpf[sl], source[sl]

        if elevation is not None:
            elevation = _to_float(elevation)
            if len(elevation) != n:
                raise ValueError("elevation must have the same length as bpf")
            elevation = elevation[sl]
        else:
            depth = _to_float(depth)
            if len(depth) != n:
                raise ValueError("depth must have the same length as bpf")
            depth = depth[sl]
            elevation = float(tip_elevation) + depth[-1] - depth

        if bpm is not None:
            return cls(pile_id, elevation, bpf, bpm_to_stroke(source), bpm=source)
        return cls(pile_id, elevation, bpf, source)

    @classmethod
    def from_csv(cls, path, pile_id, *, bpf, stroke=None, bpm=None, elevation=None,
                 depth=None, tip_elevation=None, skiprows=0, delimiter=",",
                 forward_fill_stroke=False):
        """Read a log from a delimited text file.

        Parameters
        ----------
        path : str or Path
            File to read.
        pile_id : str
            Identifier of the pile.
        bpf, stroke, bpm, elevation, depth : str or int
            Column holding each quantity, by header name or zero-based index.
            Exactly one of ``stroke``/``bpm`` and one of
            ``elevation``/``depth`` must be given.
        tip_elevation : float, optional
            Final toe elevation, ft; required with ``depth``.
        skiprows : int, default 0
            Lines to skip before the data (or before the header row when
            columns are named).
        delimiter : str, default ","
            Field delimiter.
        forward_fill_stroke : bool, default False
            See `from_arrays`.

        Returns
        -------
        DriveLog
        """
        specs = {"bpf": bpf, "stroke": stroke, "bpm": bpm,
                 "elevation": elevation, "depth": depth}
        given = {k: v for k, v in specs.items() if v is not None}
        with open(Path(path), newline="", encoding="utf-8-sig") as fh:
            rows = [r for r in csv.reader(fh, delimiter=delimiter)]
        rows = rows[skiprows:]
        if any(isinstance(v, str) for v in given.values()):
            header = [h.strip() for h in rows[0]]
            rows = rows[1:]
            for k, v in given.items():
                if isinstance(v, str):
                    if v not in header:
                        raise ValueError(f"column {v!r} not found in header {header}")
                    given[k] = header.index(v)
        rows = [r for r in rows if any(c.strip() for c in r)]

        def column(j):
            return [r[j] if j < len(r) else "" for r in rows]

        cols = {k: column(j) for k, j in given.items()}
        return cls.from_arrays(
            pile_id, cols["bpf"], stroke=cols.get("stroke"), bpm=cols.get("bpm"),
            elevation=cols.get("elevation"), depth=cols.get("depth"),
            tip_elevation=tip_elevation, forward_fill_stroke=forward_fill_stroke,
        )


@dataclass(frozen=True)
class Pile:
    """Pile geometry and stiffness.

    Attributes
    ----------
    area : float
        Cross-sectional area, ft^2.
    perimeter : float
        Perimeter, ft.
    width : float
        Width across flats (or diameter), ft.
    length : float
        Total pile length, ft.
    modulus : float
        Elastic modulus of the pile material, ksf.
    """

    area: float
    perimeter: float
    width: float
    length: float
    modulus: float

    @classmethod
    def octagonal(cls, width_in, length_ft, modulus_ksf=864_000.0):
        """Regular octagonal section from its width across flats in inches."""
        w = width_in / 12.0
        area = 2.0 * w * w * np.tan(np.pi / 8.0)
        perimeter = 8.0 * w * np.tan(np.pi / 8.0)
        return cls(float(area), float(perimeter), float(w), float(length_ft),
                   float(modulus_ksf))

    @classmethod
    def square(cls, width_in, length_ft, modulus_ksf=864_000.0):
        """Square section from its side in inches."""
        w = width_in / 12.0
        return cls(w * w, 4.0 * w, w, float(length_ft), float(modulus_ksf))

    @classmethod
    def round(cls, diameter_in, length_ft, modulus_ksf=864_000.0):
        """Solid circular section from its diameter in inches."""
        d = diameter_in / 12.0
        return cls(float(np.pi * d * d / 4.0), float(np.pi * d), d, float(length_ft),
                   float(modulus_ksf))


@dataclass(frozen=True)
class Hammer:
    """Driving hammer.

    Attributes
    ----------
    ram_weight : float
        Weight of the ram, kips.
    """

    ram_weight: float


@dataclass(frozen=True)
class FinesProfile:
    """Fines content versus elevation.

    Attributes
    ----------
    elevation : ndarray
        Elevations of the table rows, ft.
    fines : ndarray
        Fines content (percent passing the No. 200 sieve) at each row.
    """

    elevation: np.ndarray
    fines: np.ndarray

    def __post_init__(self):
        elevation = np.atleast_1d(np.asarray(self.elevation, dtype=float))
        fines = np.atleast_1d(np.asarray(self.fines, dtype=float))
        if elevation.shape != fines.shape:
            raise ValueError("elevation and fines must have the same length")
        order = np.argsort(elevation)
        object.__setattr__(self, "elevation", elevation[order])
        object.__setattr__(self, "fines", fines[order])

    @classmethod
    def uniform(cls, fines):
        """A profile with the same fines content at every elevation."""
        return cls(np.array([-1e6, 1e6]), np.array([fines, fines], dtype=float))

    def at(self, elevation):
        """Fines content interpolated at the given elevations, end values held."""
        return np.interp(np.asarray(elevation, dtype=float), self.elevation, self.fines)


@dataclass(frozen=True)
class Site:
    """Ground conditions at the pile.

    Attributes
    ----------
    mudline : float
        Elevation of the soil surface the pile was driven through, ft.
    fines : FinesProfile
        Fines content versus elevation.
    water_elevation : float, default 0.0
        Elevation of the water table, ft.
    gamma_total : float, default 0.122
        Total unit weight of the soil, ksf/ft.
    """

    mudline: float
    fines: FinesProfile
    water_elevation: float = 0.0
    gamma_total: float = 0.122
