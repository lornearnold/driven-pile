# driven-pile: public API design

Status: revised 2026-10-23 after review; implemented the same day in
`src/driven_pile/`. Revised again 2026-10-23: `DriveLog` carries its `Pile`
and `Hammer` (decision 8); `SoilProfile` replaces `Site` and `FinesProfile`
(decision 9).
Equation numbers refer to the ASCE 2027 manuscript Methods section.

## 1. Scope and principles

- A plain Python library. No command-line interface, no file-system
  conventions, no module-level data loading. Everything the model needs is
  passed in as an argument.
- Implements exactly the method the paper describes: Eq. 1 through 10, the
  unlogged-top-segment sweep, end-of-drive to long-term capacity, and the
  five-source Monte Carlo. Retired exploration modes from the research repo
  are not ported.
- Soil classification is out of scope. Fines content is an input.
- Dependencies: numpy and scipy only. matplotlib is an optional extra used
  only by the `plot_*` helpers.
- Three layers so students can use the one-call version and researchers can
  reach the pieces:
  1. `analyze(...)` does everything.
  2. Pipeline functions (`prepare_target`, `sweep_top_phi`, `run_drive`,
     `long_term_capacity`, `monte_carlo`) do one stage each.
  3. Equation-level functions (`crr_15`, `nq_vesic`, `fs_coyle_castello`,
     ...) are importable for teaching and testing.

## 2. Units

US customary throughout, matching the construction records and the paper:
feet, kips, kips per square foot (ksf), degrees. The Danish formula inputs
(ram weight, pile modulus, area) are therefore kips, ksf, and square feet.
SI conversion helpers are deferred to a later release.

## 3. Inputs

### 3.1 `DriveLog`

The driving record as the inspector wrote it: one row per foot with blows
per minute (hammer speed, from which stroke is derived) and blows per foot.
Mirrors the research repo's `PileLog`, less the site-specific fields
(coordinates, schedule length, precomputed capacity).

```python
@dataclass
class DriveLog:
    pile: Pile                 # the pile that was driven
    hammer: Hammer             # the hammer that drove it
    elevation: np.ndarray      # ft, toe elevation at the end of each logged foot
    bpm: np.ndarray            # hammer blows per minute
    bpf: np.ndarray            # blows per foot
    pile_id: str | None = None # optional identifier, carried into Target and DriveResult

    @property
    def valid(self) -> np.ndarray       # bool, True where elevation, bpm and bpf are all finite
    @property
    def stroke(self) -> np.ndarray      # ft, from bpm via free-fall timing
    @property
    def depth(self) -> np.ndarray       # ft below the first logged foot

    @classmethod
    def from_arrays(cls, pile, hammer, *, bpf, bpm=None, stroke=None,
                    elevation=None, depth=None, tip_elevation=None,
                    forward_fill=False, pile_id=None) -> "DriveLog"
    @classmethod
    def from_csv(cls, path, pile, hammer, *, bpf, bpm=None, stroke=None,
                 elevation=None, depth=None, tip_elevation=None, skiprows=0,
                 forward_fill=False, pile_id=None) -> "DriveLog"
```

A log cannot exist without the pile and hammer that produced it, so both are
required to construct one. `Pile` and `Hammer` stay separate classes because
one hammer, and often one section, is shared across a project's logs.

- Elevation is either given directly or built from `depth` and
  `tip_elevation` as `tip_elevation + depth[-1] - depth`, the same rule the
  research reader uses. Exactly one of the two forms is required.
- `from_csv` takes column names (or zero-based indices) for the bpm, bpf,
  and depth or elevation columns and a `skiprows` count, so the Pier 4
  digitized logs load with `from_csv(path, pile_id=..., depth=0, bpm=1,
  bpf=2, tip_elevation=..., skiprows=4)`. There is no site-specific reader.
- Non-numeric cells become NaN and the log is trimmed to the first and last
  rows where bpm and bpf are both numeric. `valid` marks finite rows inside
  that span. `forward_fill` carries the last recorded rate over unrecorded
  feet inside the span.
- The log stores bpm and bpf; `stroke` is a derived property. A log whose
  inspector recorded the stroke in feet instead of the blow rate (10 of the
  15 Pier 4 benchmark logs) is loaded with `stroke=` and converted to bpm on
  input with `stroke_to_bpm`, the exact inverse of `bpm_to_stroke`. The
  caller says which it is; the package does not guess from magnitude the
  way the research reader does.
- `bpm_to_stroke(bpm)` and `stroke_to_bpm(stroke)` are public functions
  (free-fall stroke from the time between blows for an open-ended diesel
  hammer, and its inverse).

### 3.2 `Pile`

```python
@dataclass(frozen=True)
class Pile:
    area: float        # cross-section, ft^2 (A in Eq. 1 and the toe area)
    perimeter: float   # ft
    width: float       # ft, B: sets the felt-zone height 1.5B and the z/B decay in K
    length: float      # ft, L in Eq. 1 (total pile length)
    modulus: float     # ksf, E in Eq. 1

    @classmethod
    def octagonal(cls, width_in, length_ft, modulus_ksf=864_000.0) -> "Pile"
    @classmethod
    def square(cls, width_in, length_ft, modulus_ksf=864_000.0) -> "Pile"
    @classmethod
    def round(cls, diameter_in, length_ft, modulus_ksf=...) -> "Pile"
```

The paper's pile: `Pile.octagonal(24, 150)` gives area 3.3125 ft^2,
perimeter 6.627 ft, width 2.0 ft.

### 3.3 `Hammer`

```python
@dataclass(frozen=True)
class Hammer:
    ram_weight: float          # kips, W in Eq. 1
```

Stroke comes from the log. The regional recalibration coefficients of Eq. 1
are a property of the capacity formula, not the hammer, and live in
`ModelOptions`.

### 3.4 `SoilProfile`

```python
@dataclass(frozen=True)
class SoilProfile:
    mudline: float                      # ft, elevation of the soil surface the pile was driven through
    fines: np.ndarray                   # percent passing No. 200, at each row of elevation
    elevation: np.ndarray | None = None # ft; None for a uniform profile (fines is one value)
    water_elevation: float = 0.0        # ft; water table for sigma'_v0
    gamma_total: float = 0.122          # ksf/ft, total unit weight
    def fines_at(self, elevation) -> np.ndarray   # linear interpolation, end values held
```

One object per pile location: the mudline and the fines profile vary from
pile to pile. To reproduce the paper's piles, the per-foot fines content the
classifier produced is passed as the `elevation`, `fines` table.

### 3.5 `ModelOptions`

Only the knobs the paper defines. Defaults are the paper's values.

```python
@dataclass(frozen=True)
class ModelOptions:
    # capacity formula (Eq. 1): Puget Sound recalibrated Danish
    c1: float = 0.45               # multiplicative recalibration coefficient
    c2: float = 0.0                # kips, additive recalibration term
    # friction angle bounds (Eq. 4 and the a-priori sweep)
    phi_lo: float = 25.0
    phi_hi: float = 45.0
    # rigidity index (Eq. 3)
    g_ratio: float = 0.05          # G / G_max at bearing-failure strain
    k2_lo: float = 30.0            # Seed-Idriss K2,max at phi_lo
    k2_hi: float = 75.0            # Seed-Idriss K2,max at phi_hi
    irr_bounds: tuple = (10.0, 500.0)
    # shaft (Eq. 5, Eq. 10)
    cc_delta_ratio: float = 0.8    # delta/phi' in the Coyle & Castello branch
    sr_psi_nc: float = 0.22        # (su/sigma'v)_NC in the Randolph-Murphy alpha
    # liquefaction (Eq. 6-8)
    zone_factor: float = 1.5       # felt zone starts zone_factor * B above a foot
    b_exponent: float = 0.34
    ru_cap_toe: float = 0.9
    # perturbation hooks used by the Monte Carlo (0 and 1 = the paper line)
    dphi: float = 0.0              # shift of the WSDOT phi(N160) line, deg
    crr_multiplier: float = 1.0
    # preprocessing of the log into the capacity target
    smoothing_window: int = 13     # Savitzky-Golay window, feet
    smoothing_order: int = 2
    tip_anchor: bool = True        # keep the unsmoothed capacity at the final foot
    # unlogged top segment sweep
    top_sweep_step: float = 1.0    # deg
    top_sweep_tol_frac: float = 0.05
    top_sweep_tol_kips: float = 1.0
    # numerics
    max_iter: int = 60
    ru_tol: float = 1e-4
```

Fixed by the method, not options: 1-ft increments, linear ru growth
(Eq. 8), 15-cycle CRR reference, Vesic bearing factor, `max(...)` shaft law
(Eq. 10), Randolph-Murphy interface alpha, no toe setup, the reason codes
for feet that cannot be reconciled, and the top-segment point estimate at the
lowest admissible phi.

### 3.6 `MonteCarloOptions`

```python
@dataclass(frozen=True)
class MonteCarloOptions:
    bias_cov: float = 0.35         # lognormal, mean 1, per realization (Reddy & Stuedlein 2013)
    crr_sigma_ln: float = 0.20     # lognormal model error on CRR
    dphi_bound: float = 2.5        # uniform +/- deg on the phi(N160) line
    target_noise: bool = True      # per-foot N(0, robust sd of raw - smoothed)
    ci_target: float = 0.02        # stop when 95% CI half-width of mean LTC < this fraction
    chunk: int = 100
    max_realizations: int = 1200
    seed: int | None = None
```

The top-segment phi is always drawn uniform over the admissible band.

## 4. Pipeline functions

```python
def prepare_target(log, soil, options) -> Target
    # Eq. 1 per foot, Savitzky-Golay smoothing, tip anchor, extension of the
    # grid up to the mudline (unlogged top segment), sigma'_v0 profile, fines on grid.

def sweep_top_phi(target, options) -> TopPhiBand
    # constant-phi sweep over [phi_lo, phi_hi]; returns lo, hi and the RMS curve

def run_drive(target, options, top_phi: float) -> DriveResult
    # the top-down marching solve, per-foot characterization

def long_term_capacity(result) -> tuple[float, float]
    # (shaft_kips, toe_eod_kips); toe is the end-of-drive value, no toe setup

def monte_carlo(log, soil, options, mc_options) -> MonteCarloResult

def analyze(log, soil, options=None, mc_options=None, run_mc=False) -> Analysis
```

`Target` is a public dataclass (pile, elevation, capacity, bpf, fines,
sigma_v0, mudline, n_top) so a user can inspect or replace the capacity
target. It carries the pile and the soil state forward, so `sweep_top_phi`
and `run_drive` take only the target and the options. The
deterministic `analyze` uses the lowest admissible top-segment phi.

## 5. Results

Plain dataclasses holding numpy arrays, one entry per foot of the analysis
grid (unlogged top segment included). No table wrapper and no pandas.

```python
@dataclass
class DriveResult:
    elevation: np.ndarray
    depth: np.ndarray
    sigma_v0: np.ndarray
    fines: np.ndarray
    phi: np.ndarray
    n160: np.ndarray
    csr: np.ndarray
    crr15: np.ndarray
    cycles_to_liq: np.ndarray
    felt: np.ndarray
    ru: np.ndarray
    fs_eod: np.ndarray
    fs_ltc: np.ndarray
    reason: np.ndarray          # str codes: ok, above_ceiling, below_floor, nonfinite, top_segment
    target: np.ndarray          # smoothed capacity target per foot
    predicted: np.ndarray       # modeled at-time-of-driving capacity per foot
    top_phi: float
    options: ModelOptions
    eod_capacity: float
    ltc: float
    ltc_shaft: float
    toe_eod: float
    rms_kips: float
    def plot_profile(self, ax=None)   # phi, N160, ru vs elevation (requires matplotlib)
    def plot_fit(self, ax=None)       # target vs predicted capacity vs elevation

@dataclass
class MonteCarloResult:
    ltc: np.ndarray             # one value per realization
    shaft: np.ndarray
    toe_eod: np.ndarray
    bias: np.ndarray
    dphi: np.ndarray
    crr_multiplier: np.ndarray
    top_phi: np.ndarray
    phi_quantiles: np.ndarray   # shape (n_feet, 5): p5, p25, p50, p75, p95
    elevation: np.ndarray
    converged: bool
    def quantiles(self, q) -> np.ndarray
    def plot_ltc(self, ax=None)

@dataclass
class TopPhiBand:
    phi: np.ndarray             # swept values
    rms: np.ndarray             # log-fit RMS at each value, kips
    lo: float
    hi: float

@dataclass
class Analysis:
    drive: DriveResult
    top_band: TopPhiBand
    mc: MonteCarloResult | None
```

## 6. Module layout and mapping from the research repo

| new module | content | from (TCE_499_pile-driving) |
|---|---|---|
| `inputs.py` | DriveLog, Pile, Hammer, SoilProfile, bpm_to_stroke, stroke_to_bpm | pile_driving.PileLog, import_log_file (parsing only), bpm_to_stroke; DriveParams geometry fields |
| `options.py` | ModelOptions, MonteCarloOptions | DriveParams (46 fields -> 22), run_mc_ltc constants |
| `formulas.py` | Eq. 1 recalibrated Danish | pile_driving.danish_ps_pile_capacity, s0_pile_capacity |
| `preprocess.py` | smoothing, tip anchor, mudline extension, sigma'_v0 grid | bayes_classify.capacity_savgol, drive_model.extend_to_mudline, sigma_v0_profile, run_drive_model.load_target |
| `spt.py` | Eq. 4 and its inverse | liquefaction.wsdot_phi_from_n160, n160_from_phi |
| `bearing.py` | Eq. 2, 3: Vesic Nq*, stress-dependent rigidity index, phi inversion | soil_capacity._Nq_vesic, qp_cohesionless, DriveParams.irr_of, drive_model._phi_fast |
| `shaft.py` | Eq. 5, 10: Coyle & Castello K with Nordlund extension, partial-ru law, Randolph-Murphy alpha | drive_model.fs_coyle_castello, nordlund_kdelta, liq_interface_alpha, Sample.fs_* |
| `liquefaction.py` | Eq. 6-9: CSR, Boulanger & Idriss CRR, NL, linear ru, Kramer & Wang Sr | liquefaction.py (minus retired ru_seed) |
| `march.py` | Sample, characterize_foot, run_drive | drive_model.run_drive and helpers |
| `top_segment.py` | sweep_top_phi | run_drive_model.top_phi_range |
| `capacity.py` | long_term_capacity | drive_model.ltc_parts (eod toe only) |
| `montecarlo.py` | monte_carlo | run_mc_ltc.run_realization (no CSV cache, no multiprocessing coupling) |
| `results.py` | DriveResult, MonteCarloResult, TopPhiBand, Analysis | DriveResult.report, result CSV writers |
| `plots.py` | optional matplotlib helpers | figures/drive_model_profiles.py, mc_uncertainty.py (simplified) |

Not ported: bayes_classify, soil_inversion (superseded by drive_model),
pile_schedule, mudline interpolation, setup.csv truth, benchmark metrics,
Janbu / Gates / WSDOT formulas, and the DriveParams modes `blend`, `const`
alpha, `density` rigidity, `hybrid` and `beta2` shafts, `fs_lim_mode`,
`K_phi_slope`, `z_c`, `cc_k_mult`, `dphi_fc`, `irr_sigma='static'`,
`ru_model='seed'`, `fp_seed` variants other than `low`, and the unliquefied-toe
LTC variant.

## 7. Code conventions

- Docstrings follow the numpy convention (Parameters, Returns, Notes,
  References sections). Each equation-level function cites its equation
  number and source in References.
- PEP 8 and standard Python practice: type hints on public signatures,
  dataclasses for records, no module-level state, no bare `except`.
- Comments describe what the code does. Design rationale, history, and
  alternatives considered belong in docs/, not in source files.
- Tests use pytest. Each equation gets a hand-calculated anchor test; each
  benchmark pile gets a regression test against the frozen outputs.

## 8. Reproducing the paper

A test fixture per benchmark pile holds the Pier 4 log, mudline, per-foot
fines, and the frozen outputs (tagged `paper-freeze-2026-10-23` in the
research repo). Because the research code's point estimate used an
ill-defined argmin over a flat RMS band, the regression test compares the
package's lowest-admissible-phi result against a re-derived reference at that
same phi, and checks the admissible band itself against the frozen one.

## 9. Decisions recorded (review of 2026-10-23)

1. US customary units only in v0.1.
2. Fines content is an (elevation, fines) table or a single uniform value.
3. `Hammer` carries only ram weight. The Eq. 1 recalibration coefficients
   are in `ModelOptions`.
4. `DriveLog` mirrors `PileLog`: elevation, bpm, bpf, with stroke derived.
   No site-specific reader.
5. Monte Carlo is off by default in `analyze`.
6. Results are dataclasses of plain numpy arrays; no table wrapper, no pandas.
7. The deterministic top-segment phi is the lowest admissible value.
8. `DriveLog` requires a `Pile` and a `Hammer` (Lorne, 2026-10-23): a log
   cannot exist without them. `Target` carries the pile, so every pipeline
   function after the log takes only `(log or target, soil, options)`.
9. `Site` and `FinesProfile` are replaced by one `SoilProfile` (Lorne,
   2026-10-23): "Site" suggested a shared object when the mudline and fines
   vary per pile, and a separate class for one interpolated column was more
   structure than the concept needed. The mudline stays with the soil
   profile. `run_drive` and `sweep_top_phi` drop the soil argument, which
   they never read.
