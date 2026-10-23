# driven-pile: public API design (for review)

Status: proposal, 2026-10-23. Nothing here is implemented yet. Equation
numbers refer to the ASCE 2027 manuscript Methods section.

## 1. Scope and principles

- A plain Python library. No command-line interface, no file-system
  conventions, no module-level data loading. Everything the model needs is
  passed in as an argument.
- Implements exactly the method the paper describes: Eq. 1 through 10, the
  unlogged-top-segment sweep, end-of-drive to long-term capacity, and the
  five-source Monte Carlo. Retired exploration modes from the research repo
  are not ported.
- Soil classification is out of scope. Fines content is an input.
- Dependencies: numpy and scipy only. matplotlib and pandas are optional
  extras used only by `plot_*` methods and `to_dataframe()`.
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
(ram weight, pile modulus, area) are therefore kips, ksf, and square feet,
not the pounds, psi, and square inches the research code used. Conversion
helpers (`si_to_us`, `us_to_si`) are deferred to a later release.

## 3. Inputs

### 3.1 `DriveLog`

The driving record, one row per foot. Either depth or elevation may be given.

```python
@dataclass
class DriveLog:
    bpf: np.ndarray                 # blows per foot
    stroke: np.ndarray              # hammer stroke / drop height, ft
    elevation: np.ndarray           # ft, toe elevation at the end of each foot
    pile_id: str = ""

    @classmethod
    def from_arrays(cls, bpf, stroke, *, elevation=None, depth=None,
                    tip_elevation=None, pile_id="") -> "DriveLog"
    @classmethod
    def from_csv(cls, path, *, bpf="bpf", stroke="stroke",
                 elevation=None, depth=None, tip_elevation=None,
                 pile_id="") -> "DriveLog"
    @classmethod
    def from_tacoma_csv(cls, path) -> "DriveLog"   # the Pier 4 digitized-log layout
```

- `from_arrays` / `from_csv` accept either `elevation` directly or `depth`
  plus `tip_elevation` (elevation = tip_elevation + (depth.max() - depth)).
  Exactly one of the two forms is required.
- Rows with non-numeric or missing bpf or stroke are dropped and the log is
  trimmed to the first and last valid rows, as the research reader does.
  Stroke may be given as blows per minute for an open-ended diesel hammer via
  a `bpm_to_stroke` helper; the log stores stroke.
- `from_tacoma_csv` reads the `Pile Location, X / Tip elevation (feet), Y /
  header / Depth, Energy (H), Blows per foot, Notes` format so the paper's
  piles load with one call. It is the only site-specific reader.

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
    c1: float = 0.45           # regional recalibration coefficient (Reddy & Stuedlein 2013)
    c2: float = 0.0            # kips, additive term of the recalibration (0 for concrete)
```

Stroke comes from the log, not the hammer. Only the Puget Sound recalibrated
Danish formula (Eq. 1) is implemented; other dynamic formulas are not ported.

### 3.4 `Site`

```python
@dataclass(frozen=True)
class Site:
    mudline: float                      # ft, elevation of the soil surface the pile was driven through
    water_elevation: float = 0.0        # ft; water table for sigma'_v0
    gamma_total: float = 0.122          # ksf/ft, total unit weight
    fines: float | FinesProfile = 15.0  # fines content, percent
```

`FinesProfile` is a small table of (elevation, fines percent) interpolated
onto the analysis grid; a float means constant fines. This replaces the
classifier's per-foot p_sand. To reproduce the paper's piles, the per-foot FC
the classifier produced is passed in as a `FinesProfile`.

### 3.5 `ModelOptions`

Only the knobs the paper defines. Each field's docstring will cite the
equation and source. Defaults are the paper's values.

```python
@dataclass(frozen=True)
class ModelOptions:
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
    tip_anchor: str = "raw"        # keep the unsmoothed capacity at the final foot
    # unlogged top segment sweep
    top_sweep_step: float = 1.0    # deg
    top_sweep_tol_frac: float = 0.05
    top_sweep_tol_kips: float = 1.0
    # numerics
    max_iter: int = 60
    ru_tol: float = 1e-4
```

Fixed constants (not options, because the paper fixes them): 1-ft increments,
linear ru growth (Eq. 8), 15-cycle CRR reference, Vesic bearing factor,
`max(...)` shaft law (Eq. 10), Randolph-Murphy interface alpha, no toe setup,
Nq clamp policy and the "below floor" and "above ceiling" reason codes.

### 3.6 `McOptions`

```python
@dataclass(frozen=True)
class McOptions:
    bias_cov: float = 0.35         # lognormal, mean 1, per realization (Reddy & Stuedlein 2013)
    crr_sigma_ln: float = 0.20     # lognormal model error on CRR
    dphi_bound: float = 2.5        # uniform +/- deg on the phi(N160) line
    target_noise: bool = True      # per-foot N(0, robust sd of raw - smoothed)
    top_phi: str = "sweep"         # uniform over the a-priori admissible band
    ci_target: float = 0.02        # stop when 95% CI half-width of mean LTC < this fraction
    chunk: int = 100
    max_realizations: int = 1200
    seed: int | None = None
```

## 4. Pipeline functions

```python
def prepare_target(log, pile, hammer, site, options) -> Target
    # Eq. 1 per foot, Savitzky-Golay smoothing, raw tip anchor, extension of the
    # grid up to the mudline (unlogged top segment), sigma'_v0 profile, fines on grid.

def sweep_top_phi(target, pile, site, options) -> TopPhiBand
    # constant-phi sweep over [phi_lo, phi_hi]; returns (lo, hi, best, rms_curve)

def run_drive(target, pile, site, options, top_phi: float) -> DriveResult
    # the top-down marching solve (Levels 1-6), per-foot characterization

def long_term_capacity(result) -> tuple[float, float]
    # (shaft_kips, toe_eod_kips); toe is the end-of-drive value, no toe setup

def monte_carlo(log, pile, hammer, site, options, mc_options) -> McResult

def analyze(log, pile, hammer, site, options=None, mc_options=None,
            run_mc=True) -> Analysis
```

`Target` is an internal-but-public dataclass (elev, capacity, bpf, fines,
sigma_v0, n_top) so a user can inspect or replace the capacity target.

## 5. Results

```python
@dataclass
class DriveResult:
    profile: FootTable      # one row per foot: elev, z, sigma_v0, fines, phi, N160,
                            # CSR, CRR15, NL, felt_arrival, felt, ru, ru_toe,
                            # fs_eod, fs_ltc, reason, liquefied, liq_when
    target: np.ndarray      # smoothed capacity target per foot
    predicted: np.ndarray   # modeled at-time-of-driving capacity per foot
    top_phi: float
    options: ModelOptions
    # derived
    eod_capacity: float
    ltc: float
    ltc_shaft: float
    toe_eod: float
    fit: FitSummary         # rms_kips, pct_ok, pct_ceiling, pct_floor, pct_liquefied
    def to_dataframe(self)           # requires pandas
    def plot_profile(self, ax=None)  # phi, N160, ru vs elevation (requires matplotlib)
    def plot_fit(self, ax=None)      # target vs predicted capacity vs elevation

@dataclass
class McResult:
    ltc: np.ndarray          # one value per realization
    shaft: np.ndarray
    toe_eod: np.ndarray
    draws: DrawTable         # bias, dphi, crr_multiplier, top_phi per realization
    phi_quantiles: FootTable # p5/p25/p50/p75/p95 of phi per foot
    n: int
    converged: bool
    def quantiles(self, q) -> np.ndarray
    def summary(self) -> dict
    def plot_ltc(self, ax=None)

@dataclass
class Analysis:
    drive: DriveResult
    top_band: TopPhiBand
    mc: McResult | None
    # convenience pass-throughs: .ltc, .ltc_shaft, .toe_eod, .profile
```

`FootTable` is a thin wrapper over a structured numpy array with named columns
and `to_dataframe()`; it avoids a hard pandas dependency.

## 6. Module layout and mapping from the research repo

| new module | content | from (TCE_499_pile-driving) |
|---|---|---|
| `inputs.py` | DriveLog, Pile, Hammer, Site, FinesProfile | pile_driving.import_log_file (reader only), DriveParams geometry fields |
| `options.py` | ModelOptions, McOptions | DriveParams (46 fields -> ~22), run_mc_ltc constants |
| `formulas.py` | Eq. 1 recalibrated Danish | pile_driving.danish_ps_pile_capacity, s0_pile_capacity |
| `preprocess.py` | smoothing, tip anchor, mudline extension, sigma'_v0 grid | bayes_classify.capacity_savgol, drive_model.extend_to_mudline, sigma_v0_profile, run_drive_model.load_target |
| `spt.py` | Eq. 4 and its inverse | liquefaction.wsdot_phi_from_n160, n160_from_phi |
| `bearing.py` | Eq. 2, 3: Vesic Nq*, stress-dependent rigidity index, phi inversion | soil_capacity._Nq_vesic, qp_cohesionless, DriveParams.irr_of, drive_model._phi_fast |
| `shaft.py` | Eq. 5, 10: Coyle & Castello K with Nordlund extension, partial-ru law, Randolph-Murphy alpha | drive_model.fs_coyle_castello, nordlund_kdelta, liq_interface_alpha, Sample.fs_* |
| `liquefaction.py` | Eq. 6-9: CSR, Boulanger & Idriss CRR, NL, linear ru, Kramer & Wang Sr | liquefaction.py (as is, minus retired ru_seed) |
| `march.py` | Sample, characterize_foot, run_drive | drive_model.run_drive and helpers |
| `top_segment.py` | sweep_top_phi | run_drive_model.top_phi_range |
| `capacity.py` | long_term_capacity | drive_model.ltc_parts (eod toe only) |
| `montecarlo.py` | monte_carlo | run_mc_ltc.run_realization (no CSV cache, no multiprocessing coupling) |
| `results.py` | DriveResult, McResult, Analysis, FootTable | DriveResult.report, result CSV writers |
| `plots.py` | optional matplotlib helpers | figures/drive_model_profiles.py, mc_uncertainty.py (simplified) |

Not ported: bayes_classify, soil_inversion (superseded by drive_model),
pile_schedule, mudline interpolation, setup.csv truth, benchmark metrics,
Janbu / Gates / WSDOT formulas, and the DriveParams modes `blend`, `const`
alpha, `density` rigidity, `hybrid` and `beta2` shafts, `fs_lim_mode`,
`K_phi_slope`, `z_c`, `cc_k_mult`, `dphi_fc`, `irr_sigma='static'`,
`ru_model='seed'`, `fp_seed` variants other than `low`, and the unliquefied-toe
LTC variant.

## 7. Reproducing the paper

A test fixture per benchmark pile holds the Tacoma log, mudline, per-foot
fines, and the frozen outputs (tagged `paper-freeze-2026-10-23` in the research
repo). The regression test asserts the deterministic LTC, shaft, and toe match
to 0.1 kip and the Monte Carlo mean matches within its own CI at a fixed seed.

## 8. Questions for review

1. Units: US customary only for v0.1, with SI helpers later. Agree?
2. Fines input: constant or (elevation, FC) table. Is a third form needed,
   for example a layer list with USCS group symbols mapped to nominal FC?
3. `Hammer` carries only ram weight and the recalibration coefficients. Should
   it also hold a name and rated energy for reporting?
4. `DriveLog` stores stroke in feet. For diesel hammers the field record is
   often blows per minute; keep `bpm_to_stroke` as a public helper, or fold a
   `stroke_from="bpm"` option into the readers?
5. `analyze` runs the Monte Carlo by default (seconds to a minute per pile).
   Default on, or off?
6. Results as structured numpy with optional pandas, versus requiring pandas.
