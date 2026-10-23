"""Write per-pile regression fixtures from the research repository.

Run from anywhere with the research repository's environment::

    cd /path/to/TCE_499_pile-driving
    uv run python /path/to/driven-pile/tests/fixtures/make_fixtures.py .

Each fixture holds the processed log, the site inputs, the capacity target
the research code produced, the admissible top-segment band from the frozen
sweep, and the research model's result at the lowest admissible top-segment
friction angle (tag paper-freeze-2026-10-23).
"""

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "src"))

import run_drive_model as rdm
from pile import drive_model as dm
from pile import pile_driving as pile

out_dir = Path(__file__).resolve().parent
logs, _ = pile.log_folder_import(verbose=False)
truth = rdm.load_setup_truth()
ps_table = rdm.load_p_sand()
params = rdm.make_params("partial", "coyle-irrs")
bands = rdm.load_top_phi_ranges()

with open(repo / "results" / "drive_model" / "eod_profiles_partial_coyle-irrs+ps.csv") as fh:
    fc_rows = {}
    for r in csv.DictReader(fh):
        fc_rows.setdefault(r["pile"], []).append((float(r["elev"]), float(r["FC"])))

for pid in sorted(truth):
    t = rdm.load_target(logs, pid, formula="danish_ps")
    if t is None:
        continue
    elev_t, cap_t, bpf_t, mudline = t
    L = logs[pid]
    m = L.valid
    p_sand = rdm.p_sand_on(elev_t, ps_table, pid)
    lo, hi, _best = bands[pid]
    res = dm.run_drive(pid, elev_t, cap_t, bpf_t, mudline,
                       replace(params, phi_top_mode="assigned", phi_top_assumed=lo),
                       p_sand=p_sand)
    qs, qp = dm.ltc_parts(res)
    d = res.to_dict()
    fines = sorted(fc_rows[pid])
    fixture = {
        "pile_id": pid,
        "length_ft": float(L.length),
        "mudline": float(mudline),
        "log": {
            "elevation": L.elevation[m].tolist(),
            # The research reader's bpm column holds the stroke in feet on
            # logs whose inspector recorded stroke; its bpm_to_stroke passes
            # values under 25 through unchanged.
            ("stroke" if max(L.bpm) < 25 else "bpm"): L.bpm[m].tolist(),
            "bpf": L.bpf[m].tolist(),
        },
        "fines": {"elevation": [e for e, _ in fines], "fines": [f for _, f in fines]},
        "target": {"elevation": elev_t.tolist(), "capacity": cap_t.tolist(),
                   "bpf": bpf_t.tolist()},
        "band": {"lo": lo, "hi": hi},
        "reference": {
            "top_phi": lo,
            "ltc": float(qs + qp), "shaft": float(qs), "toe_eod": float(qp),
            "rms_kips": float(res.report()["rms_residual_kips"]),
            "elevation": d["elev"].tolist(),
            "phi": d["phi"].tolist(),
            "n160": d["N160"].tolist(),
            "ru": d["ru"].tolist(),
            "ru_toe": d["ru_toe"].tolist(),
            "felt": d["felt"].tolist(),
            "predicted": d["predicted"].tolist(),
            "reason": [str(x) for x in d["reason"]],
        },
    }
    with open(out_dir / f"{pid}.json", "w") as fh:
        json.dump(fixture, fh)
    print(f"{pid:>7}: band {lo:.0f}-{hi:.0f}, ltc {qs + qp:7.1f} (shaft {qs:7.1f}, toe {qp:6.1f})")
