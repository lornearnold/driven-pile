import json
from pathlib import Path

import numpy as np
import pytest

import driven_pile as dp

FIXTURES = Path(__file__).parent / "fixtures"
PAPER_PILE_AREA = 477.0 / 144.0


def fixture_ids():
    return sorted(p.stem for p in FIXTURES.glob("*.json"))


def load_fixture(pile_id):
    with open(FIXTURES / f"{pile_id}.json") as fh:
        return json.load(fh)


def paper_pile(length_ft):
    """The Pier 4 pile as the research code modeled it (477 in^2 nominal area)."""
    octagon = dp.Pile.octagonal(24.0, length_ft)
    return dp.Pile(PAPER_PILE_AREA, octagon.perimeter, octagon.width, length_ft, 864_000.0)


def paper_inputs(fx):
    log = dp.DriveLog.from_arrays(
        fx["pile_id"], paper_pile(fx["length_ft"]), dp.Hammer(ram_weight=20.0),
        bpf=fx["log"]["bpf"], bpm=fx["log"].get("bpm"), stroke=fx["log"].get("stroke"),
        elevation=fx["log"]["elevation"])
    soil = dp.SoilProfile(mudline=fx["mudline"], elevation=np.array(fx["fines"]["elevation"]),
                          fines=np.array(fx["fines"]["fines"]))
    return log, soil


@pytest.fixture(params=fixture_ids())
def fixture(request):
    return load_fixture(request.param)
