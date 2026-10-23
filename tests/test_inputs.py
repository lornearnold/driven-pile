import numpy as np
import pytest

import driven_pile as dp

PILE = dp.Pile.octagonal(24.0, 150.0)
HAMMER = dp.Hammer(ram_weight=20.0)


def test_octagonal_pile_geometry():
    p = dp.Pile.octagonal(24.0, 150.0)
    assert p.area * 144 == pytest.approx(477.2, abs=0.1)
    assert p.perimeter == pytest.approx(6.627, abs=1e-3)
    assert p.width == 2.0


def test_drive_log_trims_to_recorded_span_and_builds_elevation():
    bpm = ["", "", "40", "41", "42", "43", ""]
    bpf = ["", "", "10", "12", "x", "15", ""]
    depth = [1, 2, 3, 4, 5, 6, 7]
    log = dp.DriveLog.from_arrays("T-1", PILE, HAMMER, bpm=bpm, bpf=bpf, depth=depth,
                                  tip_elevation=-100.0)
    assert len(log.bpf) == 4
    assert np.isnan(log.bpf[2])
    assert log.valid.tolist() == [True, True, False, True]
    np.testing.assert_allclose(log.elevation, [-97.0, -98.0, -99.0, -100.0])


def test_drive_log_derives_stroke_from_bpm():
    log = dp.DriveLog.from_arrays("T-2", PILE, HAMMER, bpm=[40, 40], bpf=[10, 10],
                                  elevation=[-1, -2])
    np.testing.assert_allclose(log.stroke, dp.bpm_to_stroke(40.0))


def test_drive_log_from_stroke_stores_equivalent_bpm():
    log = dp.DriveLog.from_arrays("T-2s", PILE, HAMMER, stroke=[6.0, 7.5], bpf=[10, 10],
                                  elevation=[-1, -2])
    np.testing.assert_allclose(log.stroke, [6.0, 7.5])
    np.testing.assert_allclose(dp.stroke_to_bpm(dp.bpm_to_stroke([40.0, 55.0])), [40.0, 55.0])


def test_drive_log_forward_fills_bpm():
    log = dp.DriveLog.from_arrays("T-3", PILE, HAMMER, bpm=["40", "", "42"], bpf=[10, 11, 12],
                                  elevation=[-1, -2, -3], forward_fill=True)
    np.testing.assert_allclose(log.bpm, [40.0, 40.0, 42.0])


def test_drive_log_rejects_ambiguous_inputs():
    with pytest.raises(ValueError):
        dp.DriveLog.from_arrays("T", PILE, HAMMER, bpm=[40], stroke=[6], bpf=[1], elevation=[0])
    with pytest.raises(ValueError):
        dp.DriveLog.from_arrays("T", PILE, HAMMER, bpm=[40], bpf=[1], elevation=[0], depth=[0])
    with pytest.raises(ValueError):
        dp.DriveLog.from_arrays("T", PILE, HAMMER, bpm=[40], bpf=[1], depth=[0])


def test_drive_log_from_csv_by_index_with_header_lines(tmp_path):
    f = tmp_path / "log.csv"
    f.write_text("Pile Location, X-1\nTip elevation (feet), -50,,\n-------,,,\n"
                 "Depth (feet),Energy (H),Blows per foot,Notes\n"
                 "1,,,\n2,6.0,10,\n3,6.1,12,note\n4,6.2,14,\n")
    log = dp.DriveLog.from_csv(f, "X-1", PILE, HAMMER, depth=0, stroke=1, bpf=2,
                               tip_elevation=-50.0, skiprows=4)
    np.testing.assert_allclose(log.elevation, [-48.0, -49.0, -50.0])
    np.testing.assert_allclose(log.stroke, [6.0, 6.1, 6.2])
    np.testing.assert_allclose(log.bpf, [10, 12, 14])


def test_drive_log_from_csv_by_name(tmp_path):
    f = tmp_path / "log.csv"
    f.write_text("depth_ft,bpm,bpf\n1,40,10\n2,41,12\n")
    log = dp.DriveLog.from_csv(f, "N-1", PILE, HAMMER, depth="depth_ft", bpm="bpm", bpf="bpf",
                               tip_elevation=-20.0)
    np.testing.assert_allclose(log.elevation, [-19.0, -20.0])
    assert log.stroke[0] == pytest.approx(dp.bpm_to_stroke(40.0))


def test_fines_profile_interpolates_and_holds_ends():
    fp = dp.FinesProfile([-10.0, -30.0], [10.0, 50.0])
    np.testing.assert_allclose(fp.at([0.0, -20.0, -40.0]), [10.0, 30.0, 50.0])
    assert dp.FinesProfile.uniform(15.0).at(-123.0) == 15.0
