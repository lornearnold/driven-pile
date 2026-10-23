import numpy as np
import pytest

import driven_pile as dp


def test_octagonal_pile_geometry():
    p = dp.Pile.octagonal(24.0, 150.0)
    assert p.area * 144 == pytest.approx(477.2, abs=0.1)
    assert p.perimeter == pytest.approx(6.627, abs=1e-3)
    assert p.width == 2.0


def test_drive_log_trims_to_recorded_span_and_builds_elevation():
    bpf = ["", "", "10", "12", "x", "15", ""]
    stroke = ["", "", "6.0", "6.1", "6.2", "6.3", ""]
    depth = [1, 2, 3, 4, 5, 6, 7]
    log = dp.DriveLog.from_arrays("T-1", bpf, stroke=stroke, depth=depth, tip_elevation=-100.0)
    assert len(log.bpf) == 4
    assert np.isnan(log.bpf[2])
    assert log.valid.tolist() == [True, True, False, True]
    np.testing.assert_allclose(log.elevation, [-97.0, -98.0, -99.0, -100.0])


def test_drive_log_from_bpm_derives_stroke():
    log = dp.DriveLog.from_arrays("T-2", [10, 10], bpm=[40, 40], elevation=[-1, -2])
    np.testing.assert_allclose(log.stroke, dp.bpm_to_stroke(40.0))
    assert log.bpm is not None


def test_drive_log_forward_fills_stroke():
    log = dp.DriveLog.from_arrays("T-3", [10, 11, 12], stroke=["6", "", "7"],
                                  elevation=[-1, -2, -3], forward_fill_stroke=True)
    np.testing.assert_allclose(log.stroke, [6.0, 6.0, 7.0])


def test_drive_log_rejects_ambiguous_inputs():
    with pytest.raises(ValueError):
        dp.DriveLog.from_arrays("T", [1], stroke=[1], bpm=[1], elevation=[0])
    with pytest.raises(ValueError):
        dp.DriveLog.from_arrays("T", [1], stroke=[1], depth=[0])


def test_drive_log_from_csv_by_index_with_header_lines(tmp_path):
    f = tmp_path / "log.csv"
    f.write_text("Pile Location, X-1\nTip elevation (feet), -50,,\n-------,,,\n"
                 "Depth (feet),Energy (H),Blows per foot,Notes\n"
                 "1,,,\n2,6.0,10,\n3,6.1,12,note\n4,6.2,14,\n")
    log = dp.DriveLog.from_csv(f, "X-1", depth=0, stroke=1, bpf=2,
                               tip_elevation=-50.0, skiprows=4)
    np.testing.assert_allclose(log.elevation, [-48.0, -49.0, -50.0])
    np.testing.assert_allclose(log.bpf, [10, 12, 14])


def test_drive_log_from_csv_by_name(tmp_path):
    f = tmp_path / "log.csv"
    f.write_text("depth_ft,bpm,bpf\n1,40,10\n2,41,12\n")
    log = dp.DriveLog.from_csv(f, "N-1", depth="depth_ft", bpm="bpm", bpf="bpf",
                               tip_elevation=-20.0)
    np.testing.assert_allclose(log.elevation, [-19.0, -20.0])
    assert log.stroke[0] == pytest.approx(dp.bpm_to_stroke(40.0))


def test_fines_profile_interpolates_and_holds_ends():
    fp = dp.FinesProfile([-10.0, -30.0], [10.0, 50.0])
    np.testing.assert_allclose(fp.at([0.0, -20.0, -40.0]), [10.0, 30.0, 50.0])
    assert dp.FinesProfile.uniform(15.0).at(-123.0) == 15.0
