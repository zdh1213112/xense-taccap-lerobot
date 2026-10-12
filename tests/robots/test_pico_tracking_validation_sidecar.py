#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hardware-free tests for the opt-in teleoperate validation sidecar."""

import csv
import json

import pytest

from lerobot.teleoperators.pico4 import tracker as tracker_module
from lerobot.teleoperators.pico4.tracker import Pico4TrackerReader
from lerobot.teleoperators.pico4.validation_sidecar import (
    PicoTrackingValidationSidecar,
    _stable_head_snapshot,
)


class _StableHead:
    def get_time_stamp_ns(self):
        return 123456789

    def get_headset_pose(self):
        return [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]


class _TwoPollsThenStop:
    def __init__(self):
        self.calls = 0

    def wait(self, _timeout):
        self.calls += 1
        return self.calls > 2


class _MotionSDK:
    def get_motion_tracker_snapshot(self):
        return {
            "timestamp_ns": 987654321,
            "count": 1,
            "poses": [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]],
            "serials": ["LEFT-SN"],
            "velocities": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            "accelerations": [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]],
        }


def _tracker_sample(side: str, timestamp_ns: int) -> dict:
    return {
        "valid": True,
        "side": side,
        "tracker_index": 0 if side == "left" else 1,
        "tracker_serial": "LEFT-SN" if side == "left" else "RIGHT-SN",
        "motion_timestamp_ns": timestamp_ns,
        "host_receive_perf_ns": timestamp_ns + 10,
        "host_receive_unix_ns": timestamp_ns + 20,
        "duplicate": False,
        "age_s": 0.0,
        "stale": False,
        "raw_pose_wxyz": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        "world_tracker_pose_wxyz": [4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0],
        "tcp_pose_wxyz": [7.0, 8.0, 9.0, 0.0, 0.0, 1.0, 0.0],
        "velocity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "acceleration": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    }


def test_stable_head_snapshot_preserves_native_xyzw_order():
    assert _stable_head_snapshot(_StableHead()) == (
        123456789,
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
    )


def test_real_tracker_poller_callback_exposes_native_and_derived_poses(monkeypatch):
    samples = []
    reader = Pico4TrackerReader(
        tracker_sn="LEFT-SN",
        pico_to_world=False,
        validation_callback=samples.append,
        validation_side="left",
    )
    reader._resolved_sn = "LEFT-SN"
    reader._stop_evt = _TwoPollsThenStop()
    monkeypatch.setattr(tracker_module.xrt_session, "module", lambda: _MotionSDK())

    reader._poller_loop()

    assert len(samples) == 2
    assert samples[0]["side"] == "left"
    assert samples[0]["tracker_serial"] == "LEFT-SN"
    assert samples[0]["motion_timestamp_ns"] == 987654321
    assert samples[0]["duplicate"] is False
    assert samples[1]["duplicate"] is True
    assert samples[0]["motion_sequence"] == samples[1]["motion_sequence"] == 1
    assert samples[0]["raw_pose_wxyz"].tolist() == [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    assert samples[0]["world_tracker_pose_wxyz"].tolist() == samples[0]["raw_pose_wxyz"].tolist()
    assert samples[0]["tcp_pose_wxyz"].tolist() == samples[0]["raw_pose_wxyz"].tolist()
    assert samples[0]["velocity"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert samples[0]["acceleration"] == [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]


def test_sidecar_writes_raw_world_and_tcp_without_touching_schema(tmp_path):
    output = tmp_path / "pico"
    sidecar = PicoTrackingValidationSidecar(
        output,
        {"left": "LEFT-SN", "right": "RIGHT-SN"},
    )
    for index in range(12):
        sidecar.record_tracker(_tracker_sample("left", 1_000_000_000 + index))
        sidecar.record_tracker(_tracker_sample("right", 1_000_000_000 + index))

    duplicate = _tracker_sample("left", 1_000_000_011)
    duplicate.update(duplicate=True, stale=True, age_s=0.75)
    sidecar.record_tracker(duplicate)
    sidecar.record_tracker({"valid": False, "side": "right", "reason": "serial_not_visible"})
    sidecar.close()

    with (output / "pico_tracker_poses.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert rows[0]["tracker_side"] == "left"
    assert rows[0]["tracker_serial"] == "LEFT-SN"
    assert rows[0]["x_m"] == "1.0"
    assert rows[0]["qx"] == "0.0"
    assert rows[0]["qw"] == "1.0"
    assert rows[0]["world_x_m"] == "4.0"
    assert rows[0]["world_qx"] == "1.0"
    assert rows[0]["tcp_y_m"] == "8.0"
    assert rows[0]["tcp_qy"] == "1.0"

    metadata = json.loads((output / "capture_metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == "pico_teleoperate_tracking_sidecar_v1"
    assert metadata["rows"]["tracker_by_side"] == {"left": 12, "right": 12}
    assert metadata["duplicate_polls_not_written_by_side"]["left"] == 1
    assert metadata["stale_duplicate_polls_by_side"]["left"] == 1
    assert metadata["missing_polls_by_side"]["right"] == 1
    assert metadata["selected_tracker_missing_frame_rows"] == 1


def test_existing_output_requires_explicit_overwrite(tmp_path):
    output = tmp_path / "pico"
    PicoTrackingValidationSidecar(output, {"left": "LEFT-SN"}).close()

    with pytest.raises(FileExistsError, match="already exist"):
        PicoTrackingValidationSidecar(output, {"left": "LEFT-SN"})

    PicoTrackingValidationSidecar(output, {"left": "LEFT-SN"}, overwrite=True).close()
