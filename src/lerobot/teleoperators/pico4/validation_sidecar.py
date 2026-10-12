#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Optional raw-pose sidecar for quantitative PICO/NOKOV validation.

The sidecar is deliberately outside the robot observation schema: enabling it
does not add keys to teleoperation or recording. One instance is shared by the
left/right tracker readers and writes their native-rate samples plus a headset
stream into CSV files consumed by ``nokov_ego_hand_validation``.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from . import xrt_session

HEAD_FIELDS = (
    "sample_index",
    "head_timestamp_ns",
    "host_receive_perf_ns",
    "host_receive_unix_ns",
    "valid_numeric",
    "x_m",
    "y_m",
    "z_m",
    "qx",
    "qy",
    "qz",
    "qw",
)
TRACKER_FIELDS = (
    "sample_index",
    "motion_timestamp_ns",
    "host_receive_perf_ns",
    "host_receive_unix_ns",
    "tracker_index",
    "tracker_side",
    "tracker_serial",
    "selected",
    "valid_numeric",
    "duplicate",
    "age_s",
    "stale",
    "x_m",
    "y_m",
    "z_m",
    "qx",
    "qy",
    "qz",
    "qw",
    "world_x_m",
    "world_y_m",
    "world_z_m",
    "world_qx",
    "world_qy",
    "world_qz",
    "world_qw",
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "tcp_qx",
    "tcp_qy",
    "tcp_qz",
    "tcp_qw",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "wx_rad_s",
    "wy_rad_s",
    "wz_rad_s",
    "ax_m_s2",
    "ay_m_s2",
    "az_m_s2",
    "alphax_rad_s2",
    "alphay_rad_s2",
    "alphaz_rad_s2",
)


def _pose_valid(values: list[float]) -> bool:
    if len(values) != 7 or not all(math.isfinite(float(value)) for value in values):
        return False
    norm = math.sqrt(sum(float(value) ** 2 for value in values[3:7]))
    return max(abs(float(value)) for value in values[:3]) < 10_000.0 and 0.5 < norm < 1.5


def _xyzw_fields(prefix: str, pose_wxyz: list[float]) -> dict[str, float]:
    x, y, z, qw, qx, qy, qz = pose_wxyz
    return {
        f"{prefix}x_m": x,
        f"{prefix}y_m": y,
        f"{prefix}z_m": z,
        f"{prefix}qx": qx,
        f"{prefix}qy": qy,
        f"{prefix}qz": qz,
        f"{prefix}qw": qw,
    }


def _stable_head_snapshot(xrt: ModuleType, retries: int = 4) -> tuple[int, list[float]] | None:
    for _attempt in range(retries):
        before = int(xrt.get_time_stamp_ns())
        pose = [float(value) for value in xrt.get_headset_pose()]
        after = int(xrt.get_time_stamp_ns())
        if before == after and after > 0:
            return after, pose
    return None


class PicoTrackingValidationSidecar:
    """Thread-safe writer shared by both native-rate tracker pollers."""

    def __init__(
        self,
        output_dir: str | Path,
        tracker_serials_by_side: dict[str, str],
        *,
        overwrite: bool = False,
        head_poll_hz: float = 90.0,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._paths = {
            "head": self.output_dir / "pico_head_poses.csv",
            "trackers": self.output_dir / "pico_tracker_poses.csv",
            "events": self.output_dir / "events.csv",
            "metadata": self.output_dir / "capture_metadata.json",
        }
        existing = [path for path in self._paths.values() if path.is_file() and path.stat().st_size]
        if existing and not overwrite:
            listed = "\n".join(f"  {path}" for path in existing)
            raise FileExistsError(
                "PICO validation sidecar outputs already exist; choose a new session "
                f"or set --robot.tracker_validation_overwrite=true:\n{listed}"
            )
        self._serials = dict(tracker_serials_by_side)
        self._head_period_s = 1.0 / max(1e-3, float(head_poll_hz))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._head_thread: threading.Thread | None = None
        self._closed = False
        self._started_perf_ns = time.perf_counter_ns()
        self._started_unix_ns = time.time_ns()
        self._head_rows = 0
        self._tracker_rows = 0
        self._tracker_rows_by_side = dict.fromkeys(self._serials, 0)
        self._duplicates_by_side = dict.fromkeys(self._serials, 0)
        self._missing_polls_by_side = dict.fromkeys(self._serials, 0)
        self._stale_polls_by_side = dict.fromkeys(self._serials, 0)
        self._last_tracker_timestamp: dict[str, int] = {}
        self._last_head_timestamp = -1
        self._head_snapshot_failures = 0
        self._head_error: str | None = None

        self._head_stream = self._paths["head"].open("w", encoding="utf-8-sig", newline="")
        self._tracker_stream = self._paths["trackers"].open("w", encoding="utf-8-sig", newline="")
        self._head_writer = csv.DictWriter(self._head_stream, fieldnames=HEAD_FIELDS)
        self._tracker_writer = csv.DictWriter(self._tracker_stream, fieldnames=TRACKER_FIELDS)
        self._head_writer.writeheader()
        self._tracker_writer.writeheader()

    def start_head_capture(self) -> None:
        if self._head_thread is not None:
            return
        self._stop.clear()
        self._head_thread = threading.Thread(
            target=self._head_loop,
            name="pico-validation-head-poller",
            daemon=True,
        )
        self._head_thread.start()

    def _head_loop(self) -> None:
        xrt = xrt_session.module()
        if xrt is None:
            self._head_error = "xrt session was not initialized"
            return
        while not self._stop.wait(self._head_period_s):
            try:
                snapshot = _stable_head_snapshot(xrt)
                if snapshot is None:
                    self._head_snapshot_failures += 1
                    continue
                timestamp_ns, pose_xyzw = snapshot
                if timestamp_ns == self._last_head_timestamp:
                    continue
                host_perf_ns = time.perf_counter_ns()
                host_unix_ns = time.time_ns()
                row = {
                    "sample_index": self._head_rows,
                    "head_timestamp_ns": timestamp_ns,
                    "host_receive_perf_ns": host_perf_ns,
                    "host_receive_unix_ns": host_unix_ns,
                    "valid_numeric": int(_pose_valid(pose_xyzw)),
                    **dict(zip(("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"), pose_xyzw, strict=True)),
                }
                with self._lock:
                    if self._closed:
                        return
                    self._head_writer.writerow(row)
                    self._head_rows += 1
                    self._last_head_timestamp = timestamp_ns
                    if self._head_rows % 100 == 0:
                        self._head_stream.flush()
            except Exception as exc:  # pragma: no cover - native SDK dependent
                self._head_error = repr(exc)

    def record_tracker(self, sample: dict[str, Any]) -> None:
        """Receive one poll result from the actual ``Pico4TrackerReader``."""
        side = str(sample["side"])
        with self._lock:
            if self._closed:
                return
            if not sample.get("valid", True):
                self._missing_polls_by_side[side] = self._missing_polls_by_side.get(side, 0) + 1
                return
            timestamp_ns = int(sample["motion_timestamp_ns"])
            if timestamp_ns == self._last_tracker_timestamp.get(side):
                self._duplicates_by_side[side] = self._duplicates_by_side.get(side, 0) + 1
                if sample.get("stale"):
                    self._stale_polls_by_side[side] = self._stale_polls_by_side.get(side, 0) + 1
                return
            raw = [float(value) for value in sample["raw_pose_wxyz"]]
            world = [float(value) for value in sample["world_tracker_pose_wxyz"]]
            tcp = [float(value) for value in sample["tcp_pose_wxyz"]]
            velocity = [float(value) for value in sample.get("velocity", [math.nan] * 6)]
            acceleration = [float(value) for value in sample.get("acceleration", [math.nan] * 6)]
            row = {
                "sample_index": self._tracker_rows_by_side.get(side, 0),
                "motion_timestamp_ns": timestamp_ns,
                "host_receive_perf_ns": int(sample["host_receive_perf_ns"]),
                "host_receive_unix_ns": int(sample["host_receive_unix_ns"]),
                "tracker_index": int(sample["tracker_index"]),
                "tracker_side": side,
                "tracker_serial": str(sample["tracker_serial"]),
                "selected": 1,
                "valid_numeric": int(_pose_valid(raw)),
                "duplicate": int(bool(sample.get("duplicate", False))),
                "age_s": f"{float(sample.get('age_s', 0.0)):.9f}",
                "stale": int(bool(sample.get("stale", False))),
                **_xyzw_fields("", raw),
                **_xyzw_fields("world_", world),
                **_xyzw_fields("tcp_", tcp),
                **dict(zip(("vx_m_s", "vy_m_s", "vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s"), velocity, strict=True)),
                **dict(
                    zip(
                        ("ax_m_s2", "ay_m_s2", "az_m_s2", "alphax_rad_s2", "alphay_rad_s2", "alphaz_rad_s2"),
                        acceleration,
                        strict=True,
                    )
                ),
            }
            self._tracker_writer.writerow(row)
            self._tracker_rows += 1
            self._tracker_rows_by_side[side] = self._tracker_rows_by_side.get(side, 0) + 1
            self._last_tracker_timestamp[side] = timestamp_ns
            if self._tracker_rows % 100 == 0:
                self._tracker_stream.flush()

    def stop_head_capture(self) -> None:
        self._stop.set()
        if self._head_thread is not None:
            self._head_thread.join(timeout=2.0)
            self._head_thread = None

    def close(self) -> None:
        self.stop_head_capture()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            ended_perf_ns = time.perf_counter_ns()
            ended_unix_ns = time.time_ns()
            self._head_stream.flush()
            self._tracker_stream.flush()
            self._head_stream.close()
            self._tracker_stream.close()
            with self._paths["events"].open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("event", "perf_ns", "unix_ns", "note"))
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "event": "capture_started",
                            "perf_ns": self._started_perf_ns,
                            "unix_ns": self._started_unix_ns,
                            "note": "lerobot-teleoperate sidecar",
                        },
                        {"event": "capture_completed", "perf_ns": ended_perf_ns, "unix_ns": ended_unix_ns, "note": ""},
                    )
                )
            metadata = {
                "schema": "pico_teleoperate_tracking_sidecar_v1",
                "status": "complete"
                if self._head_rows >= 10 and all(value >= 10 for value in self._tracker_rows_by_side.values())
                else "completed_with_warnings",
                "producer": "xense-taccap-lerobot BiTaccapGripper via lerobot-teleoperate",
                "coordinate_frames": {
                    "raw": "PICO raw: +X right, +Y up, +Z toward user",
                    "world": "tracker.py FLU: +X forward, +Y left, +Z up",
                    "tcp": "world tracker pose right-multiplied by the configured tracker-to-TCP transform",
                },
                "pose_csv_quaternion_order": "xyzw",
                "motion_timestamp_source": (
                    "Motion.timeStampNs when positive; otherwise enclosing PICO state packet timeStampNs"
                ),
                "tracker_serials_by_side": self._serials,
                "rows": {
                    "head": self._head_rows,
                    "tracker_total": self._tracker_rows,
                    "tracker_by_side": self._tracker_rows_by_side,
                },
                "duplicate_polls_not_written_by_side": self._duplicates_by_side,
                "missing_polls_by_side": self._missing_polls_by_side,
                "selected_tracker_missing_frame_rows": sum(self._missing_polls_by_side.values()),
                "stale_duplicate_polls_by_side": self._stale_polls_by_side,
                "head_snapshot_failures": self._head_snapshot_failures,
                "head_snapshot_error": self._head_error,
                "head_snapshot_atomicity": "timestamp double-read best effort; native pose/timestamp use separate mutexes",
                "motion_snapshot_atomicity": (
                    "native get_motion_tracker_snapshot holds one motionMutex; "
                    "older bindings fall back to timestamp double-read"
                ),
                "started_unix_ns": self._started_unix_ns,
                "ended_unix_ns": ended_unix_ns,
                "duration_s": (ended_perf_ns - self._started_perf_ns) * 1e-9,
                "files": {
                    "head": self._paths["head"].name,
                    "trackers": self._paths["trackers"].name,
                    "events": self._paths["events"].name,
                },
            }
            self._paths["metadata"].write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    def __enter__(self) -> PicoTrackingValidationSidecar:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
