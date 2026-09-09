"""Core analysis functions for the Thickness Cycle Delta Analyzer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.signal as signal

OUTPUT_COLUMNS = [
    "Cycle",
    "Point A Index",
    "Point A Time",
    "Point A Thickness",
    "Point B Index",
    "Point B Time",
    "Point B Thickness",
    "Point C Index",
    "Point C Time",
    "Point C Thickness",
    "Point D Index",
    "Point D Time",
    "Point D Thickness",
    "Delta 1",
    "Delta 2",
    "Delta 3",
]


@dataclass
class RecoveryResult:
    min_indices: np.ndarray
    max_indices: np.ndarray
    recovered_min_indices: list[int]
    recovered_max_indices: list[int]


@dataclass
class CycleAnalysisResult:
    cycle_df: pd.DataFrame
    point_b_indices: list[int]
    point_c_indices: list[int]
    recovered_point_b_indices: list[int]
    recovered_point_c_indices: list[int]
    rejected_sequences: int
    derivative_failures: int

    @property
    def transition_indices(self) -> list[int]:
        """Backward-compatible alias for Point C indices."""
        return self.point_c_indices

    @property
    def recovered_transition_indices(self) -> list[int]:
        """Backward-compatible alias for recovered-cycle Point C indices."""
        return self.recovered_point_c_indices


def detect_extrema(
    thickness_values: np.ndarray,
    min_order: int,
    max_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect local minima and maxima with SciPy relative-extrema filters."""
    min_indices = signal.argrelmin(thickness_values, order=min_order)[0]
    max_indices = signal.argrelmax(thickness_values, order=max_order)[0]
    return min_indices.astype(int), max_indices.astype(int)


def build_events(
    min_indices: np.ndarray,
    max_indices: np.ndarray,
    time_values: np.ndarray,
    thickness_values: np.ndarray,
) -> pd.DataFrame:
    """Combine extrema into one time-ordered event table."""
    min_events = pd.DataFrame(
        {
            "Index": min_indices,
            "Type": "min",
            "Time": time_values[min_indices],
            "Thickness": thickness_values[min_indices],
        }
    )
    max_events = pd.DataFrame(
        {
            "Index": max_indices,
            "Type": "max",
            "Time": time_values[max_indices],
            "Thickness": thickness_values[max_indices],
        }
    )
    return (
        pd.concat([min_events, max_events], ignore_index=True)
        .sort_values("Index")
        .reset_index(drop=True)
    )


def find_suspect_regions(events: pd.DataFrame) -> list[dict]:
    """Find broken MIN/MAX alternation suggesting one missing opposite extremum."""
    records = events.to_dict("records")
    suspects: list[dict] = []

    for first, second in zip(records, records[1:]):
        if first["Type"] == "max" and second["Type"] == "max":
            missing_type = "min"
        elif first["Type"] == "min" and second["Type"] == "min":
            missing_type = "max"
        else:
            continue

        suspects.append(
            {
                "Missing Type": missing_type,
                "Start Index": int(first["Index"]),
                "End Index": int(second["Index"]),
                "Start Time": float(first["Time"]),
                "End Time": float(second["Time"]),
            }
        )
    return suspects


def recover_missing_extremum(
    thickness_values: np.ndarray,
    min_indices: np.ndarray,
    max_indices: np.ndarray,
    issue: dict,
    recovery_order: int,
) -> RecoveryResult:
    """Recover one missing extremum inside a suspicious same-type gap."""
    updated_mins = np.asarray(min_indices, dtype=int).copy()
    updated_maxs = np.asarray(max_indices, dtype=int).copy()
    recovered_mins: list[int] = []
    recovered_maxs: list[int] = []

    missing_type = issue["Missing Type"]
    region_start = int(issue["Start Index"])
    region_end = int(issue["End Index"])

    padding = max(recovery_order * 2, 5)
    padded_start = max(0, region_start - padding)
    padded_end = min(len(thickness_values) - 1, region_end + padding)
    local_thickness = thickness_values[padded_start : padded_end + 1]

    local_max_order = max(1, (len(local_thickness) - 1) // 2)
    effective_order = min(recovery_order, local_max_order)

    if missing_type == "min":
        candidates = signal.argrelmin(local_thickness, order=effective_order)[0]
        candidates = candidates + padded_start
        candidates = candidates[(candidates > region_start) & (candidates < region_end)]
        if len(candidates) > 0:
            recovered_idx = int(candidates[np.argmin(thickness_values[candidates])])
            updated_mins = np.unique(np.append(updated_mins, recovered_idx)).astype(int)
            recovered_mins.append(recovered_idx)
    else:
        candidates = signal.argrelmax(local_thickness, order=effective_order)[0]
        candidates = candidates + padded_start
        candidates = candidates[(candidates > region_start) & (candidates < region_end)]
        if len(candidates) > 0:
            recovered_idx = int(candidates[np.argmax(thickness_values[candidates])])
            updated_maxs = np.unique(np.append(updated_maxs, recovered_idx)).astype(int)
            recovered_maxs.append(recovered_idx)

    return RecoveryResult(
        min_indices=updated_mins,
        max_indices=updated_maxs,
        recovered_min_indices=recovered_mins,
        recovered_max_indices=recovered_maxs,
    )


def _effective_savgol_window(
    segment_length: int,
    requested_window: int,
    polyorder: int = 2,
) -> int | None:
    """Return a valid odd Savitzky-Golay window for a transition-search segment."""
    if segment_length < 3 or polyorder < 1:
        return None

    max_window = segment_length if segment_length % 2 == 1 else segment_length - 1
    window = max(int(requested_window), polyorder + 1)
    if window % 2 == 0:
        window += 1
    window = min(window, max_window)
    if window <= polyorder:
        return None
    return window


def _detect_side_transition_index(
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    start_idx: int,
    end_idx: int,
    direction: str,
    smoothing_window: int = 3,
    onset_fraction: float = 0.35,
    persistence: int = 2,
    polyorder: int = 2,
) -> int | None:
    """Find a distinct transition onset inside one side of a max-anchored cycle.

    ``direction="rising"`` searches A -> maximum for Point B.
    ``direction="falling"`` searches maximum -> D for Point C.

    The maximum is only an anchor that splits the cycle. It can never be returned
    as B or C because the selected transition must be strictly inside the search
    interval.
    """
    if direction not in {"rising", "falling"}:
        return None
    if not (0.01 <= float(onset_fraction) <= 0.95):
        return None

    persistence = int(persistence)
    if persistence < 1:
        return None
    if polyorder not in (2, 3):
        return None
    if end_idx <= start_idx:
        return None

    segment_time = np.asarray(time_values[start_idx : end_idx + 1], dtype=float)
    segment_thickness = np.asarray(
        thickness_values[start_idx : end_idx + 1], dtype=float
    )

    if len(segment_time) < 4:
        return None
    if not np.isfinite(segment_time).all() or not np.isfinite(segment_thickness).all():
        return None
    if np.any(np.diff(segment_time) <= 0):
        return None

    effective_window = _effective_savgol_window(
        len(segment_time), smoothing_window, polyorder
    )
    if effective_window is None:
        return None

    smoothed_thickness = signal.savgol_filter(
        segment_thickness,
        window_length=effective_window,
        polyorder=polyorder,
        mode="interp",
    )
    slope = np.gradient(smoothed_thickness, segment_time)

    process_strength = slope if direction == "rising" else -slope
    interior_strength = process_strength[1:-1]
    if len(interior_strength) == 0 or not np.isfinite(interior_strength).any():
        return None

    strongest_local_idx = int(np.nanargmax(interior_strength) + 1)
    strongest_strength = float(process_strength[strongest_local_idx])

    pre_transition_strength = process_strength[1:strongest_local_idx]
    if (
        len(pre_transition_strength) == 0
        or not np.isfinite(pre_transition_strength).any()
    ):
        return None

    baseline_count = max(
        1, min(5, int(np.ceil(len(pre_transition_strength) * 0.5)))
    )
    baseline_strength = float(
        np.nanmedian(pre_transition_strength[:baseline_count])
    )
    if not np.isfinite(baseline_strength) or strongest_strength <= baseline_strength:
        return None

    onset_threshold = baseline_strength + float(onset_fraction) * (
        strongest_strength - baseline_strength
    )

    onset_local_idx = strongest_local_idx
    while (
        onset_local_idx > 1
        and process_strength[onset_local_idx - 1] >= onset_threshold
    ):
        onset_local_idx -= 1

    run_length = strongest_local_idx - onset_local_idx + 1
    if run_length < persistence:
        return None

    if onset_local_idx <= 0 or onset_local_idx >= len(segment_time) - 1:
        return None

    return start_idx + onset_local_idx


def _detect_transition_index(
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    max_idx: int,
    min2_idx: int,
    smoothing_window: int = 3,
    onset_fraction: float = 0.35,
    persistence: int = 2,
    polyorder: int = 2,
    min_width: float | None = None,
) -> int | None:
    """Backward-compatible Point C wrapper for the falling side of a cycle."""
    del min_width
    return _detect_side_transition_index(
        time_values=time_values,
        thickness_values=thickness_values,
        start_idx=max_idx,
        end_idx=min2_idx,
        direction="falling",
        smoothing_window=smoothing_window,
        onset_fraction=onset_fraction,
        persistence=persistence,
        polyorder=polyorder,
    )


def calculate_cycles(
    events: pd.DataFrame,
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    recovered_min_indices: list[int] | None = None,
    recovered_max_indices: list[int] | None = None,
    transition_smoothing_window: int = 3,
    transition_onset_fraction: float = 0.35,
    transition_persistence: int = 2,
    transition_polyorder: int = 2,
    transition_min_width: float | None = None,
) -> CycleAnalysisResult:
    """Calculate A -> B -> max anchor -> C -> D cycles.

    A and D are successive minima. The detected maximum between them is used only
    as an anchor to split the search:
      - B = rising-side transition inside A -> maximum
      - C = falling-side transition inside maximum -> D

    A cycle is rejected unless both transitions are distinct.
    """
    del transition_min_width

    recovered_min_set = set(recovered_min_indices or [])
    recovered_max_set = set(recovered_max_indices or [])

    cycles: list[dict] = []
    point_b_indices: list[int] = []
    point_c_indices: list[int] = []
    recovered_point_b_indices: list[int] = []
    recovered_point_c_indices: list[int] = []
    rejected_sequences = 0
    derivative_failures = 0

    records = events.to_dict("records")

    for i in range(len(records) - 2):
        point_a_event = records[i]
        max_anchor_event = records[i + 1]
        point_d_event = records[i + 2]

        is_complete_cycle = (
            point_a_event["Type"] == "min"
            and max_anchor_event["Type"] == "max"
            and point_d_event["Type"] == "min"
        )
        if not is_complete_cycle:
            if point_a_event["Type"] == "min":
                rejected_sequences += 1
            continue

        point_a_idx = int(point_a_event["Index"])
        max_anchor_idx = int(max_anchor_event["Index"])
        point_d_idx = int(point_d_event["Index"])

        point_b_idx = _detect_side_transition_index(
            time_values=time_values,
            thickness_values=thickness_values,
            start_idx=point_a_idx,
            end_idx=max_anchor_idx,
            direction="rising",
            smoothing_window=transition_smoothing_window,
            onset_fraction=transition_onset_fraction,
            persistence=transition_persistence,
            polyorder=transition_polyorder,
        )
        point_c_idx = _detect_side_transition_index(
            time_values=time_values,
            thickness_values=thickness_values,
            start_idx=max_anchor_idx,
            end_idx=point_d_idx,
            direction="falling",
            smoothing_window=transition_smoothing_window,
            onset_fraction=transition_onset_fraction,
            persistence=transition_persistence,
            polyorder=transition_polyorder,
        )

        if point_b_idx is None or point_c_idx is None:
            derivative_failures += 1
            continue

        if not (
            point_a_idx < point_b_idx < max_anchor_idx < point_c_idx < point_d_idx
        ):
            derivative_failures += 1
            continue

        thickness_a = float(thickness_values[point_a_idx])
        thickness_b = float(thickness_values[point_b_idx])
        thickness_c = float(thickness_values[point_c_idx])
        thickness_d = float(thickness_values[point_d_idx])

        cycle_uses_recovered_extremum = (
            point_a_idx in recovered_min_set
            or point_d_idx in recovered_min_set
            or max_anchor_idx in recovered_max_set
        )
        if cycle_uses_recovered_extremum:
            recovered_point_b_indices.append(point_b_idx)
            recovered_point_c_indices.append(point_c_idx)
        else:
            point_b_indices.append(point_b_idx)
            point_c_indices.append(point_c_idx)

        cycles.append(
            {
                "Cycle": len(cycles) + 1,
                "Point A Index": point_a_idx,
                "Point A Time": float(time_values[point_a_idx]),
                "Point A Thickness": thickness_a,
                "Point B Index": point_b_idx,
                "Point B Time": float(time_values[point_b_idx]),
                "Point B Thickness": thickness_b,
                "Point C Index": point_c_idx,
                "Point C Time": float(time_values[point_c_idx]),
                "Point C Thickness": thickness_c,
                "Point D Index": point_d_idx,
                "Point D Time": float(time_values[point_d_idx]),
                "Point D Thickness": thickness_d,
                "Delta 1": thickness_b - thickness_a,
                "Delta 2": thickness_b - thickness_c,
                "Delta 3": thickness_c - thickness_d,
            }
        )

    cycle_df = format_cycle_results(pd.DataFrame(cycles))
    return CycleAnalysisResult(
        cycle_df=cycle_df,
        point_b_indices=sorted(set(point_b_indices)),
        point_c_indices=sorted(set(point_c_indices)),
        recovered_point_b_indices=sorted(set(recovered_point_b_indices)),
        recovered_point_c_indices=sorted(set(recovered_point_c_indices)),
        rejected_sequences=rejected_sequences,
        derivative_failures=derivative_failures,
    )


def format_cycle_results(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee stable output columns and ordering for display/CSV export."""
    cycle_df = cycle_df.copy()
    for column in OUTPUT_COLUMNS:
        if column not in cycle_df.columns:
            cycle_df[column] = np.nan
    return cycle_df[OUTPUT_COLUMNS]


def downsample_for_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample display data only; never use this output for calculations."""
    step = max(1, len(x_values) // max_points)
    return x_values[::step], y_values[::step]
