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
    "Max Anchor Index",
    "Max Anchor Time",
    "Max Anchor Thickness",
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
        return self.point_c_indices

    @property
    def recovered_transition_indices(self) -> list[int]:
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
    """Return a valid odd Savitzky-Golay window."""
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


def _detect_plateau_transition_indices(
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    min1_idx: int,
    max_idx: int,
    min2_idx: int,
    smoothing_window: int = 3,
    plateau_fraction: float = 0.35,
    polyorder: int = 2,
) -> tuple[int, int] | None:
    """Locate B and C as the two edges of the purge/plateau around the maximum.

    A and D are minima and the detected maximum is only an anchor. The interval
    slopes are measured after light smoothing. The active rise before the maximum
    and active fall after it establish reference slope magnitudes. Starting at the
    maximum, the detector expands left and right through intervals whose absolute
    slope is small relative to those active slopes.

    B is the left edge of that low-slope plateau and C is the right edge. Small
    positive or negative drift during purge is therefore allowed. If the trace
    changes directly from rise to fall with no resolved plateau, B and C may both
    equal the maximum anchor.
    """
    if not (0.01 <= float(plateau_fraction) <= 0.95):
        return None
    if polyorder not in (2, 3):
        return None
    if not (min1_idx < max_idx < min2_idx):
        return None

    segment_time = np.asarray(time_values[min1_idx : min2_idx + 1], dtype=float)
    segment_thickness = np.asarray(
        thickness_values[min1_idx : min2_idx + 1], dtype=float
    )

    if len(segment_time) < 3:
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

    smoothed = signal.savgol_filter(
        segment_thickness,
        window_length=effective_window,
        polyorder=polyorder,
        mode="interp",
    )
    interval_slopes = np.diff(smoothed) / np.diff(segment_time)

    max_local_idx = max_idx - min1_idx
    left_slopes = interval_slopes[:max_local_idx]
    right_slopes = interval_slopes[max_local_idx:]

    positive_rise = left_slopes[left_slopes > 0]
    negative_fall = -right_slopes[right_slopes < 0]
    if len(positive_rise) == 0 or len(negative_fall) == 0:
        return None

    # A percentile is less sensitive to one noisy derivative spike than max().
    rise_reference = float(np.nanpercentile(positive_rise, 75))
    fall_reference = float(np.nanpercentile(negative_fall, 75))
    if (
        not np.isfinite(rise_reference)
        or not np.isfinite(fall_reference)
        or rise_reference <= 0
        or fall_reference <= 0
    ):
        return None

    rise_plateau_limit = float(plateau_fraction) * rise_reference
    fall_plateau_limit = float(plateau_fraction) * fall_reference

    # Walk left from the maximum through the low-slope purge region.
    left_interval_idx = max_local_idx - 1
    while (
        left_interval_idx >= 0
        and abs(interval_slopes[left_interval_idx]) <= rise_plateau_limit
    ):
        left_interval_idx -= 1
    point_b_local_idx = left_interval_idx + 1

    # Walk right from the maximum through the low-slope purge region.
    right_interval_idx = max_local_idx
    while (
        right_interval_idx < len(interval_slopes)
        and abs(interval_slopes[right_interval_idx]) <= fall_plateau_limit
    ):
        right_interval_idx += 1
    point_c_local_idx = right_interval_idx

    # B must come after A and there must be a real positive-rise regime before B.
    if point_b_local_idx <= 0:
        return None
    if not np.any(left_slopes[:point_b_local_idx] > rise_plateau_limit):
        return None

    # C must come before D and there must be a real negative-fall regime after C.
    if point_c_local_idx >= len(segment_time) - 1:
        return None
    if not np.any(interval_slopes[point_c_local_idx:] < -fall_plateau_limit):
        return None

    if not (
        0
        < point_b_local_idx
        <= max_local_idx
        <= point_c_local_idx
        < len(segment_time) - 1
    ):
        return None

    return (
        min1_idx + point_b_local_idx,
        min1_idx + point_c_local_idx,
    )


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
    """Backward-compatible one-sided Point C detector after a maximum anchor."""
    del persistence, min_width

    if not (max_idx < min2_idx):
        return None

    segment_time = np.asarray(time_values[max_idx : min2_idx + 1], dtype=float)
    segment_thickness = np.asarray(
        thickness_values[max_idx : min2_idx + 1], dtype=float
    )
    if len(segment_time) < 2 or np.any(np.diff(segment_time) <= 0):
        return None

    effective_window = _effective_savgol_window(
        len(segment_time), smoothing_window, polyorder
    )
    if effective_window is None:
        return None

    smoothed = signal.savgol_filter(
        segment_thickness,
        window_length=effective_window,
        polyorder=polyorder,
        mode="interp",
    )
    slopes = np.diff(smoothed) / np.diff(segment_time)
    negative_fall = -slopes[slopes < 0]
    if len(negative_fall) == 0:
        return None

    fall_reference = float(np.nanpercentile(negative_fall, 75))
    plateau_limit = float(onset_fraction) * fall_reference

    interval_idx = 0
    while interval_idx < len(slopes) and abs(slopes[interval_idx]) <= plateau_limit:
        interval_idx += 1

    if interval_idx >= len(slopes):
        return None
    return max_idx + interval_idx


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
    """Calculate max-anchored cycles using the two edges of the purge plateau.

    A and D are successive minima. The detected maximum anchors the low-slope
    purge/plateau region:
      - B = left edge of the plateau after the active rise
      - C = right edge of the plateau before the active fall

    The maximum may equal B, C, or both. A cycle is rejected when a meaningful
    rise, plateau boundary, or fall cannot be resolved.
    """
    del transition_persistence, transition_min_width

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

        transitions = _detect_plateau_transition_indices(
            time_values=time_values,
            thickness_values=thickness_values,
            min1_idx=point_a_idx,
            max_idx=max_anchor_idx,
            min2_idx=point_d_idx,
            smoothing_window=transition_smoothing_window,
            plateau_fraction=transition_onset_fraction,
            polyorder=transition_polyorder,
        )
        if transitions is None:
            derivative_failures += 1
            continue

        point_b_idx, point_c_idx = transitions
        if not (
            point_a_idx
            < point_b_idx
            <= max_anchor_idx
            <= point_c_idx
            < point_d_idx
        ):
            derivative_failures += 1
            continue

        thickness_a = float(thickness_values[point_a_idx])
        thickness_b = float(thickness_values[point_b_idx])
        thickness_max = float(thickness_values[max_anchor_idx])
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
                "Max Anchor Index": max_anchor_idx,
                "Max Anchor Time": float(time_values[max_anchor_idx]),
                "Max Anchor Thickness": thickness_max,
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
