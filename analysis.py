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


def _flat_aware_local_extrema_candidates(
    local_thickness: np.ndarray,
    order: int,
    missing_type: str,
) -> np.ndarray:
    """Find sharp or flat extrema for local missing-point recovery only.

    The global detector intentionally remains strict. Recovery uses an inclusive
    comparator so a flat top/bottom can be considered, then collapses each
    contiguous candidate run to one representative point. A run is accepted only
    when the entire extremal level is strictly bounded by the opposite direction
    on both sides, preventing a flat shoulder on a monotonic slope from being
    mistaken for an extremum.
    """
    values = np.asarray(local_thickness, dtype=float)
    if len(values) < 3 or order < 1 or missing_type not in {"min", "max"}:
        return np.asarray([], dtype=int)

    comparator = np.less_equal if missing_type == "min" else np.greater_equal
    inclusive_candidates = signal.argrelextrema(
        values,
        comparator=comparator,
        order=int(order),
    )[0].astype(int)
    if len(inclusive_candidates) == 0:
        return np.asarray([], dtype=int)

    groups: list[tuple[int, int]] = []
    group_start = int(inclusive_candidates[0])
    group_end = group_start
    for candidate in inclusive_candidates[1:]:
        candidate = int(candidate)
        if candidate == group_end + 1:
            group_end = candidate
        else:
            groups.append((group_start, group_end))
            group_start = candidate
            group_end = candidate
    groups.append((group_start, group_end))

    representatives: list[int] = []
    for start_idx, end_idx in groups:
        if start_idx <= 0 or end_idx >= len(values) - 1:
            continue

        left_values = values[max(0, start_idx - order) : start_idx]
        right_values = values[
            end_idx + 1 : min(len(values), end_idx + 1 + order)
        ]
        if len(left_values) == 0 or len(right_values) == 0:
            continue

        group_values = values[start_idx : end_idx + 1]
        if missing_type == "min":
            extreme_value = float(np.min(group_values))
            is_true_extremum = bool(
                np.all(left_values > extreme_value)
                and np.all(right_values > extreme_value)
            )
        else:
            extreme_value = float(np.max(group_values))
            is_true_extremum = bool(
                np.all(left_values < extreme_value)
                and np.all(right_values < extreme_value)
            )

        if not is_true_extremum:
            continue

        tied_positions = np.flatnonzero(group_values == extreme_value)
        if len(tied_positions) == 0:
            continue
        representative_offset = int(tied_positions[(len(tied_positions) - 1) // 2])
        representatives.append(start_idx + representative_offset)

    return np.asarray(representatives, dtype=int)


def recover_missing_extremum(
    thickness_values: np.ndarray,
    min_indices: np.ndarray,
    max_indices: np.ndarray,
    issue: dict,
    recovery_order: int,
) -> RecoveryResult:
    """Recover one missing extremum, including a flat local top or bottom."""
    updated_mins = np.asarray(min_indices, dtype=int).copy()
    updated_maxs = np.asarray(max_indices, dtype=int).copy()
    recovered_mins: list[int] = []
    recovered_maxs: list[int] = []

    missing_type = str(issue["Missing Type"])
    region_start = int(issue["Start Index"])
    region_end = int(issue["End Index"])

    padding = max(recovery_order * 2, 5)
    padded_start = max(0, region_start - padding)
    padded_end = min(len(thickness_values) - 1, region_end + padding)
    local_thickness = np.asarray(
        thickness_values[padded_start : padded_end + 1], dtype=float
    )

    local_max_order = max(1, (len(local_thickness) - 1) // 2)
    effective_order = min(int(recovery_order), local_max_order)
    local_candidates = _flat_aware_local_extrema_candidates(
        local_thickness=local_thickness,
        order=effective_order,
        missing_type=missing_type,
    )
    candidates = local_candidates + padded_start
    candidates = candidates[(candidates > region_start) & (candidates < region_end)]

    if len(candidates) > 0 and missing_type == "min":
        recovered_idx = int(candidates[np.argmin(thickness_values[candidates])])
        updated_mins = np.unique(np.append(updated_mins, recovered_idx)).astype(int)
        recovered_mins.append(recovered_idx)
    elif len(candidates) > 0 and missing_type == "max":
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


def _find_fall_onset_interval(
    slopes: np.ndarray,
    plateau_limit: float,
    fall_reference: float,
    start_idx: int = 0,
) -> int | None:
    """Return the first sustained or clearly instantaneous active-fall interval.

    A normal fall needs two consecutive intervals below the plateau threshold.
    A single interval can also define the onset when its magnitude reaches the
    representative active-fall rate. To avoid accepting an obvious spike/rebound,
    that single-drop path is rejected when the next interval rebounds upward with
    comparable magnitude.
    """
    for interval_idx in range(int(start_idx), len(slopes)):
        current_slope = float(slopes[interval_idx])

        sustained_fall = (
            interval_idx + 1 < len(slopes)
            and current_slope < -plateau_limit
            and slopes[interval_idx + 1] < -plateau_limit
        )

        exceptional_single_drop = current_slope <= -fall_reference
        strong_immediate_rebound = (
            interval_idx + 1 < len(slopes)
            and slopes[interval_idx + 1] >= fall_reference
        )

        if sustained_fall or (
            exceptional_single_drop and not strong_immediate_rebound
        ):
            return interval_idx

    return None


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
    """Locate B and C around the purge/plateau near the maximum.

    B is the left edge of the low-slope purge region after the active rise.

    C is asymmetric on purpose. Starting at the maximum, the detector accepts
    either (1) the first two-interval sustained negative fall or (2) one clearly
    instantaneous drop whose magnitude reaches the representative active-fall
    rate and is not immediately reversed by a comparably strong rebound.

    This preserves a true one-sample reaction while ignoring ordinary isolated
    downward excursions during purge. If the trace changes directly from rise to
    a real fall, B and C may both equal the maximum.
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

    left_interval_idx = max_local_idx - 1
    while (
        left_interval_idx >= 0
        and abs(interval_slopes[left_interval_idx]) <= rise_plateau_limit
    ):
        left_interval_idx -= 1
    point_b_local_idx = left_interval_idx + 1

    point_c_local_idx = _find_fall_onset_interval(
        slopes=interval_slopes,
        plateau_limit=fall_plateau_limit,
        fall_reference=fall_reference,
        start_idx=max_local_idx,
    )
    if point_c_local_idx is None:
        return None

    if point_b_local_idx <= 0:
        return None
    if not np.any(left_slopes[:point_b_local_idx] > rise_plateau_limit):
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

    interval_idx = _find_fall_onset_interval(
        slopes=slopes,
        plateau_limit=plateau_limit,
        fall_reference=fall_reference,
        start_idx=0,
    )
    if interval_idx is None:
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
    """Calculate max-anchored cycles using purge entry B and fall-onset C.

    A and D are successive minima. The detected maximum anchors the purge region:
      - B = left edge of the low-slope plateau after the active rise
      - C = first sustained active fall, or one clearly instantaneous strong drop

    The maximum may equal B, C, or both. A cycle is rejected when a meaningful
    rise or fall cannot be resolved.
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
