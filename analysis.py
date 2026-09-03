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
    transition_indices: list[int]
    recovered_transition_indices: list[int]
    rejected_sequences: int
    derivative_failures: int


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
    """Return a valid odd Savitzky-Golay window for one B -> D segment."""
    if segment_length < 5 or polyorder < 1:
        return None

    max_window = segment_length if segment_length % 2 == 1 else segment_length - 1
    window = max(int(requested_window), polyorder + 1)
    if window % 2 == 0:
        window += 1
    window = min(window, max_window)
    if window <= polyorder:
        return None
    return window


def _detect_transition_index(
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    max_idx: int,
    min2_idx: int,
    smoothing_window: int = 5,
    onset_fraction: float = 0.35,
    persistence: int = 2,
    polyorder: int = 2,
    min_width: float | None = None,
) -> int | None:
    """Locate Point C as the onset of the rapid etch-rate regime.

    Point C is defined physically as the start of the sustained rapid thickness
    decrease that leads to Point D. The detector therefore does *not* use the
    largest second derivative or the center of a large inflection.

    Procedure for each B -> D segment:
      1. Lightly smooth thickness with a low-order Savitzky-Golay filter.
      2. Compute dh/dt and find the most negative slope (strongest etch rate).
      3. Estimate the pre-etch/purge slope from the early part of B -> etch.
      4. Set an onset threshold between the purge slope and strongest etch slope.
      5. Walk backward from the strongest etch point to the beginning of the
         connected threshold-crossing region. That beginning is Point C.
      6. Require the etch-rate region to persist for at least ``persistence``
         samples so isolated derivative noise does not create Point C.

    ``onset_fraction`` controls how far from purge toward the maximum etch rate
    the threshold lies. Lower values detect an earlier onset; higher values put
    C closer to the steep drop. ``min_width`` is retained only for compatibility
    with an older dashboard version and is intentionally ignored.
    """
    del min_width

    if not (0.01 <= float(onset_fraction) <= 0.95):
        return None
    persistence = int(persistence)
    if persistence < 1:
        return None
    if polyorder not in (2, 3):
        return None

    segment_time = np.asarray(time_values[max_idx : min2_idx + 1], dtype=float)
    segment_thickness = np.asarray(
        thickness_values[max_idx : min2_idx + 1], dtype=float
    )

    if len(segment_time) < 5:
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

    if len(slope) < 3 or not np.isfinite(slope[1:-1]).any():
        return None
    etch_local_idx = int(np.nanargmin(slope[1:-1]) + 1)
    strongest_etch_slope = float(slope[etch_local_idx])

    pre_etch_slopes = slope[1:etch_local_idx]
    if len(pre_etch_slopes) == 0 or not np.isfinite(pre_etch_slopes).any():
        return None

    baseline_count = max(1, min(5, int(np.ceil(len(pre_etch_slopes) * 0.5))))
    purge_slope = float(np.nanmedian(pre_etch_slopes[:baseline_count]))

    if not np.isfinite(purge_slope) or strongest_etch_slope >= purge_slope:
        return None

    onset_threshold = purge_slope + float(onset_fraction) * (
        strongest_etch_slope - purge_slope
    )

    onset_local_idx = etch_local_idx
    while onset_local_idx > 1 and slope[onset_local_idx - 1] <= onset_threshold:
        onset_local_idx -= 1

    run_length = etch_local_idx - onset_local_idx + 1
    if run_length < persistence:
        return None

    return max_idx + onset_local_idx


def calculate_cycles(
    events: pd.DataFrame,
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    recovered_min_indices: list[int] | None = None,
    recovered_max_indices: list[int] | None = None,
    transition_smoothing_window: int = 5,
    transition_onset_fraction: float = 0.35,
    transition_persistence: int = 2,
    transition_polyorder: int = 2,
    transition_min_width: float | None = None,
) -> CycleAnalysisResult:
    """Calculate complete ALD/ALE-process MIN -> MAX -> C -> MIN cycles."""
    recovered_min_set = set(recovered_min_indices or [])
    recovered_max_set = set(recovered_max_indices or [])

    cycles: list[dict] = []
    transition_indices: list[int] = []
    recovered_transition_indices: list[int] = []
    rejected_sequences = 0
    derivative_failures = 0

    records = events.to_dict("records")

    for i in range(len(records) - 2):
        point_a = records[i]
        point_b = records[i + 1]
        point_d = records[i + 2]

        is_complete_cycle = (
            point_a["Type"] == "min"
            and point_b["Type"] == "max"
            and point_d["Type"] == "min"
        )
        if not is_complete_cycle:
            if point_a["Type"] == "min":
                rejected_sequences += 1
            continue

        point_a_idx = int(point_a["Index"])
        point_b_idx = int(point_b["Index"])
        point_d_idx = int(point_d["Index"])
        thickness_a = float(point_a["Thickness"])
        thickness_b = float(point_b["Thickness"])
        thickness_d = float(point_d["Thickness"])

        transition_idx = _detect_transition_index(
            time_values=time_values,
            thickness_values=thickness_values,
            max_idx=point_b_idx,
            min2_idx=point_d_idx,
            smoothing_window=transition_smoothing_window,
            onset_fraction=transition_onset_fraction,
            persistence=transition_persistence,
            polyorder=transition_polyorder,
            min_width=transition_min_width,
        )
        if transition_idx is None:
            derivative_failures += 1
            continue

        transition_time = float(time_values[transition_idx])
        transition_thickness = float(thickness_values[transition_idx])

        cycle_uses_recovered_extremum = (
            point_a_idx in recovered_min_set
            or point_d_idx in recovered_min_set
            or point_b_idx in recovered_max_set
        )
        if cycle_uses_recovered_extremum:
            recovered_transition_indices.append(transition_idx)
        else:
            transition_indices.append(transition_idx)

        cycles.append(
            {
                "Cycle": len(cycles) + 1,
                "Point A Index": point_a_idx,
                "Point A Time": float(point_a["Time"]),
                "Point A Thickness": thickness_a,
                "Point B Index": point_b_idx,
                "Point B Time": float(point_b["Time"]),
                "Point B Thickness": thickness_b,
                "Point C Index": transition_idx,
                "Point C Time": transition_time,
                "Point C Thickness": transition_thickness,
                "Point D Index": point_d_idx,
                "Point D Time": float(point_d["Time"]),
                "Point D Thickness": thickness_d,
                "Delta 1": thickness_b - thickness_a,
                "Delta 2": thickness_b - transition_thickness,
                "Delta 3": transition_thickness - thickness_d,
            }
        )

    cycle_df = format_cycle_results(pd.DataFrame(cycles))
    return CycleAnalysisResult(
        cycle_df=cycle_df,
        transition_indices=sorted(set(transition_indices)),
        recovered_transition_indices=sorted(set(recovered_transition_indices)),
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
