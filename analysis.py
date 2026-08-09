"""Core analysis functions for the Thickness Cycle Delta Analyzer.

The functions in this module are intentionally independent of Streamlit so they
can be tested, reused, and reasoned about separately from the user interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.signal as signal

AnalysisType = Literal["2-step", "3-step"]

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
    """Updated extrema arrays plus any point newly recovered in one local gap."""

    min_indices: np.ndarray
    max_indices: np.ndarray
    recovered_min_indices: list[int]
    recovered_max_indices: list[int]


@dataclass
class CycleAnalysisResult:
    """Cycle-level output and diagnostic metadata."""

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
    """Find broken alternation that suggests one missing opposite extremum.

    MAX -> MAX suggests a missing minimum between the two maxima.
    MIN -> MIN suggests a missing maximum between the two minima.
    """
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
    """Recover one missing extremum inside a suspicious gap.

    Recovery is additive: existing global extrema are preserved. The local
    search only adds one candidate inside the selected same-type gap.
    """
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

    # Guard against an order that is much larger than the local search region.
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


def _detect_transition_index(
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    max_idx: int,
    min2_idx: int,
) -> int | None:
    """Locate Point C as the strongest change toward the final steep drop.

    The method follows the dashboard logic developed for 3-step cycles:
    1. take the B -> D segment,
    2. compute the first derivative,
    3. compute the change in slope,
    4. select the most negative interior slope change.
    """
    segment_time = time_values[max_idx : min2_idx + 1]
    segment_thickness = thickness_values[max_idx : min2_idx + 1]

    if len(segment_time) < 5:
        return None

    # np.gradient requires meaningful spacing. Duplicate time points make the
    # derivative ill-defined, so reject the transition rather than hide it.
    if np.any(np.diff(segment_time) == 0):
        return None

    slope = np.gradient(segment_thickness, segment_time)
    slope_change = np.diff(slope)
    interior_slope_change = slope_change[1:-1]

    if len(interior_slope_change) == 0 or not np.isfinite(interior_slope_change).any():
        return None

    transition_local_idx = int(np.nanargmin(interior_slope_change) + 1)
    return max_idx + transition_local_idx


def calculate_cycles(
    events: pd.DataFrame,
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    analysis_type: AnalysisType,
    recovered_min_indices: list[int] | None = None,
    recovered_max_indices: list[int] | None = None,
) -> CycleAnalysisResult:
    """Calculate only complete successive MIN -> MAX -> MIN cycles.

    2-step mode:
      A = first minimum
      B = maximum
      D = next minimum
      Delta 1 = B - A
      Delta 2 = B - D

    3-step mode:
      A = first minimum
      B = maximum
      C = transition point between B and D
      D = next minimum
      Delta 1 = B - A
      Delta 2 = B - C
      Delta 3 = C - D
    """
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

        cycle_data = {
            "Cycle": len(cycles) + 1,
            "Point A Index": point_a_idx,
            "Point A Time": float(point_a["Time"]),
            "Point A Thickness": thickness_a,
            "Point B Index": point_b_idx,
            "Point B Time": float(point_b["Time"]),
            "Point B Thickness": thickness_b,
            "Point C Index": np.nan,
            "Point C Time": np.nan,
            "Point C Thickness": np.nan,
            "Point D Index": point_d_idx,
            "Point D Time": float(point_d["Time"]),
            "Point D Thickness": thickness_d,
            "Delta 1": thickness_b - thickness_a,
        }

        if analysis_type == "2-step":
            cycle_data["Delta 2"] = thickness_b - thickness_d
            cycles.append(cycle_data)
            continue

        transition_idx = _detect_transition_index(
            time_values=time_values,
            thickness_values=thickness_values,
            max_idx=point_b_idx,
            min2_idx=point_d_idx,
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

        cycle_data.update(
            {
                "Point C Index": transition_idx,
                "Point C Time": transition_time,
                "Point C Thickness": transition_thickness,
                "Delta 2": thickness_b - transition_thickness,
                "Delta 3": transition_thickness - thickness_d,
            }
        )
        cycles.append(cycle_data)

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
