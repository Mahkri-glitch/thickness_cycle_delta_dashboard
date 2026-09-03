"""Core analysis functions for the Thickness Cycle Delta Analyzer.

The functions in this module are intentionally independent of Streamlit so they
can be tested, reused, and reasoned about separately from the user interface.
"""

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


def _effective_savgol_window(
    segment_length: int,
    requested_window: int,
    polyorder: int,
) -> int | None:
    """Return a valid odd Savitzky-Golay window for one B -> D segment."""
    if segment_length < 5 or polyorder < 2:
        return None

    max_window = segment_length if segment_length % 2 == 1 else segment_length - 1
    window = max(int(requested_window), polyorder + 2)
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
    smoothing_window: int = 11,
    polyorder: int = 2,
    min_width: float = 5.0,
) -> int | None:
    """Locate Point C from a broad, persistent negative-curvature feature.

    Point C is intended to mark the transition into the final ALE-related drop,
    not simply the sharpest individual step between samples. To reduce the
    sensitivity of the old raw second-derivative method to abrupt C -> D jumps:

    1. Smooth the B -> D thickness trace with a low-order Savitzky-Golay fit.
    2. Compute first and second derivatives using the measured time spacing.
    3. Search the negative second derivative for local peaks.
    4. Require a minimum peak width so one- or two-sample step artifacts are
       rejected even when their instantaneous curvature is very large.
    5. Choose the remaining candidate with the greatest prominence.

    ``polyorder`` is intentionally restricted to 2 or 3. Higher-order local
    polynomials can follow sharp point-to-point structure and reintroduce the
    sensitivity this filter is designed to suppress.
    """
    if polyorder not in (2, 3):
        return None
    if min_width < 1:
        return None

    segment_time = np.asarray(time_values[max_idx : min2_idx + 1], dtype=float)
    segment_thickness = np.asarray(
        thickness_values[max_idx : min2_idx + 1], dtype=float
    )

    if len(segment_time) < 7:
        return None
    if not np.isfinite(segment_time).all() or not np.isfinite(segment_thickness).all():
        return None

    # Derivatives require strictly increasing time. Duplicate or reversed time
    # points are rejected rather than silently generating unstable gradients.
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
    second_derivative = np.gradient(slope, segment_time)

    # Negative second derivative means the downward slope is becoming steeper.
    # Search its magnitude as a positive peak signal.
    transition_strength = -second_derivative

    # Savitzky-Golay interpolation is least trustworthy at the window edges, and
    # B/D themselves are extrema rather than Point C candidates.
    edge_buffer = max(2, effective_window // 2)
    if len(transition_strength) <= 2 * edge_buffer + 1:
        edge_buffer = 1

    interior_strength = transition_strength[edge_buffer:-edge_buffer]
    if len(interior_strength) < 3 or not np.isfinite(interior_strength).any():
        return None

    # Width is measured in samples at half prominence. The minimum-width rule is
    # the main safeguard against a very steep but nearly instantaneous C -> D
    # step overwhelming a broader process transition.
    peaks, properties = signal.find_peaks(
        interior_strength,
        prominence=0,
        width=float(min_width),
    )
    if len(peaks) == 0:
        return None

    best_peak = int(peaks[np.argmax(properties["prominences"])])
    transition_local_idx = edge_buffer + best_peak
    return max_idx + transition_local_idx


def calculate_cycles(
    events: pd.DataFrame,
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    recovered_min_indices: list[int] | None = None,
    recovered_max_indices: list[int] | None = None,
    transition_smoothing_window: int = 11,
    transition_polyorder: int = 2,
    transition_min_width: float = 5.0,
) -> CycleAnalysisResult:
    """Calculate complete ALD/ALE-process MIN -> MAX -> MIN cycles.

    A = first minimum
    B = maximum
    C = broad transition point between B and D
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

        transition_idx = _detect_transition_index(
            time_values=time_values,
            thickness_values=thickness_values,
            max_idx=point_b_idx,
            min2_idx=point_d_idx,
            smoothing_window=transition_smoothing_window,
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
