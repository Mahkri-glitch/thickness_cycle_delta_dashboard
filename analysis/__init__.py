"""Localized transition-analysis overlay.

This package intentionally shadows the legacy top-level ``analysis.py`` module.
The legacy module is loaded under a private name and re-exported so all existing
API behavior remains available. Only the A/B/M/C/D transition detector is
replaced: preliminary B/C are found from raw interval slopes, then Savitzky-Golay
smoothing is applied only inside small neighborhoods around those preliminary
transition points.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


_BASE_PATH = Path(__file__).resolve().parent.parent / "analysis.py"
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_thickness_cycle_analysis_base",
    _BASE_PATH,
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load base analysis module from {_BASE_PATH}")

_base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = _base
_BASE_SPEC.loader.exec_module(_base)

# Re-export the established API first. The localized detector below then
# deliberately replaces only the transition function used by calculate_cycles.
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


def _default_local_radius(smoothing_window: int) -> int:
    """Choose a compact transition neighborhood from the requested SG window.

    The neighborhood is deliberately a little wider than the smoothing kernel so
    the fit has context on both sides of the preliminary transition without
    smoothing the full A->M or M->D side of the cycle.
    """
    window = max(3, int(smoothing_window))
    return max(2, (window + 3) // 2)


def _localized_refine_point_b(
    segment_time: np.ndarray,
    segment_thickness: np.ndarray,
    preliminary_b_idx: int,
    max_local_idx: int,
    local_radius: int,
    smoothing_window: int,
    plateau_limit: float,
    active_limit: float,
    polyorder: int,
) -> int:
    """Refine B using smoothing only near the preliminary rise->purge boundary.

    Refinement is conservative and one-way: B can move earlier only when the
    localized trace contains a sustained active-rise run followed by a sustained
    purge-like run. Otherwise the preliminary B is preserved exactly, including
    a valid B=M case.
    """
    preliminary = int(preliminary_b_idx)
    radius = max(1, int(local_radius))
    start = max(0, preliminary - radius)
    end = min(int(max_local_idx), preliminary + radius)
    if end - start < 2:
        return preliminary

    local_time = np.asarray(segment_time[start : end + 1], dtype=float)
    local_thickness = np.asarray(segment_thickness[start : end + 1], dtype=float)
    slopes = _isolated_smoothed_slopes(
        local_time,
        local_thickness,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
    )
    if len(slopes) != len(local_time) - 1:
        return preliminary

    baseline_local_idx = preliminary - start
    active_mask = slopes >= float(active_limit)
    purge_mask = np.abs(slopes) <= float(plateau_limit)

    # Require stable active-rise evidence at/before the preliminary boundary.
    active_run = _last_sustained_true_run(
        active_mask,
        0,
        min(len(slopes), baseline_local_idx + 1),
    )
    if active_run is None:
        return preliminary

    active_end = int(active_run[1])
    purge_run = _first_sustained_true_run(
        purge_mask,
        active_end + 1,
        len(slopes),
    )
    if purge_run is None:
        return preliminary

    active_side_point = active_end + 1
    purge_side_point = int(purge_run[0])
    if active_side_point > purge_side_point:
        return preliminary

    # Same tie policy as the established detector: bias ambiguity toward the
    # M/purge region, which means the earlier candidate for B.
    midpoint_local = (active_side_point + purge_side_point + 1) // 2
    candidate = start + midpoint_local
    return min(preliminary, int(candidate))


def _localized_refine_point_c(
    segment_time: np.ndarray,
    segment_thickness: np.ndarray,
    preliminary_c_idx: int,
    max_local_idx: int,
    min2_local_idx: int,
    local_radius: int,
    smoothing_window: int,
    plateau_limit: float,
    active_limit: float,
    fall_reference: float,
    polyorder: int,
) -> int:
    """Refine C using smoothing only near the preliminary purge->fall boundary.

    C can move later only when the localized trace contains sustained purge-like
    behavior followed by a confirmed active fall. Raw local slopes remain the
    single-drop safeguard so a genuinely instantaneous fall can still give C=M.
    """
    preliminary = int(preliminary_c_idx)
    radius = max(1, int(local_radius))
    start = max(int(max_local_idx), preliminary - radius)
    end = min(int(min2_local_idx), preliminary + radius)
    if end - start < 2:
        return preliminary

    local_time = np.asarray(segment_time[start : end + 1], dtype=float)
    local_thickness = np.asarray(segment_thickness[start : end + 1], dtype=float)
    slopes = _isolated_smoothed_slopes(
        local_time,
        local_thickness,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
    )
    if len(slopes) != len(local_time) - 1:
        return preliminary

    raw_slopes = np.diff(local_thickness) / np.diff(local_time)
    baseline_local_idx = preliminary - start
    active_start = _find_confirmed_active_fall_start(
        slopes=slopes,
        start_idx=baseline_local_idx,
        active_limit=active_limit,
        fall_reference=fall_reference,
        instantaneous_slopes=raw_slopes,
    )
    if active_start is None or active_start <= baseline_local_idx:
        return preliminary

    purge_mask = np.abs(slopes) <= float(plateau_limit)
    purge_run = _last_sustained_true_run(
        purge_mask,
        0,
        active_start,
    )
    if purge_run is None:
        return preliminary

    purge_side_point = int(purge_run[1]) + 1
    active_side_point = int(active_start)
    if purge_side_point > active_side_point:
        return preliminary

    # Same tie policy as the established detector: bias ambiguity toward the
    # M/purge region, which means the later candidate for C.
    midpoint_local = (purge_side_point + active_side_point) // 2
    candidate = start + midpoint_local
    return max(preliminary, int(candidate))


def _detect_plateau_transition_indices(
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    min1_idx: int,
    max_idx: int,
    min2_idx: int,
    smoothing_window: int = 3,
    plateau_fraction: float = 0.35,
    polyorder: int = 2,
    local_radius: int | None = None,
) -> tuple[int, int] | None:
    """Locate B/C with raw preliminary detection plus localized smoothing.

    A, M, and D remain raw extrema. Preliminary B/C are computed from raw
    time-aware interval slopes using the established thresholds, persistence, and
    transition-band rules. Only after that are compact neighborhoods around B and
    C smoothed and used for conservative one-way refinement. No SG filter is
    applied over the full A->M or M->D side.

    ``local_radius`` is a half-width in samples. When omitted, it is derived from
    ``smoothing_window`` so the existing Streamlit smoothing slider remains the
    only required control.
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

    max_local_idx = int(max_idx - min1_idx)
    min2_local_idx = len(segment_time) - 1
    raw_interval_slopes = np.diff(segment_thickness) / np.diff(segment_time)
    left_slopes = raw_interval_slopes[:max_local_idx]
    right_slopes = raw_interval_slopes[max_local_idx:]

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
    rise_active_limit = (1.0 - float(plateau_fraction)) * rise_reference
    fall_active_limit = (1.0 - float(plateau_fraction)) * fall_reference

    # Established preliminary B, now evaluated on raw slopes so smoothing is not
    # applied outside the local transition neighborhood.
    left_interval_idx = max_local_idx - 1
    while (
        left_interval_idx >= 0
        and abs(raw_interval_slopes[left_interval_idx]) <= rise_plateau_limit
    ):
        left_interval_idx -= 1
    baseline_b_local_idx = left_interval_idx + 1

    # Established preliminary C, including the raw one-sample strong-drop path.
    baseline_c_local_idx = _find_fall_onset_interval(
        slopes=raw_interval_slopes,
        plateau_limit=fall_plateau_limit,
        fall_reference=fall_reference,
        start_idx=max_local_idx,
        instantaneous_slopes=raw_interval_slopes,
    )
    if baseline_c_local_idx is None:
        return None

    # Preserve the existing blip-resistant transition-band refinement as the
    # preliminary raw estimate before localized smoothing is applied.
    preliminary_b_local_idx = _refine_point_b_with_transition_band(
        slopes=raw_interval_slopes,
        max_local_idx=max_local_idx,
        plateau_limit=rise_plateau_limit,
        active_limit=rise_active_limit,
        baseline_b_idx=baseline_b_local_idx,
    )
    preliminary_c_local_idx = _refine_point_c_with_transition_band(
        slopes=raw_interval_slopes,
        max_local_idx=max_local_idx,
        plateau_limit=fall_plateau_limit,
        active_limit=fall_active_limit,
        fall_reference=fall_reference,
        baseline_c_idx=baseline_c_local_idx,
        instantaneous_slopes=raw_interval_slopes,
    )

    radius = (
        _default_local_radius(smoothing_window)
        if local_radius is None
        else max(1, int(local_radius))
    )
    point_b_local_idx = _localized_refine_point_b(
        segment_time=segment_time,
        segment_thickness=segment_thickness,
        preliminary_b_idx=preliminary_b_local_idx,
        max_local_idx=max_local_idx,
        local_radius=radius,
        smoothing_window=smoothing_window,
        plateau_limit=rise_plateau_limit,
        active_limit=rise_active_limit,
        polyorder=polyorder,
    )
    point_c_local_idx = _localized_refine_point_c(
        segment_time=segment_time,
        segment_thickness=segment_thickness,
        preliminary_c_idx=preliminary_c_local_idx,
        max_local_idx=max_local_idx,
        min2_local_idx=min2_local_idx,
        local_radius=radius,
        smoothing_window=smoothing_window,
        plateau_limit=fall_plateau_limit,
        active_limit=fall_active_limit,
        fall_reference=fall_reference,
        polyorder=polyorder,
    )

    if point_b_local_idx <= 0:
        return None
    if not np.any(left_slopes[:point_b_local_idx] > rise_plateau_limit):
        return None
    if not (
        0
        < point_b_local_idx
        <= max_local_idx
        <= point_c_local_idx
        < min2_local_idx
    ):
        return None

    return (
        min1_idx + int(point_b_local_idx),
        min1_idx + int(point_c_local_idx),
    )


# Patch the private base module because its existing calculate_cycles function
# resolves this global name at runtime. That preserves the public function
# signature and all established cycle/result bookkeeping while switching only the
# transition detector to the localized implementation above.
_base._detect_plateau_transition_indices = _detect_plateau_transition_indices
calculate_cycles = _base.calculate_cycles
