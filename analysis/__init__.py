"""Compatibility layer for optional localized smoothing of equality cases.

The established top-level ``analysis.py`` remains the primary analysis engine.
In particular, its normal transition smoothing and B/C detector are re-exported
unchanged. This package adds only an optional second pass for cycles where the
primary detector returns B=M and/or C=M.

The separate localized smoothing control is rendered during ``calculate_cycles``
when Streamlit is running. A value of 1 means off. No non-equality cycle is
modified by the localized pass.
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

# Re-export the established implementation unchanged. The original detector is
# deliberately NOT patched: the existing Streamlit smoothing slider therefore
# keeps exactly its prior whole-side isolated-smoothing behavior.
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


def _equality_masks(cycle_df):
    """Return masks for primary B=M and C=M results."""
    if cycle_df is None or cycle_df.empty:
        empty = np.asarray([], dtype=bool)
        return empty, empty

    b_equal = (
        cycle_df["Point B Index"].to_numpy(dtype=int)
        == cycle_df["Max Anchor Index"].to_numpy(dtype=int)
    )
    c_equal = (
        cycle_df["Point C Index"].to_numpy(dtype=int)
        == cycle_df["Max Anchor Index"].to_numpy(dtype=int)
    )
    return b_equal, c_equal


def _render_localized_equality_control(cycle_df) -> int:
    """Render the separate Streamlit control and return its odd SG window.

    The primary analysis has already run before this function is called, so the
    counts shown here are always the equality cases produced by the normal
    detector, not by the localized second pass.
    """
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # pragma: no cover - Streamlit is optional for pure analysis
        return 1

    try:
        if get_script_run_ctx() is None:
            return 1
    except Exception:  # pragma: no cover
        return 1

    b_equal, c_equal = _equality_masks(cycle_df)
    flagged = b_equal | c_equal
    flagged_count = int(np.count_nonzero(flagged))
    b_count = int(np.count_nonzero(b_equal))
    c_count = int(np.count_nonzero(c_equal))

    st.sidebar.header("Localized equality-case smoothing")
    st.sidebar.caption(
        f"Primary detector flags: {flagged_count} cycle(s) total; "
        f"B=M in {b_count}, C=M in {c_count}."
    )

    if flagged_count == 0:
        st.sidebar.caption("No B=M or C=M cases need a localized second pass.")
        return 1

    flagged_cycles = (
        cycle_df.loc[flagged, "Cycle"].astype(int).tolist()
        if "Cycle" in cycle_df.columns
        else []
    )
    with st.sidebar.expander("Flagged cycles", expanded=False):
        if flagged_cycles:
            st.write(", ".join(str(value) for value in flagged_cycles))

    return int(
        st.sidebar.slider(
            "Localized smoothing window for B=M / C=M cases",
            min_value=1,
            max_value=21,
            value=1,
            step=2,
            key="localized_equality_smoothing_window",
            help=(
                "1 = off. Values 3, 5, 7, ... apply Savitzky-Golay smoothing "
                "only in a compact neighborhood next to M, and only on sides "
                "where the primary detector returned B=M or C=M. The normal "
                "Smoothing window above remains the primary detector control."
            ),
        )
    )


def _primary_references(
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    a_idx: int,
    m_idx: int,
    d_idx: int,
    primary_smoothing_window: int,
    polyorder: int,
) -> tuple[float, float] | None:
    """Get rise/fall references using the normal primary smoothing semantics."""
    left_slopes = _base._isolated_smoothed_slopes(
        np.asarray(time_values[a_idx : m_idx + 1], dtype=float),
        np.asarray(thickness_values[a_idx : m_idx + 1], dtype=float),
        smoothing_window=int(primary_smoothing_window),
        polyorder=int(polyorder),
    )
    right_slopes = _base._isolated_smoothed_slopes(
        np.asarray(time_values[m_idx : d_idx + 1], dtype=float),
        np.asarray(thickness_values[m_idx : d_idx + 1], dtype=float),
        smoothing_window=int(primary_smoothing_window),
        polyorder=int(polyorder),
    )

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
    return rise_reference, fall_reference


def _localized_refine_equal_b(
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    a_idx: int,
    m_idx: int,
    local_window: int,
    plateau_limit: float,
    active_limit: float,
    polyorder: int,
) -> int:
    """Try to separate B from M using only a compact pre-M neighborhood."""
    span = max(5, int(local_window) + 2)
    start = max(int(a_idx), int(m_idx) - span)
    if m_idx - start < 2:
        return int(m_idx)

    local_time = np.asarray(time_values[start : m_idx + 1], dtype=float)
    local_thickness = np.asarray(thickness_values[start : m_idx + 1], dtype=float)
    local_slopes = _base._isolated_smoothed_slopes(
        local_time,
        local_thickness,
        smoothing_window=int(local_window),
        polyorder=int(polyorder),
    )
    max_local_idx = len(local_time) - 1
    if len(local_slopes) != max_local_idx:
        return int(m_idx)

    candidate_local = _base._refine_point_b_with_transition_band(
        slopes=local_slopes,
        max_local_idx=max_local_idx,
        plateau_limit=float(plateau_limit),
        active_limit=float(active_limit),
        baseline_b_idx=max_local_idx,
    )
    candidate = start + int(candidate_local)
    if not (a_idx < candidate <= m_idx):
        return int(m_idx)
    return int(candidate)


def _localized_refine_equal_c(
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    m_idx: int,
    d_idx: int,
    local_window: int,
    plateau_limit: float,
    active_limit: float,
    fall_reference: float,
    polyorder: int,
) -> int:
    """Try to separate C from M using only a compact post-M neighborhood."""
    span = max(5, int(local_window) + 2)
    end = min(int(d_idx), int(m_idx) + span)
    if end - m_idx < 2:
        return int(m_idx)

    local_time = np.asarray(time_values[m_idx : end + 1], dtype=float)
    local_thickness = np.asarray(thickness_values[m_idx : end + 1], dtype=float)
    local_slopes = _base._isolated_smoothed_slopes(
        local_time,
        local_thickness,
        smoothing_window=int(local_window),
        polyorder=int(polyorder),
    )
    if len(local_slopes) != len(local_time) - 1:
        return int(m_idx)

    raw_local_slopes = np.diff(local_thickness) / np.diff(local_time)
    candidate_local = _base._refine_point_c_with_transition_band(
        slopes=local_slopes,
        max_local_idx=0,
        plateau_limit=float(plateau_limit),
        active_limit=float(active_limit),
        fall_reference=float(fall_reference),
        baseline_c_idx=0,
        instantaneous_slopes=raw_local_slopes,
    )
    candidate = int(m_idx) + int(candidate_local)
    if not (m_idx <= candidate < d_idx):
        return int(m_idx)
    return int(candidate)


def _apply_localized_equality_smoothing(
    primary_result,
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    recovered_min_indices: list[int] | None,
    recovered_max_indices: list[int] | None,
    primary_smoothing_window: int,
    plateau_fraction: float,
    polyorder: int,
    local_window: int,
):
    """Apply the optional second pass only to primary equality cases."""
    if int(local_window) <= 1 or primary_result.cycle_df.empty:
        return primary_result

    cycle_df = primary_result.cycle_df.copy()
    recovered_min_set = set(recovered_min_indices or [])
    recovered_max_set = set(recovered_max_indices or [])

    for row_idx in cycle_df.index:
        a_idx = int(cycle_df.at[row_idx, "Point A Index"])
        b_idx = int(cycle_df.at[row_idx, "Point B Index"])
        m_idx = int(cycle_df.at[row_idx, "Max Anchor Index"])
        c_idx = int(cycle_df.at[row_idx, "Point C Index"])
        d_idx = int(cycle_df.at[row_idx, "Point D Index"])

        b_equal = b_idx == m_idx
        c_equal = c_idx == m_idx
        if not (b_equal or c_equal):
            continue

        references = _primary_references(
            time_values=time_values,
            thickness_values=thickness_values,
            a_idx=a_idx,
            m_idx=m_idx,
            d_idx=d_idx,
            primary_smoothing_window=int(primary_smoothing_window),
            polyorder=int(polyorder),
        )
        if references is None:
            continue
        rise_reference, fall_reference = references

        rise_plateau_limit = float(plateau_fraction) * rise_reference
        fall_plateau_limit = float(plateau_fraction) * fall_reference
        rise_active_limit = (1.0 - float(plateau_fraction)) * rise_reference
        fall_active_limit = (1.0 - float(plateau_fraction)) * fall_reference

        new_b = b_idx
        new_c = c_idx
        if b_equal:
            new_b = _localized_refine_equal_b(
                time_values=time_values,
                thickness_values=thickness_values,
                a_idx=a_idx,
                m_idx=m_idx,
                local_window=int(local_window),
                plateau_limit=rise_plateau_limit,
                active_limit=rise_active_limit,
                polyorder=int(polyorder),
            )
        if c_equal:
            new_c = _localized_refine_equal_c(
                time_values=time_values,
                thickness_values=thickness_values,
                m_idx=m_idx,
                d_idx=d_idx,
                local_window=int(local_window),
                plateau_limit=fall_plateau_limit,
                active_limit=fall_active_limit,
                fall_reference=fall_reference,
                polyorder=int(polyorder),
            )

        if not (a_idx < new_b <= m_idx <= new_c < d_idx):
            continue

        if new_b != b_idx:
            cycle_df.at[row_idx, "Point B Index"] = int(new_b)
            cycle_df.at[row_idx, "Point B Time"] = float(time_values[new_b])
            cycle_df.at[row_idx, "Point B Thickness"] = float(thickness_values[new_b])
        if new_c != c_idx:
            cycle_df.at[row_idx, "Point C Index"] = int(new_c)
            cycle_df.at[row_idx, "Point C Time"] = float(time_values[new_c])
            cycle_df.at[row_idx, "Point C Thickness"] = float(thickness_values[new_c])

        thickness_a = float(thickness_values[a_idx])
        thickness_b = float(thickness_values[new_b])
        thickness_c = float(thickness_values[new_c])
        thickness_d = float(thickness_values[d_idx])
        cycle_df.at[row_idx, "Delta 1"] = thickness_b - thickness_a
        cycle_df.at[row_idx, "Delta 2"] = thickness_b - thickness_c
        cycle_df.at[row_idx, "Delta 3"] = thickness_c - thickness_d

    point_b_indices: list[int] = []
    point_c_indices: list[int] = []
    recovered_point_b_indices: list[int] = []
    recovered_point_c_indices: list[int] = []

    for _, row in cycle_df.iterrows():
        a_idx = int(row["Point A Index"])
        b_idx = int(row["Point B Index"])
        m_idx = int(row["Max Anchor Index"])
        c_idx = int(row["Point C Index"])
        d_idx = int(row["Point D Index"])
        uses_recovered = (
            a_idx in recovered_min_set
            or d_idx in recovered_min_set
            or m_idx in recovered_max_set
        )
        if uses_recovered:
            recovered_point_b_indices.append(b_idx)
            recovered_point_c_indices.append(c_idx)
        else:
            point_b_indices.append(b_idx)
            point_c_indices.append(c_idx)

    return _base.CycleAnalysisResult(
        cycle_df=_base.format_cycle_results(cycle_df),
        point_b_indices=sorted(set(point_b_indices)),
        point_c_indices=sorted(set(point_c_indices)),
        recovered_point_b_indices=sorted(set(recovered_point_b_indices)),
        recovered_point_c_indices=sorted(set(recovered_point_c_indices)),
        rejected_sequences=primary_result.rejected_sequences,
        derivative_failures=primary_result.derivative_failures,
    )


def calculate_cycles(
    events,
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    recovered_min_indices: list[int] | None = None,
    recovered_max_indices: list[int] | None = None,
    transition_smoothing_window: int = 3,
    transition_onset_fraction: float = 0.35,
    transition_persistence: int = 2,
    transition_polyorder: int = 2,
    transition_min_width: float | None = None,
):
    """Run the unchanged primary analysis, then optional equality-only smoothing."""
    primary_result = _base.calculate_cycles(
        events=events,
        time_values=time_values,
        thickness_values=thickness_values,
        recovered_min_indices=recovered_min_indices,
        recovered_max_indices=recovered_max_indices,
        transition_smoothing_window=int(transition_smoothing_window),
        transition_onset_fraction=float(transition_onset_fraction),
        transition_persistence=int(transition_persistence),
        transition_polyorder=int(transition_polyorder),
        transition_min_width=transition_min_width,
    )

    local_window = _render_localized_equality_control(primary_result.cycle_df)
    return _apply_localized_equality_smoothing(
        primary_result=primary_result,
        time_values=np.asarray(time_values, dtype=float),
        thickness_values=np.asarray(thickness_values, dtype=float),
        recovered_min_indices=recovered_min_indices,
        recovered_max_indices=recovered_max_indices,
        primary_smoothing_window=int(transition_smoothing_window),
        plateau_fraction=float(transition_onset_fraction),
        polyorder=int(transition_polyorder),
        local_window=int(local_window),
    )
