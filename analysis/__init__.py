"""Compatibility layer for equality-case localized smoothing.

The established top-level ``analysis.py`` remains the primary analysis engine.
Its normal transition detector and the existing global smoothing-window behavior
are re-exported unchanged. This package adds one optional second pass only for
cycles where the primary result has B=M and/or C=M.

For an eligible cycle, the same established detector is rerun with a separate
localized smoothing window. Only the side(s) that were equal to M in the primary
result may be replaced. Ordinary cycles and already-resolved sides are untouched.
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

# Re-export the established implementation unchanged. In particular, the normal
# detector is NOT patched, so the original Streamlit smoothing slider retains its
# established A->M / M->D isolated-smoothing behavior.
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


def _equality_masks(cycle_df):
    """Return boolean masks for primary B=M and C=M results."""
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


def _streamlit_context_available() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # pragma: no cover
        return False
    try:
        return get_script_run_ctx() is not None
    except Exception:  # pragma: no cover
        return False


def _render_localized_equality_control(cycle_df) -> int:
    """Render the separate equality-case smoothing control.

    A value of 1 means off. Counts and cycle numbers come from the primary result,
    before any localized rerun is attempted.
    """
    if not _streamlit_context_available():
        return 1

    import streamlit as st

    b_equal, c_equal = _equality_masks(cycle_df)
    flagged = b_equal | c_equal
    flagged_count = int(np.count_nonzero(flagged))
    b_count = int(np.count_nonzero(b_equal))
    c_count = int(np.count_nonzero(c_equal))

    st.sidebar.header("Localized equality-case smoothing")
    st.sidebar.caption(
        f"Primary detector flags: {flagged_count} cycle(s); "
        f"B=M in {b_count}, C=M in {c_count}."
    )

    if flagged_count == 0:
        st.sidebar.caption("No B=M or C=M cases need a localized rerun.")
        return 1

    flagged_cycles = cycle_df.loc[flagged, "Cycle"].astype(int).tolist()
    with st.sidebar.expander("Flagged cycles", expanded=False):
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
                "1 = off. Values 3, 5, 7, ... rerun the established B/C detector "
                "only for cycles whose primary result has B=M and/or C=M. Only "
                "the equality side is allowed to change."
            ),
        )
    )


def _rebuild_result_lists(
    cycle_df,
    primary_result,
    recovered_min_indices: list[int] | None,
    recovered_max_indices: list[int] | None,
):
    recovered_min_set = set(recovered_min_indices or [])
    recovered_max_set = set(recovered_max_indices or [])

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


def _apply_localized_equality_rerun(
    primary_result,
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    recovered_min_indices: list[int] | None,
    recovered_max_indices: list[int] | None,
    plateau_fraction: float,
    polyorder: int,
    local_window: int,
):
    """Rerun the established detector only on primary equality cases.

    The local pass does not invent a new transition rule. It simply calls the same
    detector used by the primary analysis with ``smoothing_window=local_window``.
    If primary B=M, only B is eligible for replacement. If primary C=M, only C is
    eligible. A non-equality side is frozen at its primary value.

    Returns ``(result, moved_b_count, moved_c_count)``.
    """
    if int(local_window) <= 1 or primary_result.cycle_df.empty:
        return primary_result, 0, 0

    cycle_df = primary_result.cycle_df.copy()
    moved_b = 0
    moved_c = 0

    for row_idx in cycle_df.index:
        a_idx = int(cycle_df.at[row_idx, "Point A Index"])
        primary_b = int(cycle_df.at[row_idx, "Point B Index"])
        m_idx = int(cycle_df.at[row_idx, "Max Anchor Index"])
        primary_c = int(cycle_df.at[row_idx, "Point C Index"])
        d_idx = int(cycle_df.at[row_idx, "Point D Index"])

        b_equal = primary_b == m_idx
        c_equal = primary_c == m_idx
        if not (b_equal or c_equal):
            continue

        local_transitions = _base._detect_plateau_transition_indices(
            time_values=time_values,
            thickness_values=thickness_values,
            min1_idx=a_idx,
            max_idx=m_idx,
            min2_idx=d_idx,
            smoothing_window=int(local_window),
            plateau_fraction=float(plateau_fraction),
            polyorder=int(polyorder),
        )
        if local_transitions is None:
            continue

        local_b, local_c = map(int, local_transitions)
        new_b = local_b if b_equal else primary_b
        new_c = local_c if c_equal else primary_c

        if not (a_idx < new_b <= m_idx <= new_c < d_idx):
            continue

        if b_equal and new_b != primary_b:
            cycle_df.at[row_idx, "Point B Index"] = new_b
            cycle_df.at[row_idx, "Point B Time"] = float(time_values[new_b])
            cycle_df.at[row_idx, "Point B Thickness"] = float(thickness_values[new_b])
            moved_b += 1

        if c_equal and new_c != primary_c:
            cycle_df.at[row_idx, "Point C Index"] = new_c
            cycle_df.at[row_idx, "Point C Time"] = float(time_values[new_c])
            cycle_df.at[row_idx, "Point C Thickness"] = float(thickness_values[new_c])
            moved_c += 1

        # Re-read the possibly changed points before updating delta values.
        final_b = int(cycle_df.at[row_idx, "Point B Index"])
        final_c = int(cycle_df.at[row_idx, "Point C Index"])
        thickness_a = float(thickness_values[a_idx])
        thickness_b = float(thickness_values[final_b])
        thickness_c = float(thickness_values[final_c])
        thickness_d = float(thickness_values[d_idx])
        cycle_df.at[row_idx, "Delta 1"] = thickness_b - thickness_a
        cycle_df.at[row_idx, "Delta 2"] = thickness_b - thickness_c
        cycle_df.at[row_idx, "Delta 3"] = thickness_c - thickness_d

    result = _rebuild_result_lists(
        cycle_df=cycle_df,
        primary_result=primary_result,
        recovered_min_indices=recovered_min_indices,
        recovered_max_indices=recovered_max_indices,
    )
    return result, moved_b, moved_c


def _render_localized_effect(local_window: int, moved_b: int, moved_c: int) -> None:
    """Tell the user whether the selected local window actually changed anything."""
    if int(local_window) <= 1 or not _streamlit_context_available():
        return

    import streamlit as st

    total = int(moved_b) + int(moved_c)
    if total:
        st.sidebar.success(
            f"Localized rerun moved {total} equality boundary/boundaries "
            f"(B: {moved_b}, C: {moved_c})."
        )
    else:
        st.sidebar.info(
            "This localized window did not move any flagged B=M/C=M boundary. "
            "Try a different odd window; genuine direct transitions may remain at M."
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
    localized_equality_smoothing_window: int | None = None,
):
    """Run unchanged primary analysis, then an equality-only detector rerun."""
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

    if localized_equality_smoothing_window is None:
        local_window = _render_localized_equality_control(primary_result.cycle_df)
    else:
        local_window = int(localized_equality_smoothing_window)

    result, moved_b, moved_c = _apply_localized_equality_rerun(
        primary_result=primary_result,
        time_values=np.asarray(time_values, dtype=float),
        thickness_values=np.asarray(thickness_values, dtype=float),
        recovered_min_indices=recovered_min_indices,
        recovered_max_indices=recovered_max_indices,
        plateau_fraction=float(transition_onset_fraction),
        polyorder=int(transition_polyorder),
        local_window=int(local_window),
    )
    if localized_equality_smoothing_window is None:
        _render_localized_effect(local_window, moved_b, moved_c)
    return result
