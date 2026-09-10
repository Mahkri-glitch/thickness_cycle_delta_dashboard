import numpy as np
import pandas as pd

import analysis as analysis_module
from analysis import (
    _apply_localized_equality_rerun,
    _equality_masks,
)


def _primary_result(b_idx: int, m_idx: int, c_idx: int):
    time = np.arange(7, dtype=float)
    thickness = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0], dtype=float)
    a_idx = 0
    d_idx = 6
    row = {
        "Cycle": 1,
        "Point A Index": a_idx,
        "Point A Time": time[a_idx],
        "Point A Thickness": thickness[a_idx],
        "Point B Index": b_idx,
        "Point B Time": time[b_idx],
        "Point B Thickness": thickness[b_idx],
        "Max Anchor Index": m_idx,
        "Max Anchor Time": time[m_idx],
        "Max Anchor Thickness": thickness[m_idx],
        "Point C Index": c_idx,
        "Point C Time": time[c_idx],
        "Point C Thickness": thickness[c_idx],
        "Point D Index": d_idx,
        "Point D Time": time[d_idx],
        "Point D Thickness": thickness[d_idx],
        "Delta 1": thickness[b_idx] - thickness[a_idx],
        "Delta 2": thickness[b_idx] - thickness[c_idx],
        "Delta 3": thickness[c_idx] - thickness[d_idx],
    }
    result = analysis_module._base.CycleAnalysisResult(
        cycle_df=pd.DataFrame([row]),
        point_b_indices=[b_idx],
        point_c_indices=[c_idx],
        recovered_point_b_indices=[],
        recovered_point_c_indices=[],
        rejected_sequences=0,
        derivative_failures=0,
    )
    return result, time, thickness


def test_equality_masks_flag_b_and_c_independently():
    result, _, _ = _primary_result(b_idx=3, m_idx=3, c_idx=4)
    b_equal, c_equal = _equality_masks(result.cycle_df)
    assert b_equal.tolist() == [True]
    assert c_equal.tolist() == [False]


def test_localized_window_reruns_established_detector_and_moves_only_equal_b(monkeypatch):
    primary, time, thickness = _primary_result(b_idx=3, m_idx=3, c_idx=4)
    seen_windows = []

    def fake_detector(**kwargs):
        seen_windows.append(kwargs["smoothing_window"])
        return (2, 5)

    monkeypatch.setattr(
        analysis_module._base,
        "_detect_plateau_transition_indices",
        fake_detector,
    )

    result, moved_b, moved_c = _apply_localized_equality_rerun(
        primary_result=primary,
        time_values=time,
        thickness_values=thickness,
        recovered_min_indices=None,
        recovered_max_indices=None,
        plateau_fraction=0.35,
        polyorder=2,
        local_window=7,
    )

    row = result.cycle_df.iloc[0]
    assert seen_windows == [7]
    assert int(row["Point B Index"]) == 2
    # C was already resolved in the primary pass, so the local rerun cannot alter it.
    assert int(row["Point C Index"]) == 4
    assert moved_b == 1
    assert moved_c == 0


def test_localized_window_reruns_established_detector_and_moves_only_equal_c(monkeypatch):
    primary, time, thickness = _primary_result(b_idx=2, m_idx=3, c_idx=3)

    def fake_detector(**kwargs):
        assert kwargs["smoothing_window"] == 5
        return (1, 4)

    monkeypatch.setattr(
        analysis_module._base,
        "_detect_plateau_transition_indices",
        fake_detector,
    )

    result, moved_b, moved_c = _apply_localized_equality_rerun(
        primary_result=primary,
        time_values=time,
        thickness_values=thickness,
        recovered_min_indices=None,
        recovered_max_indices=None,
        plateau_fraction=0.35,
        polyorder=2,
        local_window=5,
    )

    row = result.cycle_df.iloc[0]
    # B was already resolved in the primary pass and must remain frozen.
    assert int(row["Point B Index"]) == 2
    assert int(row["Point C Index"]) == 4
    assert moved_b == 0
    assert moved_c == 1


def test_non_equality_cycle_is_never_rerun(monkeypatch):
    primary, time, thickness = _primary_result(b_idx=2, m_idx=3, c_idx=4)

    def should_not_run(**kwargs):
        raise AssertionError("localized detector should not run for an ordinary cycle")

    monkeypatch.setattr(
        analysis_module._base,
        "_detect_plateau_transition_indices",
        should_not_run,
    )

    result, moved_b, moved_c = _apply_localized_equality_rerun(
        primary_result=primary,
        time_values=time,
        thickness_values=thickness,
        recovered_min_indices=None,
        recovered_max_indices=None,
        plateau_fraction=0.35,
        polyorder=2,
        local_window=9,
    )

    row = result.cycle_df.iloc[0]
    assert int(row["Point B Index"]) == 2
    assert int(row["Point C Index"]) == 4
    assert moved_b == 0
    assert moved_c == 0


def test_genuine_direct_transition_can_remain_b_equals_m_equals_c(monkeypatch):
    primary, time, thickness = _primary_result(b_idx=3, m_idx=3, c_idx=3)

    def fake_detector(**kwargs):
        return (3, 3)

    monkeypatch.setattr(
        analysis_module._base,
        "_detect_plateau_transition_indices",
        fake_detector,
    )

    result, moved_b, moved_c = _apply_localized_equality_rerun(
        primary_result=primary,
        time_values=time,
        thickness_values=thickness,
        recovered_min_indices=None,
        recovered_max_indices=None,
        plateau_fraction=0.35,
        polyorder=2,
        local_window=11,
    )

    row = result.cycle_df.iloc[0]
    assert int(row["Point B Index"]) == 3
    assert int(row["Point C Index"]) == 3
    assert moved_b == 0
    assert moved_c == 0
