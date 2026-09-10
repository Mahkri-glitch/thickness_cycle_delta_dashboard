import numpy as np
import pandas as pd

import analysis
from analysis import (
    CycleAnalysisResult,
    OUTPUT_COLUMNS,
    _apply_localized_equality_smoothing,
    _detect_plateau_transition_indices,
    _equality_masks,
)


def _result_from_row(row: dict) -> CycleAnalysisResult:
    cycle_df = pd.DataFrame([row])
    for column in OUTPUT_COLUMNS:
        if column not in cycle_df.columns:
            cycle_df[column] = np.nan
    cycle_df = cycle_df[OUTPUT_COLUMNS]
    return CycleAnalysisResult(
        cycle_df=cycle_df,
        point_b_indices=[int(row["Point B Index"])],
        point_c_indices=[int(row["Point C Index"])],
        recovered_point_b_indices=[],
        recovered_point_c_indices=[],
        rejected_sequences=0,
        derivative_failures=0,
    )


def _row(time, thickness, a, b, m, c, d):
    return {
        "Cycle": 1,
        "Point A Index": a,
        "Point A Time": float(time[a]),
        "Point A Thickness": float(thickness[a]),
        "Point B Index": b,
        "Point B Time": float(time[b]),
        "Point B Thickness": float(thickness[b]),
        "Max Anchor Index": m,
        "Max Anchor Time": float(time[m]),
        "Max Anchor Thickness": float(thickness[m]),
        "Point C Index": c,
        "Point C Time": float(time[c]),
        "Point C Thickness": float(thickness[c]),
        "Point D Index": d,
        "Point D Time": float(time[d]),
        "Point D Thickness": float(thickness[d]),
        "Delta 1": float(thickness[b] - thickness[a]),
        "Delta 2": float(thickness[b] - thickness[c]),
        "Delta 3": float(thickness[c] - thickness[d]),
    }


def test_primary_detector_is_the_established_detector_not_a_localized_replacement():
    assert _detect_plateau_transition_indices is analysis._base._detect_plateau_transition_indices


def test_equality_masks_flag_only_b_equals_m_and_c_equals_m():
    df = pd.DataFrame(
        {
            "Point B Index": [3, 4, 4],
            "Max Anchor Index": [4, 4, 4],
            "Point C Index": [5, 5, 4],
        }
    )
    b_equal, c_equal = _equality_masks(df)
    assert b_equal.tolist() == [False, True, True]
    assert c_equal.tolist() == [False, False, True]


def test_localized_second_pass_does_not_touch_non_equality_cycles():
    time = np.arange(10, dtype=float)
    thickness = np.array(
        [0.0, 1.0, 2.0, 3.0, 3.05, 3.02, 3.00, 2.0, 1.0, 0.0]
    )
    primary = _result_from_row(_row(time, thickness, 0, 3, 4, 6, 9))

    refined = _apply_localized_equality_smoothing(
        primary_result=primary,
        time_values=time,
        thickness_values=thickness,
        recovered_min_indices=None,
        recovered_max_indices=None,
        primary_smoothing_window=3,
        plateau_fraction=0.35,
        polyorder=2,
        local_window=7,
    )

    assert refined.cycle_df.equals(primary.cycle_df)


def test_localized_b_smoothing_can_separate_a_flagged_b_equals_m_case():
    time = np.arange(10, dtype=float)
    thickness = np.array(
        [0.0, 1.0, 2.0, 3.0, 3.05, 3.06, 3.07, 2.0, 1.0, 0.0]
    )
    primary = _result_from_row(_row(time, thickness, 0, 6, 6, 6, 9))

    refined = _apply_localized_equality_smoothing(
        primary_result=primary,
        time_values=time,
        thickness_values=thickness,
        recovered_min_indices=None,
        recovered_max_indices=None,
        primary_smoothing_window=3,
        plateau_fraction=0.35,
        polyorder=2,
        local_window=3,
    )

    row = refined.cycle_df.iloc[0]
    assert int(row["Point B Index"]) < int(row["Max Anchor Index"])
    assert int(row["Point C Index"]) == int(row["Max Anchor Index"])


def test_localized_c_smoothing_can_separate_a_flagged_c_equals_m_case():
    time = np.arange(11, dtype=float)
    thickness = np.array(
        [0.0, 1.0, 2.0, 2.8, 3.0, 3.05, 3.03, 3.01, 2.0, 1.0, 0.0]
    )
    primary = _result_from_row(_row(time, thickness, 0, 4, 5, 5, 10))

    refined = _apply_localized_equality_smoothing(
        primary_result=primary,
        time_values=time,
        thickness_values=thickness,
        recovered_min_indices=None,
        recovered_max_indices=None,
        primary_smoothing_window=3,
        plateau_fraction=0.35,
        polyorder=2,
        local_window=3,
    )

    row = refined.cycle_df.iloc[0]
    assert int(row["Point B Index"]) == 4
    assert int(row["Point C Index"]) > int(row["Max Anchor Index"])


def test_true_triangle_equality_remains_valid_under_localized_smoothing():
    time = np.arange(7, dtype=float)
    thickness = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    primary = _result_from_row(_row(time, thickness, 0, 3, 3, 3, 6))

    refined = _apply_localized_equality_smoothing(
        primary_result=primary,
        time_values=time,
        thickness_values=thickness,
        recovered_min_indices=None,
        recovered_max_indices=None,
        primary_smoothing_window=3,
        plateau_fraction=0.35,
        polyorder=2,
        local_window=5,
    )

    row = refined.cycle_df.iloc[0]
    assert int(row["Point B Index"]) == 3
    assert int(row["Point C Index"]) == 3
