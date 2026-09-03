import numpy as np
import pandas as pd

from analysis import (
    OUTPUT_COLUMNS,
    _detect_transition_index,
    build_events,
    calculate_cycles,
    find_suspect_regions,
    format_cycle_results,
)


def test_build_events_is_sorted():
    time = np.arange(7, dtype=float)
    thickness = np.array([0, 1, 0, 2, 0, 1, 0], dtype=float)
    events = build_events(
        np.array([2, 4, 6]), np.array([1, 3, 5]), time, thickness
    )
    assert events["Index"].tolist() == [1, 2, 3, 4, 5, 6]


def test_find_suspect_regions_detects_missing_minimum():
    events = pd.DataFrame(
        [
            {"Index": 1, "Type": "min", "Time": 1.0, "Thickness": 0.0},
            {"Index": 2, "Type": "max", "Time": 2.0, "Thickness": 1.0},
            {"Index": 4, "Type": "max", "Time": 4.0, "Thickness": 1.2},
        ]
    )
    suspects = find_suspect_regions(events)
    assert len(suspects) == 1
    assert suspects[0]["Missing Type"] == "min"


def test_transition_detector_rejects_narrow_terminal_step():
    time = np.arange(51, dtype=float)

    # Broad process transition centered near t=25 plus a much larger, nearly
    # instantaneous terminal drop near t=45. The old raw second-derivative
    # method tended to lock onto the terminal step.
    thickness = 10.0 - 0.03 * time - 1.5 / (1.0 + np.exp(-(time - 25.0) / 3.0))
    thickness[time >= 45] -= 2.5

    transition_idx = _detect_transition_index(
        time,
        thickness,
        max_idx=0,
        min2_idx=50,
        smoothing_window=11,
        polyorder=2,
        min_width=5,
    )

    assert transition_idx is not None
    assert 15 <= transition_idx <= 30


def test_ald_ale_cycle_math_and_output_columns():
    time = np.arange(36, dtype=float)
    thickness = np.empty_like(time)
    thickness[0] = 1.0
    thickness[1] = 3.0
    thickness[2:] = (
        5.0
        - 0.03 * (time[2:] - 2.0)
        - 2.0 / (1.0 + np.exp(-(time[2:] - 18.0) / 3.0))
    )

    events = pd.DataFrame(
        [
            {"Index": 0, "Type": "min", "Time": 0.0, "Thickness": thickness[0]},
            {"Index": 2, "Type": "max", "Time": 2.0, "Thickness": thickness[2]},
            {"Index": 35, "Type": "min", "Time": 35.0, "Thickness": thickness[35]},
        ]
    )

    result = calculate_cycles(
        events,
        time,
        thickness,
        transition_smoothing_window=11,
        transition_polyorder=2,
        transition_min_width=5,
    )

    assert len(result.cycle_df) == 1
    row = result.cycle_df.iloc[0]
    assert row["Delta 1"] == thickness[2] - thickness[0]
    assert np.isclose(row["Delta 2"] + row["Delta 3"], thickness[2] - thickness[35])
    assert 2 < row["Point C Index"] < 35
    assert list(result.cycle_df.columns) == OUTPUT_COLUMNS


def test_format_cycle_results_on_empty_dataframe():
    formatted = format_cycle_results(pd.DataFrame())
    assert list(formatted.columns) == OUTPUT_COLUMNS
    assert formatted.empty
