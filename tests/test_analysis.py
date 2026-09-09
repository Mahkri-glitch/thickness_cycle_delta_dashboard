import numpy as np
import pandas as pd

from analysis import (
    OUTPUT_COLUMNS,
    _detect_side_transition_index,
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


def test_rising_and_falling_transitions_stay_off_maximum():
    time = np.arange(41, dtype=float)
    thickness = np.array(
        [
            1.00, 1.00, 1.00, 1.01, 1.02, 1.05, 1.20, 1.70, 2.60, 3.60,
            4.20, 4.50, 4.65, 4.75, 4.82, 4.88, 4.92, 4.95, 4.97, 4.99,
            5.00,
            4.99, 4.98, 4.97, 4.95, 4.90, 4.70, 4.20, 3.40, 2.50,
            1.80, 1.40, 1.20, 1.10, 1.05, 1.03, 1.02, 1.01, 1.00, 1.00,
            1.00,
        ],
        dtype=float,
    )

    point_b = _detect_side_transition_index(
        time,
        thickness,
        start_idx=0,
        end_idx=20,
        direction="rising",
        smoothing_window=3,
        onset_fraction=0.35,
        persistence=2,
    )
    point_c = _detect_side_transition_index(
        time,
        thickness,
        start_idx=20,
        end_idx=40,
        direction="falling",
        smoothing_window=3,
        onset_fraction=0.35,
        persistence=2,
    )

    assert point_b is not None and point_c is not None
    assert 0 < point_b < 20 < point_c < 40


def test_falling_wrapper_keeps_c_after_maximum():
    time = np.arange(21, dtype=float)
    thickness = np.array(
        [5.0, 5.0, 5.0, 4.99, 4.98, 4.95, 4.8, 4.2, 3.0, 2.0]
        + [1.5] * 11,
        dtype=float,
    )
    transition_idx = _detect_transition_index(
        time,
        thickness,
        max_idx=0,
        min2_idx=20,
        smoothing_window=3,
        onset_fraction=0.35,
        persistence=2,
    )
    assert transition_idx is not None
    assert 0 < transition_idx < 20


def test_cycle_requires_two_distinct_transitions():
    time = np.arange(41, dtype=float)
    thickness = np.array(
        [
            1.00, 1.00, 1.00, 1.01, 1.02, 1.05, 1.20, 1.70, 2.60, 3.60,
            4.20, 4.50, 4.65, 4.75, 4.82, 4.88, 4.92, 4.95, 4.97, 4.99,
            5.00,
            4.99, 4.98, 4.97, 4.95, 4.90, 4.70, 4.20, 3.40, 2.50,
            1.80, 1.40, 1.20, 1.10, 1.05, 1.03, 1.02, 1.01, 1.00, 1.00,
            1.00,
        ],
        dtype=float,
    )
    events = pd.DataFrame(
        [
            {"Index": 0, "Type": "min", "Time": 0.0, "Thickness": thickness[0]},
            {"Index": 20, "Type": "max", "Time": 20.0, "Thickness": thickness[20]},
            {"Index": 40, "Type": "min", "Time": 40.0, "Thickness": thickness[40]},
        ]
    )

    result = calculate_cycles(
        events,
        time,
        thickness,
        transition_smoothing_window=3,
        transition_onset_fraction=0.35,
        transition_persistence=2,
    )

    assert len(result.cycle_df) == 1
    row = result.cycle_df.iloc[0]

    b = int(row["Point B Index"])
    c = int(row["Point C Index"])
    assert 0 < b < 20 < c < 40
    assert row["Point B Index"] != 20
    assert row["Point C Index"] != 20
    assert np.isclose(row["Delta 1"], thickness[b] - thickness[0])
    assert np.isclose(row["Delta 2"], thickness[b] - thickness[c])
    assert np.isclose(row["Delta 3"], thickness[c] - thickness[40])
    assert list(result.cycle_df.columns) == OUTPUT_COLUMNS


def test_unresolved_side_rejects_cycle():
    time = np.arange(12, dtype=float)
    thickness = np.array(
        [1.0, 2.0, 3.0, 2.9, 2.8, 2.6, 2.0, 1.5, 1.2, 1.1, 1.0, 1.0],
        dtype=float,
    )
    events = pd.DataFrame(
        [
            {"Index": 0, "Type": "min", "Time": 0.0, "Thickness": thickness[0]},
            {"Index": 2, "Type": "max", "Time": 2.0, "Thickness": thickness[2]},
            {"Index": 11, "Type": "min", "Time": 11.0, "Thickness": thickness[11]},
        ]
    )

    result = calculate_cycles(
        events,
        time,
        thickness,
        transition_smoothing_window=3,
        transition_onset_fraction=0.35,
        transition_persistence=2,
    )

    assert result.cycle_df.empty
    assert result.derivative_failures == 1


def test_format_cycle_results_on_empty_dataframe():
    formatted = format_cycle_results(pd.DataFrame())
    assert list(formatted.columns) == OUTPUT_COLUMNS
    assert formatted.empty
