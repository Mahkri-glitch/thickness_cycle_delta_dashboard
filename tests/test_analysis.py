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


def test_etch_onset_is_before_abrupt_drop():
    time = np.arange(21, dtype=float)
    # Purge/plateau through t=10, then a near-instantaneous etch drop.
    thickness = np.array(
        [10.0] * 11 + [7.0, 6.0, 5.0] + [5.0] * 7,
        dtype=float,
    )

    transition_idx = _detect_transition_index(
        time,
        thickness,
        max_idx=0,
        min2_idx=20,
        smoothing_window=5,
        onset_fraction=0.35,
        persistence=2,
    )

    assert transition_idx is not None
    # C should land on the purge-side edge, not the center of the drop.
    assert 9 <= transition_idx <= 10


def test_etch_onset_handles_gradual_transition():
    time = np.arange(51, dtype=float)
    thickness = (
        10.0
        - 0.03 * time
        - 1.5 / (1.0 + np.exp(-(time - 25.0) / 3.0))
    )

    transition_idx = _detect_transition_index(
        time,
        thickness,
        max_idx=0,
        min2_idx=50,
        smoothing_window=5,
        onset_fraction=0.35,
        persistence=2,
    )

    assert transition_idx is not None
    # Strongest slope is near 25; onset should be earlier.
    assert 15 <= transition_idx < 25


def test_lower_onset_fraction_moves_c_earlier_or_equal():
    time = np.arange(51, dtype=float)
    thickness = (
        10.0
        - 0.03 * time
        - 1.5 / (1.0 + np.exp(-(time - 25.0) / 3.0))
    )

    early = _detect_transition_index(
        time,
        thickness,
        0,
        50,
        smoothing_window=5,
        onset_fraction=0.20,
        persistence=2,
    )
    late = _detect_transition_index(
        time,
        thickness,
        0,
        50,
        smoothing_window=5,
        onset_fraction=0.60,
        persistence=2,
    )

    assert early is not None and late is not None
    assert early <= late


def test_ald_ale_cycle_math_and_output_columns():
    time = np.arange(24, dtype=float)
    thickness = np.array(
        [1.0, 2.0, 3.0] + [3.0] * 8 + [2.0, 1.0, 0.5] + [0.5] * 10,
        dtype=float,
    )
    events = pd.DataFrame(
        [
            {"Index": 0, "Type": "min", "Time": 0.0, "Thickness": thickness[0]},
            {"Index": 2, "Type": "max", "Time": 2.0, "Thickness": thickness[2]},
            {"Index": 23, "Type": "min", "Time": 23.0, "Thickness": thickness[23]},
        ]
    )

    result = calculate_cycles(
        events,
        time,
        thickness,
        transition_smoothing_window=5,
        transition_onset_fraction=0.35,
        transition_persistence=2,
    )

    assert len(result.cycle_df) == 1
    row = result.cycle_df.iloc[0]
    assert row["Delta 1"] == thickness[2] - thickness[0]
    assert np.isclose(
        row["Delta 2"] + row["Delta 3"],
        thickness[2] - thickness[23],
    )
    assert 2 < row["Point C Index"] < 23
    assert list(result.cycle_df.columns) == OUTPUT_COLUMNS


def test_format_cycle_results_on_empty_dataframe():
    formatted = format_cycle_results(pd.DataFrame())
    assert list(formatted.columns) == OUTPUT_COLUMNS
    assert formatted.empty
