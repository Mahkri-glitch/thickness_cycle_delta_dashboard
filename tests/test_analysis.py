import numpy as np
import pandas as pd

from analysis import (
    OUTPUT_COLUMNS,
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


def test_two_step_cycle_math_and_output_columns():
    time = np.arange(5, dtype=float)
    thickness = np.array([1.0, 1.5, 3.0, 2.0, 1.0])
    events = pd.DataFrame(
        [
            {"Index": 0, "Type": "min", "Time": 0.0, "Thickness": 1.0},
            {"Index": 2, "Type": "max", "Time": 2.0, "Thickness": 3.0},
            {"Index": 4, "Type": "min", "Time": 4.0, "Thickness": 1.0},
        ]
    )

    result = calculate_cycles(events, time, thickness, "2-step")
    row = result.cycle_df.iloc[0]

    assert row["Delta 1"] == 2.0
    assert row["Delta 2"] == 2.0
    assert np.isnan(row["Delta 3"])
    assert list(result.cycle_df.columns) == OUTPUT_COLUMNS


def test_format_cycle_results_on_empty_dataframe():
    formatted = format_cycle_results(pd.DataFrame())
    assert list(formatted.columns) == OUTPUT_COLUMNS
    assert formatted.empty
