import numpy as np
import pandas as pd

from analysis import (
    OUTPUT_COLUMNS,
    _detect_plateau_transition_indices,
    _detect_transition_index,
    build_events,
    calculate_cycles,
    find_suspect_regions,
    format_cycle_results,
    recover_missing_extremum,
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


def test_missing_point_recovery_accepts_flat_minimum():
    thickness = np.array([3.0, 2.0, 1.0, 1.0, 1.0, 2.0, 3.0])
    issue = {"Missing Type": "min", "Start Index": 0, "End Index": 6}

    recovery = recover_missing_extremum(
        thickness_values=thickness,
        min_indices=np.array([], dtype=int),
        max_indices=np.array([0, 6], dtype=int),
        issue=issue,
        recovery_order=1,
    )

    assert recovery.recovered_min_indices == [3]


def test_missing_point_recovery_accepts_flat_maximum():
    thickness = np.array([0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0])
    issue = {"Missing Type": "max", "Start Index": 0, "End Index": 6}

    recovery = recover_missing_extremum(
        thickness_values=thickness,
        min_indices=np.array([0, 6], dtype=int),
        max_indices=np.array([], dtype=int),
        issue=issue,
        recovery_order=1,
    )

    assert recovery.recovered_max_indices == [3]


def test_missing_point_recovery_rejects_flat_shoulder():
    thickness = np.array([5.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    issue = {"Missing Type": "min", "Start Index": 0, "End Index": 6}

    recovery = recover_missing_extremum(
        thickness_values=thickness,
        min_indices=np.array([], dtype=int),
        max_indices=np.array([0, 6], dtype=int),
        issue=issue,
        recovery_order=1,
    )

    assert recovery.recovered_min_indices == []


def test_plateau_edges_are_second_transition_after_minimum_and_before_fall():
    time = np.arange(10, dtype=float)
    thickness = np.array(
        [0.0, 1.0, 2.0, 3.0, 3.05, 3.02, 3.00, 2.0, 1.0, 0.0],
        dtype=float,
    )

    transitions = _detect_plateau_transition_indices(
        time,
        thickness,
        min1_idx=0,
        max_idx=4,
        min2_idx=9,
        smoothing_window=3,
        plateau_fraction=0.35,
    )

    assert transitions == (3, 6)


def test_slight_positive_or_negative_purge_drift_is_allowed():
    time = np.arange(13, dtype=float)
    thickness = np.array(
        [0.0, 0.8, 1.7, 2.7, 3.0, 3.04, 3.02, 3.05, 3.01, 2.9, 2.0, 1.0, 0.0],
        dtype=float,
    )

    transitions = _detect_plateau_transition_indices(
        time,
        thickness,
        min1_idx=0,
        max_idx=7,
        min2_idx=12,
        smoothing_window=3,
        plateau_fraction=0.35,
    )

    assert transitions is not None
    b, c = transitions
    assert 0 < b <= 7 <= c < 12
    assert b > 1
    assert c >= 7


def test_isolated_downward_excursion_during_purge_does_not_end_c():
    time = np.arange(13, dtype=float)
    thickness = np.array(
        [
            0.0,
            1.0,
            2.0,
            3.0,
            3.05,
            3.03,
            2.70,
            2.69,
            2.68,
            2.00,
            1.20,
            0.40,
            0.00,
        ],
        dtype=float,
    )

    transitions = _detect_plateau_transition_indices(
        time,
        thickness,
        min1_idx=0,
        max_idx=4,
        min2_idx=12,
        smoothing_window=3,
        plateau_fraction=0.35,
    )

    assert transitions == (3, 8)


def test_single_strong_drop_can_define_point_c():
    time = np.arange(10, dtype=float)
    thickness = np.array(
        [0.0, 1.0, 2.0, 3.0, 3.05, 3.04, 3.03, 1.00, 1.00, 0.90],
        dtype=float,
    )

    transitions = _detect_plateau_transition_indices(
        time,
        thickness,
        min1_idx=0,
        max_idx=4,
        min2_idx=9,
        smoothing_window=3,
        plateau_fraction=0.35,
    )

    assert transitions == (3, 6)


def test_transition_band_ignores_near_max_blip_for_point_b():
    slopes = np.array([1.0, 1.0, 1.0, 0.45, 0.10, 0.10, 0.50, 0.10, -1.0])
    thickness = np.cumsum(np.r_[0.0, slopes])
    time = np.arange(len(thickness), dtype=float)

    transitions = _detect_plateau_transition_indices(
        time,
        thickness,
        min1_idx=0,
        max_idx=8,
        min2_idx=9,
        smoothing_window=3,
        plateau_fraction=0.35,
    )

    assert transitions == (4, 8)


def test_transition_band_moves_c_through_gradual_purge_drift():
    # Cycle-143-style shape: two clearly purge-like intervals after M, then
    # several moderate negative slopes, followed by an unmistakable active fall.
    slopes = np.array(
        [
            1.0,
            1.0,
            1.0,
            0.05,
            0.02,
            -0.02,
            -0.02,
            -0.20,
            -0.20,
            -0.20,
            -0.25,
            -2.00,
            -0.50,
            -0.50,
            -0.10,
        ]
    )
    thickness = np.cumsum(np.r_[0.0, slopes])
    time = np.arange(len(thickness), dtype=float)

    transitions = _detect_plateau_transition_indices(
        time,
        thickness,
        min1_idx=0,
        max_idx=5,
        min2_idx=15,
        smoothing_window=3,
        plateau_fraction=0.35,
    )

    assert transitions == (3, 9)


def test_isolated_smoothing_prevents_post_max_drop_from_smearing_into_b():
    # With whole-cycle SG smoothing, the extreme post-M drop can bend the fitted
    # pre-M samples downward and collapse B onto M. A->M and M->D are now smoothed
    # independently, so B remains at the actual rise->purge boundary while C=M is
    # still valid because the fall genuinely begins immediately after the maximum.
    time = np.arange(9, dtype=float)
    thickness = np.array(
        [0.0, 1.0, 2.0, 3.0, 3.10, 3.11, -5.0, -6.0, -7.0],
        dtype=float,
    )

    transitions = _detect_plateau_transition_indices(
        time,
        thickness,
        min1_idx=0,
        max_idx=5,
        min2_idx=8,
        smoothing_window=5,
        plateau_fraction=0.35,
    )

    assert transitions == (3, 5)


def test_maximum_can_be_b_and_c_when_no_plateau_is_resolved():
    time = np.arange(7, dtype=float)
    thickness = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0], dtype=float)

    transitions = _detect_plateau_transition_indices(
        time,
        thickness,
        min1_idx=0,
        max_idx=3,
        min2_idx=6,
        smoothing_window=3,
        plateau_fraction=0.35,
    )

    assert transitions == (3, 3)


def test_falling_wrapper_finds_sustained_or_instantaneous_fall_start():
    time = np.arange(8, dtype=float)
    thickness = np.array([3.0, 3.02, 3.01, 3.00, 2.95, 2.0, 1.0, 0.0])

    transition_idx = _detect_transition_index(
        time,
        thickness,
        max_idx=0,
        min2_idx=7,
        smoothing_window=3,
        onset_fraction=0.35,
    )

    assert transition_idx is not None
    assert 0 <= transition_idx < 7


def test_cycle_math_uses_plateau_edges_and_stores_max_anchor():
    time = np.arange(10, dtype=float)
    thickness = np.array(
        [0.0, 1.0, 2.0, 3.0, 3.05, 3.02, 3.00, 2.0, 1.0, 0.0],
        dtype=float,
    )
    events = pd.DataFrame(
        [
            {"Index": 0, "Type": "min", "Time": 0.0, "Thickness": thickness[0]},
            {"Index": 4, "Type": "max", "Time": 4.0, "Thickness": thickness[4]},
            {"Index": 9, "Type": "min", "Time": 9.0, "Thickness": thickness[9]},
        ]
    )

    result = calculate_cycles(
        events,
        time,
        thickness,
        transition_smoothing_window=3,
        transition_onset_fraction=0.35,
    )

    assert len(result.cycle_df) == 1
    row = result.cycle_df.iloc[0]
    b = int(row["Point B Index"])
    c = int(row["Point C Index"])

    assert (b, c) == (3, 6)
    assert int(row["Max Anchor Index"]) == 4
    assert row["Max Anchor Time"] == time[4]
    assert row["Max Anchor Thickness"] == thickness[4]
    assert np.isclose(row["Delta 1"], thickness[b] - thickness[0])
    assert np.isclose(row["Delta 2"], thickness[b] - thickness[c])
    assert np.isclose(row["Delta 3"], thickness[c] - thickness[9])
    assert list(result.cycle_df.columns) == OUTPUT_COLUMNS


def test_format_cycle_results_on_empty_dataframe():
    formatted = format_cycle_results(pd.DataFrame())
    assert list(formatted.columns) == OUTPUT_COLUMNS
    assert formatted.empty
