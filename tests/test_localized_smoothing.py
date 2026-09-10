import numpy as np

from analysis import (
    _default_local_radius,
    _detect_plateau_transition_indices,
    _localized_refine_point_b,
)


def test_default_local_radius_stays_compact_relative_to_sg_window():
    assert _default_local_radius(3) == 3
    assert _default_local_radius(5) == 4
    assert _default_local_radius(7) == 5


def test_localized_refinement_ignores_data_outside_b_neighborhood():
    time = np.arange(12, dtype=float)

    # The two traces are identical around preliminary B=5 but radically different
    # far away. Local B refinement should therefore return the same point.
    base = np.array(
        [0.0, 1.0, 2.0, 3.0, 3.8, 4.0, 4.05, 4.06, 4.07, 4.08, 4.09, 4.10],
        dtype=float,
    )
    altered = base.copy()
    altered[0] = -1000.0
    altered[10] = 900.0
    altered[11] = -900.0

    kwargs = dict(
        segment_time=time,
        preliminary_b_idx=5,
        max_local_idx=9,
        local_radius=2,
        smoothing_window=3,
        plateau_limit=0.35,
        active_limit=0.65,
        polyorder=2,
    )
    first = _localized_refine_point_b(segment_thickness=base, **kwargs)
    second = _localized_refine_point_b(segment_thickness=altered, **kwargs)

    assert first == second


def test_direct_triangle_still_allows_b_and_c_to_equal_maximum():
    time = np.arange(7, dtype=float)
    thickness = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0], dtype=float)

    transitions = _detect_plateau_transition_indices(
        time,
        thickness,
        min1_idx=0,
        max_idx=3,
        min2_idx=6,
        smoothing_window=5,
        plateau_fraction=0.35,
    )

    assert transitions == (3, 3)


def test_local_smoothing_does_not_let_post_max_drop_pull_b_to_maximum():
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


def test_gradual_post_max_drift_can_move_c_later_without_forcing_it():
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
