"""Streamlit user interface for cyclic ALD/ALE thickness delta analysis."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Reload local analysis logic on Streamlit reruns so deployments do not retain
# an older calculate_cycles() signature in memory.
import analysis as analysis_core

analysis_core = importlib.reload(analysis_core)

build_events = analysis_core.build_events
calculate_cycles = analysis_core.calculate_cycles
detect_extrema = analysis_core.detect_extrema
downsample_for_plot = analysis_core.downsample_for_plot
find_suspect_regions = analysis_core.find_suspect_regions
recover_missing_extremum = analysis_core.recover_missing_extremum
flat_aware_local_extrema_candidates = analysis_core._flat_aware_local_extrema_candidates

warnings.simplefilter(action="ignore", category=FutureWarning)


def _normalize_header(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _is_time_header(value) -> bool:
    text = _normalize_header(value)
    return (
        "time" in text
        or text in {"t", "seconds", "second", "sec", "secs"}
        or text.startswith("elapsed")
    )


def _is_thickness_header(value) -> bool:
    text = _normalize_header(value)
    return (
        "thickness" in text
        or "thick" in text
        or "thk" in text
        or text in {"film thickness", "film_thickness"}
    )


def detect_excel_header_row(
    file_bytes: bytes,
    sheet_name: str,
    scan_rows: int = 30,
) -> int | None:
    """Find a row containing separate time and thickness header cells."""
    preview = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
        header=None,
        nrows=scan_rows,
    )

    for row_idx in range(len(preview)):
        row_values = preview.iloc[row_idx].tolist()
        time_positions = [
            col_idx
            for col_idx, value in enumerate(row_values)
            if _is_time_header(value)
        ]
        thickness_positions = [
            col_idx
            for col_idx, value in enumerate(row_values)
            if _is_thickness_header(value)
        ]
        if any(
            time_idx != thickness_idx
            for time_idx in time_positions
            for thickness_idx in thickness_positions
        ):
            return row_idx
    return None


def find_default_column_index(columns: list, kind: str) -> int | None:
    matcher = _is_time_header if kind == "time" else _is_thickness_header
    for idx, column in enumerate(columns):
        if matcher(column):
            return idx
    return None


@st.cache_data
def get_excel_sheets(file_bytes: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


@st.cache_data
def load_excel(file_bytes: bytes, sheet_name: str) -> tuple[pd.DataFrame, int | None]:
    header_row = detect_excel_header_row(file_bytes, sheet_name)
    if header_row is None:
        return (
            pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=0),
            None,
        )

    return (
        pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet_name,
            header=header_row,
        ),
        header_row,
    )


@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def prepare_numeric_data(
    df: pd.DataFrame, time_col: str, thickness_col: str
) -> pd.DataFrame:
    cleaned = df[[time_col, thickness_col]].copy()
    cleaned[time_col] = pd.to_numeric(cleaned[time_col], errors="coerce")
    cleaned[thickness_col] = pd.to_numeric(cleaned[thickness_col], errors="coerce")
    cleaned = cleaned.dropna().copy()
    return cleaned.sort_values(time_col, kind="mergesort").reset_index(drop=True)


def classify_time_order(df: pd.DataFrame, time_col: str) -> str:
    values = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return "insufficient"
    diffs = np.diff(values)
    if np.all(diffs >= 0):
        return "ascending"
    if np.all(diffs <= 0):
        return "descending"
    return "mixed"


def _recovery_issue_key(issue: dict) -> str:
    return (
        f"{issue['Missing Type']}:"
        f"{int(issue['Start Index'])}:"
        f"{int(issue['End Index'])}"
    )


def _approx_cycle_number(issue: dict, base_min_indices: np.ndarray) -> int:
    """Return a stable approximate cycle number for a broken extrema region."""
    start_idx = int(issue["Start Index"])
    completed_starts = int(np.count_nonzero(base_min_indices <= start_idx))
    return max(1, completed_starts)


def _candidate_index(recovery, missing_type: str) -> int | None:
    if missing_type == "min" and recovery.recovered_min_indices:
        return int(recovery.recovered_min_indices[0])
    if missing_type == "max" and recovery.recovered_max_indices:
        return int(recovery.recovered_max_indices[0])
    return None


def _is_flat_recovery_candidate(
    thickness_values: np.ndarray,
    candidate_idx: int,
    region_start: int,
    region_end: int,
) -> bool:
    """Return True when the recovery point belongs to a repeated-value plateau."""
    candidate_idx = int(candidate_idx)
    value = float(thickness_values[candidate_idx])
    left_equal = (
        candidate_idx > int(region_start)
        and np.isclose(
            float(thickness_values[candidate_idx - 1]),
            value,
            rtol=0.0,
            atol=1e-12,
        )
    )
    right_equal = (
        candidate_idx < int(region_end)
        and np.isclose(
            float(thickness_values[candidate_idx + 1]),
            value,
            rtol=0.0,
            atol=1e-12,
        )
    )
    return bool(left_equal or right_equal)


def _find_flat_only_recovery_candidate(
    thickness_values: np.ndarray,
    issue: dict,
    recovery_order: int = 1,
) -> int | None:
    """Find the strongest true flat extremum inside one broken region."""
    missing_type = str(issue["Missing Type"])
    region_start = int(issue["Start Index"])
    region_end = int(issue["End Index"])
    order = max(1, int(recovery_order))

    padding = max(order * 2, 5)
    padded_start = max(0, region_start - padding)
    padded_end = min(len(thickness_values) - 1, region_end + padding)
    local_thickness = np.asarray(
        thickness_values[padded_start : padded_end + 1], dtype=float
    )
    local_max_order = max(1, (len(local_thickness) - 1) // 2)
    effective_order = min(order, local_max_order)

    local_candidates = flat_aware_local_extrema_candidates(
        local_thickness=local_thickness,
        order=effective_order,
        missing_type=missing_type,
    )
    candidates = local_candidates + padded_start
    candidates = candidates[(candidates > region_start) & (candidates < region_end)]
    flat_candidates = np.asarray(
        [
            int(idx)
            for idx in candidates
            if _is_flat_recovery_candidate(
                thickness_values,
                int(idx),
                region_start,
                region_end,
            )
        ],
        dtype=int,
    )
    if len(flat_candidates) == 0:
        return None

    if missing_type == "min":
        return int(
            flat_candidates[
                np.argmin(np.asarray(thickness_values)[flat_candidates])
            ]
        )
    if missing_type == "max":
        return int(
            flat_candidates[
                np.argmax(np.asarray(thickness_values)[flat_candidates])
            ]
        )
    return None


def plot_full_dataset(df, time_col, thickness_col, start_time, end_time) -> None:
    x = df[time_col].to_numpy(dtype=float)
    y = df[thickness_col].to_numpy(dtype=float)
    plot_x, plot_y = downsample_for_plot(x, y, max_points=5000)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(plot_x, plot_y, linewidth=2, alpha=0.7)
    ax.axvline(start_time, linestyle="--", linewidth=2)
    ax.axvline(end_time, linestyle="--", linewidth=2)
    ax.set_xlabel(time_col)
    ax.set_ylabel(thickness_col)
    st.pyplot(fig)
    plt.close(fig)


def plot_cycle_analysis(
    time_values,
    thickness_values,
    min_indices,
    max_indices,
    recovered_min_indices,
    recovered_max_indices,
    point_b_indices,
    point_c_indices,
    recovered_point_b_indices,
    recovered_point_c_indices,
    time_col,
    thickness_col,
    selected_issue,
) -> None:
    plot_time, plot_thickness = downsample_for_plot(
        time_values, thickness_values, max_points=8000
    )

    fig, ax = plt.subplots(figsize=(16, 9))

    if selected_issue is not None:
        ax.axvspan(
            time_values[int(selected_issue["Start Index"])],
            time_values[int(selected_issue["End Index"])],
            alpha=0.12,
            label="Selected missing-point region",
        )

    ax.plot(
        plot_time,
        plot_thickness,
        linewidth=2,
        alpha=0.6,
        linestyle="--",
        label="Thickness",
    )
    ax.scatter(
        time_values[min_indices],
        thickness_values[min_indices],
        s=75,
        marker="o",
        label="Point A/D candidate (minimum)",
    )
    ax.scatter(
        time_values[max_indices],
        thickness_values[max_indices],
        s=75,
        marker="s",
        label="Maximum anchor",
    )

    if recovered_min_indices:
        ax.scatter(
            time_values[recovered_min_indices],
            thickness_values[recovered_min_indices],
            s=180,
            marker="X",
            label="Recovered minimum",
        )
    if recovered_max_indices:
        ax.scatter(
            time_values[recovered_max_indices],
            thickness_values[recovered_max_indices],
            s=180,
            marker="P",
            label="Recovered maximum anchor",
        )
    if point_b_indices:
        ax.scatter(
            time_values[point_b_indices],
            thickness_values[point_b_indices],
            s=110,
            marker=">",
            label="Point B (purge plateau entry)",
        )
    if point_c_indices:
        ax.scatter(
            time_values[point_c_indices],
            thickness_values[point_c_indices],
            s=110,
            marker="^",
            label="Point C (fall onset)",
        )
    if recovered_point_b_indices:
        ax.scatter(
            time_values[recovered_point_b_indices],
            thickness_values[recovered_point_b_indices],
            s=160,
            marker="*",
            label="Point B from recovered cycle",
        )
    if recovered_point_c_indices:
        ax.scatter(
            time_values[recovered_point_c_indices],
            thickness_values[recovered_point_c_indices],
            s=160,
            marker="*",
            label="Point C from recovered cycle",
        )

    ax.set_title("ALD/ALE Process Delta Analyzer")
    ax.set_xlabel(time_col)
    ax.set_ylabel(thickness_col)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def plot_missing_point_region(
    issue: dict,
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    base_min_indices: np.ndarray,
    base_max_indices: np.ndarray,
    preview_candidate_idx: int | None,
    saved_candidate_idx: int | None,
    time_col: str,
    thickness_col: str,
    padding_points: int = 8,
) -> None:
    """Zoom into one broken extrema region and show preview/saved recovery points."""
    start_idx = int(issue["Start Index"])
    end_idx = int(issue["End Index"])
    left_idx = max(0, start_idx - int(padding_points))
    right_idx = min(len(time_values) - 1, end_idx + int(padding_points))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        time_values[left_idx : right_idx + 1],
        thickness_values[left_idx : right_idx + 1],
        linewidth=2,
        label="Thickness",
    )
    ax.axvspan(
        time_values[start_idx],
        time_values[end_idx],
        alpha=0.10,
        label="Local recovery search region",
    )

    local_mins = base_min_indices[
        (base_min_indices >= left_idx) & (base_min_indices <= right_idx)
    ]
    local_maxs = base_max_indices[
        (base_max_indices >= left_idx) & (base_max_indices <= right_idx)
    ]
    if len(local_mins):
        ax.scatter(
            time_values[local_mins],
            thickness_values[local_mins],
            s=90,
            marker="o",
            label="Global minima",
        )
    if len(local_maxs):
        ax.scatter(
            time_values[local_maxs],
            thickness_values[local_maxs],
            s=90,
            marker="s",
            label="Global maxima",
        )

    if saved_candidate_idx is not None:
        ax.scatter(
            [time_values[saved_candidate_idx]],
            [thickness_values[saved_candidate_idx]],
            s=210,
            marker="P",
            label="Saved recovered point",
        )
        ax.annotate(
            "SAVED",
            (time_values[saved_candidate_idx], thickness_values[saved_candidate_idx]),
            xytext=(0, 24),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )

    if (
        preview_candidate_idx is not None
        and preview_candidate_idx != saved_candidate_idx
    ):
        ax.scatter(
            [time_values[preview_candidate_idx]],
            [thickness_values[preview_candidate_idx]],
            s=180,
            marker="X",
            label="Current local-order preview",
        )
        ax.annotate(
            "PREVIEW",
            (time_values[preview_candidate_idx], thickness_values[preview_candidate_idx]),
            xytext=(0, -28),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )

    missing_type = str(issue["Missing Type"]).upper()
    ax.set_title(f"Missing-point region: possible missing {missing_type}")
    ax.set_xlabel(time_col)
    ax.set_ylabel(thickness_col)
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def plot_selected_cycle(
    cycle_row: pd.Series,
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    time_col: str,
    thickness_col: str,
    padding_points: int = 5,
) -> None:
    """Plot one accepted cycle with the exact A/B/max/C/D points used in analysis."""
    a_idx = int(cycle_row["Point A Index"])
    b_idx = int(cycle_row["Point B Index"])
    max_idx = int(cycle_row["Max Anchor Index"])
    c_idx = int(cycle_row["Point C Index"])
    d_idx = int(cycle_row["Point D Index"])

    left_idx = max(0, a_idx - int(padding_points))
    right_idx = min(len(time_values) - 1, d_idx + int(padding_points))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        time_values[left_idx : right_idx + 1],
        thickness_values[left_idx : right_idx + 1],
        linewidth=2,
        label="Thickness",
    )

    if b_idx < c_idx:
        ax.axvspan(
            time_values[b_idx],
            time_values[c_idx],
            alpha=0.10,
            label="Detected purge region",
        )

    point_specs = [
        ("A", a_idx, "o", 120, (6, -20)),
        ("B", b_idx, ">", 150, (-22, 18)),
        ("M", max_idx, "s", 150, (0, 32)),
        ("C", c_idx, "^", 150, (22, 18)),
        ("D", d_idx, "o", 120, (6, -20)),
    ]

    for label, idx, marker, size, offset in point_specs:
        ax.scatter(
            [time_values[idx]],
            [thickness_values[idx]],
            s=size,
            marker=marker,
            label=f"{label}: " + ("Maximum anchor" if label == "M" else f"Point {label}"),
        )
        ax.annotate(
            label,
            (time_values[idx], thickness_values[idx]),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )

    cycle_number = int(cycle_row["Cycle"])
    ax.set_title(f"Cycle {cycle_number}: isolated A/B/M/C/D inspection")
    ax.set_xlabel(time_col)
    ax.set_ylabel(thickness_col)
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def show_interpretation_guide() -> None:
    with st.expander("How to interpret the analysis"):
        st.markdown(
            """
**Complete-cycle rule**  
The extrema detector first identifies successive **minimum → maximum → minimum**
sequences in forward physical time.

**A/B/C/D definition**
- **Point A** = first minimum
- **Point B** = entry into the low-slope purge/plateau after the active rise
- **Maximum anchor** = a reference point inside or at an edge of the purge region
- **Point C** = onset of the active fall after purge
- **Point D** = next minimum
- **Δ1 = B − A**
- **Δ2 = B − C**
- **Δ3 = C − D**

A valid cycle follows **A < B ≤ maximum ≤ C < D**. The maximum is not forced to
be a separate process point: it may equal B, C, or both if that is what the
sampled trace resolves.

**Transition detection**  
B is found from the low-slope purge region after the active rise. C is intentionally
asymmetric: it is the first sustained two-interval fall, or one exceptionally
strong single drop when the reaction is resolved in only one sampling interval.

**Smoothing window**  
The minimum is **3 samples** so short ellipsometry transitions are not
unnecessarily smeared.

**Purge plateau threshold**  
This controls how much local slope can still count as purge relative to the active
rise/fall rate.

**Individual cycle inspector**  
For long datasets, select any accepted cycle below the main plot to see a zoomed
view with the exact **A, B, maximum anchor, C, and D** used for the calculations.

**Per-region missing-point recovery**  
Broken **MAX → MAX** or **MIN → MIN** regions can each have their own local extrema
order. Changing the local order previews a candidate only in that region. Use
**Save/apply** to keep that order while you tune other regions. Saved local orders
are reapplied independently during the current Streamlit session and do not change
the global minimum/maximum orders.

**Bulk flat recovery**  
Use **Save/apply all flat missing points** to scan every broken region with local
order 1 and save only genuine repeated-value flat minima/maxima. Ordinary sharp
candidates are left for manual review, and existing manual saves are not
overwritten.

**Time direction**  
Uploaded data are automatically sorted by the selected time column before
analysis.

**Ellipsometer Excel headers**  
For Excel files, the dashboard scans the first 30 rows for separate time and
thickness header cells. A title such as **Thickness vs Time** in one cell is
ignored.
            """
        )


st.set_page_config(page_title="Thickness Cycle Delta Analyzer", layout="wide")
st.title("Thickness Cycle Delta Analyzer")
st.write(
    "Analyze cyclic thickness-vs-time data for the ALD/ALE process using "
    "minimum/maximum anchors, purge plateau boundaries, and Δ1/Δ2/Δ3 calculations."
)
show_interpretation_guide()

uploaded_file = st.file_uploader(
    "Upload a thickness-vs-time file", type=["xlsx", "xls", "csv"]
)
if uploaded_file is None:
    st.info("Upload an Excel or CSV file to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()
detected_header_row = None

if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
    sheet_name = st.selectbox("Sheet", get_excel_sheets(file_bytes))
    raw_df, detected_header_row = load_excel(file_bytes, sheet_name)
else:
    raw_df = load_csv(file_bytes)

if detected_header_row is not None and detected_header_row > 0:
    st.info(
        f"Detected the data headers on Excel row {detected_header_row + 1}; "
        "rows above it were treated as ellipsometer metadata."
    )
elif uploaded_file.name.lower().endswith((".xlsx", ".xls")) and detected_header_row is None:
    st.warning(
        "Could not automatically identify a row containing separate time and thickness "
        "headers in the first 30 rows. The first row was used as the header; "
        "verify the column selections below."
    )

if len(raw_df.columns) < 2:
    st.error("The file needs at least two columns.")
    st.stop()

columns = list(raw_df.columns)
time_match = find_default_column_index(columns, "time")
thickness_match = find_default_column_index(columns, "thickness")

default_time = time_match if time_match is not None else 0
default_thickness = (
    thickness_match if thickness_match is not None else min(1, len(columns) - 1)
)

left, right = st.columns(2)
with left:
    time_col = st.selectbox("Time column", columns, index=default_time)
with right:
    thickness_col = st.selectbox("Thickness column", columns, index=default_thickness)

if time_col == thickness_col:
    st.error("Choose different columns for time and thickness.")
    st.stop()

input_time_order = classify_time_order(raw_df, time_col)
full_df = prepare_numeric_data(raw_df, time_col, thickness_col)

if len(full_df) < 3:
    st.error("Not enough numeric data points were found after cleaning.")
    st.stop()

if input_time_order == "descending":
    st.info(
        "The uploaded time column runs backward. It has been reordered into "
        "ascending physical time before A/B/C/D detection."
    )
elif input_time_order == "mixed":
    st.warning(
        "The uploaded time column is not monotonic. Rows have been sorted by time "
        "before analysis; verify that this is appropriate for the dataset."
    )

st.caption(
    "Point indices refer to the chronologically sorted analysis window; "
    "Point times are the recommended reference when comparing files."
)

st.sidebar.header("Analysis window")
window_start, window_end = st.sidebar.slider(
    "Select data range",
    min_value=0,
    max_value=len(full_df) - 1,
    value=(0, len(full_df) - 1),
    step=1,
)

analysis_df = full_df.iloc[window_start : window_end + 1].reset_index(drop=True)
time_values = analysis_df[time_col].to_numpy(dtype=float)
thickness_values = analysis_df[thickness_col].to_numpy(dtype=float)

start_time = float(full_df.iloc[window_start][time_col])
end_time = float(full_df.iloc[window_end][time_col])

st.subheader("Select Analysis Window")
plot_full_dataset(full_df, time_col, thickness_col, start_time, end_time)

max_allowed_order = max(1, min(250, (len(full_df) - 1) // 2))
default_order = min(10, max_allowed_order)

st.sidebar.header("ALD/ALE Process")
st.sidebar.caption("A → B → purge/max anchor → C → D")
st.sidebar.caption("Δ1 = B − A, Δ2 = B − C, Δ3 = C − D")

st.sidebar.header("Extrema filters")
min_order = st.sidebar.slider(
    "Minimum filter order", 1, max_allowed_order, default_order, 1, key="min_order"
)
max_order = st.sidebar.slider(
    "Maximum filter order", 1, max_allowed_order, default_order, 1, key="max_order"
)

window_max_order = max(1, (len(analysis_df) - 1) // 2)
if min_order > window_max_order or max_order > window_max_order:
    st.sidebar.warning(
        "The selected extrema order is large relative to the current analysis window."
    )

st.sidebar.header("Purge plateau detection")
max_transition_window = min(51, max(3, len(analysis_df)))
if max_transition_window % 2 == 0:
    max_transition_window -= 1
max_transition_window = max(3, max_transition_window)

transition_smoothing_window = st.sidebar.slider(
    "Smoothing window (samples)",
    min_value=3,
    max_value=max_transition_window,
    value=3,
    step=2,
    help="Light smoothing before interval slopes are calculated.",
)

plateau_percent = st.sidebar.slider(
    "Purge plateau threshold (% of active slope)",
    min_value=10,
    max_value=80,
    value=35,
    step=5,
    help=(
        "Lower values require a flatter purge region. Higher values allow more "
        "positive/negative drift in the purge region."
    ),
)
transition_onset_fraction = plateau_percent / 100.0

# Global extrema establish the base event sequence. Local recovery never changes
# these global orders; saved overrides are layered on afterward.
base_min_indices, base_max_indices = detect_extrema(
    thickness_values, min_order, max_order
)
primary_events = build_events(
    base_min_indices, base_max_indices, time_values, thickness_values
)
suspect_regions = find_suspect_regions(primary_events)

min_indices = base_min_indices.copy()
max_indices = base_max_indices.copy()
recovered_min_indices: list[int] = []
recovered_max_indices: list[int] = []
selected_issue: dict | None = None
selected_preview_candidate_idx: int | None = None
selected_saved_candidate_idx: int | None = None
selected_issue_saved_order: int | None = None
selected_issue_approx_cycle: int | None = None

# Saved orders are scoped to the uploaded file + selected analysis window/columns.
file_token = hashlib.sha1(file_bytes).hexdigest()[:12]
recovery_context_key = (
    f"{file_token}:{window_start}:{window_end}:"
    f"{str(time_col)}:{str(thickness_col)}"
)
if "local_recovery_profiles" not in st.session_state:
    st.session_state["local_recovery_profiles"] = {}
all_recovery_profiles = st.session_state["local_recovery_profiles"]
saved_recovery_orders = all_recovery_profiles.setdefault(recovery_context_key, {})

# Bulk-flat recoveries store the exact candidate index as well as order=1 so a
# later general local search cannot replace the flat point with a sharp candidate.
if "bulk_flat_recovery_candidates" not in st.session_state:
    st.session_state["bulk_flat_recovery_candidates"] = {}
all_bulk_flat_candidates = st.session_state["bulk_flat_recovery_candidates"]
bulk_flat_saved_candidates = all_bulk_flat_candidates.setdefault(
    recovery_context_key, {}
)

st.sidebar.header("Missing Point Recovery")
use_recovery = st.sidebar.checkbox("Enable missing-point recovery", value=False)

if use_recovery:
    notice_key = f"bulk_flat_notice:{recovery_context_key}"
    if notice_key in st.session_state:
        st.sidebar.success(st.session_state.pop(notice_key))

    current_issue_keys = {_recovery_issue_key(issue) for issue in suspect_regions}
    active_saved_count = sum(
        1 for key in saved_recovery_orders if key in current_issue_keys
    )
    active_bulk_flat_count = sum(
        1 for key in bulk_flat_saved_candidates if key in current_issue_keys
    )
    if active_saved_count:
        st.sidebar.caption(
            f"{active_saved_count} saved local recovery order(s) active "
            f"({active_bulk_flat_count} bulk-flat)."
        )

    if not suspect_regions:
        st.sidebar.caption("No broken MIN/MAX alternation was detected.")
    else:
        with st.sidebar.expander("Bulk flat-point recovery", expanded=False):
            st.caption(
                "Scans every broken region with local order 1 and saves only "
                "genuine repeated-value flat minima/maxima. Manual saves are "
                "left unchanged."
            )
            if st.button(
                "Save/apply all flat missing points",
                key=f"bulk_flat_apply:{recovery_context_key}",
                use_container_width=True,
            ):
                flat_found = 0
                newly_saved = 0
                manual_preserved = 0

                for issue in suspect_regions:
                    issue_key = _recovery_issue_key(issue)
                    flat_candidate_idx = _find_flat_only_recovery_candidate(
                        thickness_values=thickness_values,
                        issue=issue,
                        recovery_order=1,
                    )
                    if flat_candidate_idx is None:
                        continue

                    flat_found += 1
                    if (
                        issue_key in saved_recovery_orders
                        and issue_key not in bulk_flat_saved_candidates
                    ):
                        manual_preserved += 1
                        continue

                    saved_recovery_orders[issue_key] = 1
                    bulk_flat_saved_candidates[issue_key] = int(flat_candidate_idx)
                    newly_saved += 1

                st.session_state[notice_key] = (
                    f"Flat scan found {flat_found} region(s); saved/refreshed "
                    f"{newly_saved}. Preserved {manual_preserved} manual save(s)."
                )
                st.rerun()

        issue_number = st.sidebar.selectbox(
            "Missing-point region",
            list(range(len(suspect_regions))),
            format_func=lambda idx: (
                f"~Cycle {_approx_cycle_number(suspect_regions[idx], base_min_indices)}: "
                f"missing {suspect_regions[idx]['Missing Type'].upper()} "
                f"({suspect_regions[idx]['Start Time']:.3f}–"
                f"{suspect_regions[idx]['End Time']:.3f})"
                + (
                    " • flat-auto"
                    if _recovery_issue_key(suspect_regions[idx])
                    in bulk_flat_saved_candidates
                    else (
                        " • saved"
                        if _recovery_issue_key(suspect_regions[idx])
                        in saved_recovery_orders
                        else ""
                    )
                )
            ),
            key="missing_point_region_selector",
        )
        selected_issue = suspect_regions[issue_number]
        selected_issue_key = _recovery_issue_key(selected_issue)
        selected_issue_approx_cycle = _approx_cycle_number(
            selected_issue, base_min_indices
        )
        missing_type = str(selected_issue["Missing Type"])
        global_default_order = min_order if missing_type == "min" else max_order
        selected_issue_saved_order = saved_recovery_orders.get(selected_issue_key)
        slider_default = int(
            selected_issue_saved_order
            if selected_issue_saved_order is not None
            else global_default_order
        )
        slider_key = (
            f"local_recovery_order:{recovery_context_key}:{selected_issue_key}"
        )
        if slider_key not in st.session_state:
            st.session_state[slider_key] = slider_default

        recovery_order = st.sidebar.slider(
            f"Local {missing_type} order for this region",
            min_value=1,
            max_value=max_allowed_order,
            step=1,
            key=slider_key,
            help=(
                "This order is used only inside the selected broken region. "
                "Move the slider to preview the candidate, then save/apply it."
            ),
        )

        preview_recovery = recover_missing_extremum(
            thickness_values=thickness_values,
            min_indices=base_min_indices,
            max_indices=base_max_indices,
            issue=selected_issue,
            recovery_order=int(recovery_order),
        )
        selected_preview_candidate_idx = _candidate_index(
            preview_recovery, missing_type
        )

        if selected_issue_key in bulk_flat_saved_candidates:
            selected_saved_candidate_idx = int(
                bulk_flat_saved_candidates[selected_issue_key]
            )
        elif selected_issue_saved_order is not None:
            saved_preview = recover_missing_extremum(
                thickness_values=thickness_values,
                min_indices=base_min_indices,
                max_indices=base_max_indices,
                issue=selected_issue,
                recovery_order=int(selected_issue_saved_order),
            )
            selected_saved_candidate_idx = _candidate_index(
                saved_preview, missing_type
            )

        save_col, reset_col = st.sidebar.columns(2)
        with save_col:
            save_local = st.button(
                "Save/apply",
                key=f"save_local_recovery:{recovery_context_key}:{selected_issue_key}",
                use_container_width=True,
            )
        with reset_col:
            reset_local = st.button(
                "Reset region",
                key=f"reset_local_recovery:{recovery_context_key}:{selected_issue_key}",
                use_container_width=True,
            )

        if save_local:
            if selected_preview_candidate_idx is None:
                st.sidebar.warning(
                    "This local order does not produce a recovery candidate."
                )
            else:
                saved_recovery_orders[selected_issue_key] = int(recovery_order)
                bulk_flat_saved_candidates.pop(selected_issue_key, None)
                st.rerun()

        if reset_local:
            saved_recovery_orders.pop(selected_issue_key, None)
            bulk_flat_saved_candidates.pop(selected_issue_key, None)
            st.session_state.pop(slider_key, None)
            st.rerun()

        if saved_recovery_orders and st.sidebar.button(
            "Clear all saved local recoveries",
            key=f"clear_local_recoveries:{recovery_context_key}",
            use_container_width=True,
        ):
            saved_recovery_orders.clear()
            bulk_flat_saved_candidates.clear()
            st.rerun()

    # Apply every saved local override independently. Bulk-flat saves use their
    # exact stored candidate so a different sharp point cannot replace them.
    for issue in suspect_regions:
        issue_key = _recovery_issue_key(issue)
        if issue_key not in saved_recovery_orders:
            continue

        missing_type = str(issue["Missing Type"])
        if issue_key in bulk_flat_saved_candidates:
            recovered_idx = int(bulk_flat_saved_candidates[issue_key])
            if missing_type == "min":
                min_indices = np.unique(
                    np.append(min_indices, recovered_idx)
                ).astype(int)
                recovered_min_indices.append(recovered_idx)
            elif missing_type == "max":
                max_indices = np.unique(
                    np.append(max_indices, recovered_idx)
                ).astype(int)
                recovered_max_indices.append(recovered_idx)
            continue

        saved_order = int(saved_recovery_orders[issue_key])
        recovery = recover_missing_extremum(
            thickness_values=thickness_values,
            min_indices=min_indices,
            max_indices=max_indices,
            issue=issue,
            recovery_order=saved_order,
        )
        min_indices = recovery.min_indices
        max_indices = recovery.max_indices
        recovered_min_indices.extend(recovery.recovered_min_indices)
        recovered_max_indices.extend(recovery.recovered_max_indices)

    recovered_min_indices = sorted(set(recovered_min_indices))
    recovered_max_indices = sorted(set(recovered_max_indices))

events = build_events(min_indices, max_indices, time_values, thickness_values)

calculate_kwargs = {
    "events": events,
    "time_values": time_values,
    "thickness_values": thickness_values,
    "recovered_min_indices": recovered_min_indices,
    "recovered_max_indices": recovered_max_indices,
}
cycle_parameters = inspect.signature(calculate_cycles).parameters

if "analysis_type" in cycle_parameters:
    calculate_kwargs["analysis_type"] = "3-step"
if "transition_smoothing_window" in cycle_parameters:
    calculate_kwargs["transition_smoothing_window"] = transition_smoothing_window
if "transition_onset_fraction" in cycle_parameters:
    calculate_kwargs["transition_onset_fraction"] = transition_onset_fraction
if "transition_polyorder" in cycle_parameters:
    calculate_kwargs["transition_polyorder"] = 2
if (
    "transition_min_width" in cycle_parameters
    and "transition_onset_fraction" not in cycle_parameters
):
    calculate_kwargs["transition_min_width"] = 2

result = calculate_cycles(**calculate_kwargs)
cycle_df = result.cycle_df

point_b_indices = getattr(result, "point_b_indices", [])
point_c_indices = getattr(result, "point_c_indices", result.transition_indices)
recovered_point_b_indices = getattr(result, "recovered_point_b_indices", [])
recovered_point_c_indices = getattr(
    result, "recovered_point_c_indices", result.recovered_transition_indices
)

st.subheader("ALD/ALE Process Cycle Detection")
plot_cycle_analysis(
    time_values,
    thickness_values,
    min_indices,
    max_indices,
    recovered_min_indices,
    recovered_max_indices,
    point_b_indices,
    point_c_indices,
    recovered_point_b_indices,
    recovered_point_c_indices,
    time_col,
    thickness_col,
    selected_issue,
)

if use_recovery and selected_issue is not None:
    st.subheader("Inspect Missing-Point Region")
    selected_is_bulk_flat = (
        _recovery_issue_key(selected_issue) in bulk_flat_saved_candidates
    )
    source_text = (
        " This point was saved by the bulk flat-recovery scan."
        if selected_is_bulk_flat
        else ""
    )
    st.caption(
        f"Approximate cycle {selected_issue_approx_cycle}: possible missing "
        f"{str(selected_issue['Missing Type']).upper()}. The preview marker follows "
        "the current local-order slider; only the saved marker is used in analysis."
        + source_text
    )
    missing_padding = st.slider(
        "Missing-region padding (samples)",
        min_value=0,
        max_value=50,
        value=8,
        step=1,
        key=f"missing_region_padding:{recovery_context_key}",
    )
    plot_missing_point_region(
        issue=selected_issue,
        time_values=time_values,
        thickness_values=thickness_values,
        base_min_indices=base_min_indices,
        base_max_indices=base_max_indices,
        preview_candidate_idx=selected_preview_candidate_idx,
        saved_candidate_idx=selected_saved_candidate_idx,
        time_col=time_col,
        thickness_col=thickness_col,
        padding_points=missing_padding,
    )

    if selected_preview_candidate_idx is None:
        st.warning(
            "The current local order does not produce a candidate in this region."
        )
    elif selected_issue_saved_order is None:
        st.info(
            "A candidate is previewed above. Save/apply the local order to include "
            "that recovered point in cycle detection."
        )

    if selected_saved_candidate_idx is not None:
        missing_type = str(selected_issue["Missing Type"])
        if missing_type == "max":
            recovered_cycle_rows = cycle_df.loc[
                cycle_df["Max Anchor Index"] == selected_saved_candidate_idx
            ]
        else:
            recovered_cycle_rows = cycle_df.loc[
                (cycle_df["Point A Index"] == selected_saved_candidate_idx)
                | (cycle_df["Point D Index"] == selected_saved_candidate_idx)
            ]

        if not recovered_cycle_rows.empty:
            recovered_cycle_row = recovered_cycle_rows.iloc[0]
            recovered_cycle_number = int(recovered_cycle_row["Cycle"])
            st.success(
                f"This saved recovery contributes to accepted Cycle "
                f"{recovered_cycle_number}."
            )
            plot_selected_cycle(
                cycle_row=recovered_cycle_row,
                time_values=time_values,
                thickness_values=thickness_values,
                time_col=time_col,
                thickness_col=thickness_col,
                padding_points=5,
            )
        else:
            st.caption(
                "The recovered extremum is saved, but this region does not yet form "
                "an accepted A/B/M/C/D cycle under the current transition settings."
            )

st.subheader("Analysis Diagnostics")
row1 = st.columns(4)
row1[0].metric("Full data points", len(full_df))
row1[1].metric("Analyzed data points", len(analysis_df))
row1[2].metric("Detected extrema", len(events))
row1[3].metric("Successful cycles", len(cycle_df))

row2 = st.columns(4)
row2[0].metric("Detected minima", len(min_indices))
row2[1].metric("Maximum anchors", len(max_indices))
row2[2].metric("Rejected sequences", result.rejected_sequences)
row2[3].metric("Transition failures", result.derivative_failures)

if use_recovery:
    active_saved_keys = {
        _recovery_issue_key(issue) for issue in suspect_regions
    } & set(saved_recovery_orders)
    active_bulk_keys = active_saved_keys & set(bulk_flat_saved_candidates)
    st.caption(
        f"Applied local recoveries: {len(active_saved_keys)} region(s) "
        f"({len(active_bulk_keys)} bulk-flat); recovered minima: "
        f"{len(recovered_min_indices)}, recovered maxima: "
        f"{len(recovered_max_indices)}."
    )

if len(cycle_df) > 0:
    st.success(
        f"{len(cycle_df)} complete cycles with purge/fall boundaries were identified."
    )
else:
    st.warning(
        "No complete cycles with valid purge/fall boundaries were detected. "
        "Adjust the analysis window, extrema orders, local recoveries, plateau "
        "threshold, or smoothing."
    )

st.subheader("Δ1, Δ2, and Δ3 by cycle")
if cycle_df.empty:
    st.write("No complete ALD/ALE cycles are available in the selected analysis window.")
else:
    st.dataframe(cycle_df, use_container_width=True, hide_index=True)

    st.subheader("Inspect Individual Cycle")
    selector_col, padding_col = st.columns([2, 1])
    cycle_numbers = cycle_df["Cycle"].astype(int).tolist()

    with selector_col:
        selected_cycle_number = st.selectbox(
            "Cycle",
            cycle_numbers,
            format_func=lambda value: f"Cycle {value}",
            key="individual_cycle_selector",
        )
    with padding_col:
        padding_points = st.slider(
            "Padding (samples)",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            help="Extra samples shown before A and after D for visual context.",
            key="individual_cycle_padding",
        )

    selected_cycle_row = cycle_df.loc[
        cycle_df["Cycle"].astype(int) == int(selected_cycle_number)
    ].iloc[0]

    plot_selected_cycle(
        cycle_row=selected_cycle_row,
        time_values=time_values,
        thickness_values=thickness_values,
        time_col=time_col,
        thickness_col=thickness_col,
        padding_points=padding_points,
    )

    point_summary = pd.DataFrame(
        [
            {
                "Point": "A",
                "Index": int(selected_cycle_row["Point A Index"]),
                "Time": float(selected_cycle_row["Point A Time"]),
                "Thickness": float(selected_cycle_row["Point A Thickness"]),
            },
            {
                "Point": "B",
                "Index": int(selected_cycle_row["Point B Index"]),
                "Time": float(selected_cycle_row["Point B Time"]),
                "Thickness": float(selected_cycle_row["Point B Thickness"]),
            },
            {
                "Point": "M (max anchor)",
                "Index": int(selected_cycle_row["Max Anchor Index"]),
                "Time": float(selected_cycle_row["Max Anchor Time"]),
                "Thickness": float(selected_cycle_row["Max Anchor Thickness"]),
            },
            {
                "Point": "C",
                "Index": int(selected_cycle_row["Point C Index"]),
                "Time": float(selected_cycle_row["Point C Time"]),
                "Thickness": float(selected_cycle_row["Point C Thickness"]),
            },
            {
                "Point": "D",
                "Index": int(selected_cycle_row["Point D Index"]),
                "Time": float(selected_cycle_row["Point D Time"]),
                "Thickness": float(selected_cycle_row["Point D Thickness"]),
            },
        ]
    )
    st.dataframe(point_summary, use_container_width=True, hide_index=True)

    st.subheader("Average Δ Values")
    cols = st.columns(3)
    cols[0].metric("Average Δ1", f"{cycle_df['Delta 1'].mean():.4f}")
    cols[1].metric("Average Δ2", f"{cycle_df['Delta 2'].mean():.4f}")
    cols[2].metric("Average Δ3", f"{cycle_df['Delta 3'].mean():.4f}")

    csv = cycle_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download delta results as CSV",
        data=csv,
        file_name="cycle_delta_results.csv",
        mime="text/csv",
    )
