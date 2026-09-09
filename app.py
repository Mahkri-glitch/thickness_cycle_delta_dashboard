"""Streamlit user interface for cyclic ALD/ALE thickness delta analysis."""

from __future__ import annotations

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
            label="Suspected missing-point region",
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
            label="Point C (purge plateau exit)",
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
            label="Detected purge plateau",
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
- **Point C** = exit from the low-slope purge/plateau before the active fall
- **Point D** = next minimum
- **Δ1 = B − A**
- **Δ2 = B − C**
- **Δ3 = C − D**

A valid cycle follows **A < B ≤ maximum ≤ C < D**. The maximum is not forced to
be a separate process point: it may equal B, C, or both if that is what the
sampled trace resolves.

**Transition detection**  
The program lightly smooths A → D, calculates the slope between adjacent samples,
and uses the active rise before the maximum and active fall after it as reference
rates. Starting at the maximum, it expands left and right through the contiguous
**low-slope region**. Those two edges are B and C.

This is meant to capture the purge step even when it has a slight positive or
negative drift instead of a perfectly flat plateau.

**Smoothing window**  
The minimum is **3 samples** so short ellipsometry transitions are not
unnecessarily smeared.

**Purge plateau threshold**  
This is the largest slope magnitude that can still count as part of the purge
plateau, expressed as a percentage of the active rise/fall rate. Lower values
require a flatter purge. Higher values allow more gradual drift and produce a
wider B → C region.

**Individual cycle inspector**  
For long datasets, select any accepted cycle below the main plot to see a zoomed
view with the exact **A, B, maximum anchor, C, and D** used for the calculations.
The inspector does not re-run or change the detector; it only visualizes the
stored cycle result.

**Time direction**  
Uploaded data are automatically sorted by the selected time column before
analysis.

**Ellipsometer Excel headers**  
For Excel files, the dashboard scans the first 30 rows for separate time and
thickness header cells. A title such as **Thickness vs Time** in one cell is
ignored.

**Missing-point recovery**  
MAX → MAX suggests a missing minimum; MIN → MIN suggests a missing maximum.
Recovery searches locally and adds one candidate without deleting existing
extrema.
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
        "positive/negative drift and widen the B-to-C plateau region."
    ),
)
transition_onset_fraction = plateau_percent / 100.0

min_indices, max_indices = detect_extrema(thickness_values, min_order, max_order)
primary_events = build_events(min_indices, max_indices, time_values, thickness_values)
suspect_regions = find_suspect_regions(primary_events)

recovered_min_indices: list[int] = []
recovered_max_indices: list[int] = []
selected_issue: dict | None = None

st.sidebar.header("Missing Point Recovery")
use_recovery = st.sidebar.checkbox("Enable missing-point recovery", value=False)

if use_recovery:
    if not suspect_regions:
        st.sidebar.caption("No broken MIN/MAX alternation was detected.")
    else:
        issue_number = st.sidebar.selectbox(
            "Suspected missing point",
            list(range(len(suspect_regions))),
            format_func=lambda idx: (
                f"Issue {idx + 1}: missing {suspect_regions[idx]['Missing Type'].upper()} "
                f"between t={suspect_regions[idx]['Start Time']:.3f} and "
                f"t={suspect_regions[idx]['End Time']:.3f}"
            ),
        )
        selected_issue = suspect_regions[issue_number]
        missing_type = selected_issue["Missing Type"]
        recovery_order = st.sidebar.slider(
            f"Local {missing_type} order",
            min_value=1,
            max_value=max_allowed_order,
            value=min_order if missing_type == "min" else max_order,
            step=1,
        )
        recovery = recover_missing_extremum(
            thickness_values=thickness_values,
            min_indices=min_indices,
            max_indices=max_indices,
            issue=selected_issue,
            recovery_order=recovery_order,
        )
        min_indices = recovery.min_indices
        max_indices = recovery.max_indices
        recovered_min_indices = recovery.recovered_min_indices
        recovered_max_indices = recovery.recovered_max_indices

        if not recovered_min_indices and not recovered_max_indices:
            st.sidebar.warning("No local candidate was found in this gap.")

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
row2[3].metric("Plateau failures", result.derivative_failures)

if len(cycle_df) > 0:
    st.success(
        f"{len(cycle_df)} complete cycles with purge plateau boundaries were identified."
    )
else:
    st.warning(
        "No complete cycles with valid purge plateau boundaries were detected. "
        "Adjust the analysis window, extrema orders, plateau threshold, or smoothing."
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
