"""Streamlit user interface for cyclic ALD/ALE thickness delta analysis."""

from __future__ import annotations

import importlib
import inspect
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit can rerun app.py without fully re-importing local modules. Reload the
# analysis module explicitly so a newly deployed analysis.py cannot leave this
# app calling an older calculate_cycles() signature from memory.
import analysis as analysis_core
analysis_core = importlib.reload(analysis_core)

build_events = analysis_core.build_events
calculate_cycles = analysis_core.calculate_cycles
detect_extrema = analysis_core.detect_extrema
downsample_for_plot = analysis_core.downsample_for_plot
find_suspect_regions = analysis_core.find_suspect_regions
recover_missing_extremum = analysis_core.recover_missing_extremum

warnings.simplefilter(action="ignore", category=FutureWarning)


@st.cache_data
def get_excel_sheets(file_bytes: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


@st.cache_data
def load_excel(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def prepare_numeric_data(df: pd.DataFrame, time_col: str, thickness_col: str) -> pd.DataFrame:
    cleaned = df[[time_col, thickness_col]].copy()
    cleaned[time_col] = pd.to_numeric(cleaned[time_col], errors="coerce")
    cleaned[thickness_col] = pd.to_numeric(cleaned[thickness_col], errors="coerce")
    return cleaned.dropna().reset_index(drop=True)


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
    transition_indices,
    recovered_transition_indices,
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
        label="Point B candidate (maximum)",
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
            label="Recovered maximum",
        )
    if transition_indices:
        ax.scatter(
            time_values[transition_indices],
            thickness_values[transition_indices],
            s=100,
            marker="^",
            label="Point C (ALD/ALE transition)",
        )
    if recovered_transition_indices:
        ax.scatter(
            time_values[recovered_transition_indices],
            thickness_values[recovered_transition_indices],
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


def show_interpretation_guide() -> None:
    with st.expander("How to interpret the analysis"):
        st.markdown(
            """
**Complete-cycle rule**  
Only successive **minimum → maximum → minimum** sequences are included.

**ALD/ALE Process**
- **Point A** = first minimum
- **Point B** = maximum
- **Point C** = broad transition into the final ALE-related drop
- **Point D** = next minimum
- **Δ1 = B − A**
- **Δ2 = B − C**
- **Δ3 = C − D**

**Point C transition detection**  
The B → D segment is smoothed with a low-order Savitzky-Golay polynomial before
first and second derivatives are calculated. Point C is selected from a broad
negative-curvature feature. The minimum-width filter rejects very narrow
curvature spikes caused by abrupt point-to-point changes.

The polynomial order is intentionally limited to **2 or 3**. Increase the
minimum transition width if a sharp C → D step is still being selected instead
of the broader transition.

**Extrema order**  
The minimum and maximum filter orders control how many neighboring points a
candidate must beat to count as a local extremum. These are independent of the
Point C smoothing controls.

**Missing-point recovery**  
MAX → MAX suggests a missing minimum; MIN → MIN suggests a missing maximum.
Recovery searches locally and adds one candidate without deleting existing
extrema.
            """
        )


st.set_page_config(page_title="Thickness Cycle Delta Analyzer", layout="wide")
st.title("Thickness Cycle Delta Analyzer")
st.write(
    "Analyze cyclic thickness-vs-time data for the ALD/ALE process with "
    "local-extrema detection and Δ1/Δ2/Δ3 calculations."
)
show_interpretation_guide()

uploaded_file = st.file_uploader(
    "Upload a thickness-vs-time file", type=["xlsx", "xls", "csv"]
)
if uploaded_file is None:
    st.info("Upload an Excel or CSV file to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()
if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
    sheet_name = st.selectbox("Sheet", get_excel_sheets(file_bytes))
    raw_df = load_excel(file_bytes, sheet_name)
else:
    raw_df = load_csv(file_bytes)

if len(raw_df.columns) < 2:
    st.error("The file needs at least two columns.")
    st.stop()

columns = list(raw_df.columns)
default_time = columns.index("Time") if "Time" in columns else 0
default_thickness = (
    columns.index("Thickness") if "Thickness" in columns else min(1, len(columns) - 1)
)

left, right = st.columns(2)
with left:
    time_col = st.selectbox("Time column", columns, index=default_time)
with right:
    thickness_col = st.selectbox("Thickness column", columns, index=default_thickness)

if time_col == thickness_col:
    st.error("Choose different columns for time and thickness.")
    st.stop()

full_df = prepare_numeric_data(raw_df, time_col, thickness_col)
if len(full_df) < 3:
    st.error("Not enough numeric data points were found after cleaning.")
    st.stop()

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

st.sidebar.header("Point C transition filter")
max_transition_window = min(101, max(5, len(analysis_df)))
if max_transition_window % 2 == 0:
    max_transition_window -= 1
max_transition_window = max(5, max_transition_window)
default_transition_window = min(11, max_transition_window)

transition_smoothing_window = st.sidebar.slider(
    "Smoothing window (samples)",
    min_value=5,
    max_value=max_transition_window,
    value=default_transition_window,
    step=2,
)
transition_polyorder = st.sidebar.select_slider(
    "Polynomial order", options=[2, 3], value=2
)
transition_min_width = st.sidebar.slider(
    "Minimum transition width (samples)",
    min_value=2,
    max_value=20,
    value=5,
    step=1,
    help="Increase this to reject narrow, steep C→D curvature spikes.",
)

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

# Build the call from the function signature. This makes a Streamlit process
# tolerant of the previous 3-step calculate_cycles signature during a hot reload,
# while using all new ALD/ALE transition controls whenever the current analysis.py
# is loaded.
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
if "transition_polyorder" in cycle_parameters:
    calculate_kwargs["transition_polyorder"] = transition_polyorder
if "transition_min_width" in cycle_parameters:
    calculate_kwargs["transition_min_width"] = transition_min_width

result = calculate_cycles(**calculate_kwargs)
cycle_df = result.cycle_df

st.subheader("ALD/ALE Process Cycle Detection")
plot_cycle_analysis(
    time_values,
    thickness_values,
    min_indices,
    max_indices,
    recovered_min_indices,
    recovered_max_indices,
    result.transition_indices,
    result.recovered_transition_indices,
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
row2[1].metric("Detected maxima", len(max_indices))
row2[2].metric("Rejected sequences", result.rejected_sequences)
row2[3].metric("Transition failures", result.derivative_failures)

if len(cycle_df) > 0:
    st.success(
        f"{len(cycle_df)} complete minimum → maximum → minimum cycles were identified. "
        "Only cycles with a valid Point C transition are included in Δ1/Δ2/Δ3."
    )
else:
    st.warning(
        "No complete cycles with a valid Point C transition were detected. Adjust "
        "the analysis window, extrema orders, smoothing window, or transition width."
    )

st.subheader("Δ1, Δ2, and Δ3 by cycle")
if cycle_df.empty:
    st.write("No complete ALD/ALE cycles are available in the selected analysis window.")
else:
    st.dataframe(cycle_df, use_container_width=True, hide_index=True)
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
