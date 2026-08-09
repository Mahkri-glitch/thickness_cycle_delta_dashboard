"""Streamlit user interface for cyclic thickness delta analysis."""

from __future__ import annotations

import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analysis import (
    build_events,
    calculate_cycles,
    detect_extrema,
    downsample_for_plot,
    find_suspect_regions,
    recover_missing_extremum,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
# Cached file readers
# -----------------------------------------------------------------------------
@st.cache_data
def get_excel_sheets(file_bytes: bytes) -> list[str]:
    """Return Excel sheet names without re-reading the file on every rerun."""
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    return excel_file.sheet_names


@st.cache_data
def load_excel(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def prepare_numeric_data(df: pd.DataFrame, time_col: str, thickness_col: str) -> pd.DataFrame:
    """Keep the two analysis columns, coerce to numeric, and drop invalid rows."""
    cleaned = df[[time_col, thickness_col]].copy()
    cleaned[time_col] = pd.to_numeric(cleaned[time_col], errors="coerce")
    cleaned[thickness_col] = pd.to_numeric(cleaned[thickness_col], errors="coerce")
    return cleaned.dropna().reset_index(drop=True)


def plot_full_dataset(
    df: pd.DataFrame,
    time_col: str,
    thickness_col: str,
    start_time: float,
    end_time: float,
) -> None:
    """Render a lightweight overview with the selected analysis-window boundaries."""
    x = df[time_col].to_numpy(dtype=float)
    y = df[thickness_col].to_numpy(dtype=float)
    plot_x, plot_y = downsample_for_plot(x, y, max_points=5000)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(plot_x, plot_y, linewidth=2, alpha=0.7)

    # Show the current analysis-window boundaries on the full trace.
    ax.axvline(
        x=start_time,
        linestyle="--",
        linewidth=2,
    )
    ax.axvline(
        x=end_time,
        linestyle="--",
        linewidth=2,
    )

    ax.set_xlabel(time_col)
    ax.set_ylabel(thickness_col)
    st.pyplot(fig)
    plt.close(fig)


def plot_cycle_analysis(
    time_values: np.ndarray,
    thickness_values: np.ndarray,
    min_indices: np.ndarray,
    max_indices: np.ndarray,
    recovered_min_indices: list[int],
    recovered_max_indices: list[int],
    transition_indices: list[int],
    recovered_transition_indices: list[int],
    analysis_type: str,
    time_col: str,
    thickness_col: str,
    selected_issue: dict | None,
) -> None:
    """Plot extrema and transitions while keeping the line rendering efficient."""
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

    if analysis_type == "3-step":
        if transition_indices:
            ax.scatter(
                time_values[transition_indices],
                thickness_values[transition_indices],
                s=100,
                marker="^",
                label="Point C (transition)",
            )

        if recovered_transition_indices:
            ax.scatter(
                time_values[recovered_transition_indices],
                thickness_values[recovered_transition_indices],
                s=160,
                marker="*",
                label="Point C from recovered cycle",
            )

        ax.set_title("3-Step Delta Analyzer")
    else:
        ax.set_title("2-Step Delta Analyzer")

    ax.set_xlabel(time_col)
    ax.set_ylabel(thickness_col)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def show_interpretation_guide() -> None:
    """Keep the scientific meaning close to the controls that generate it."""
    with st.expander("How to interpret the analysis"):
        st.markdown(
            """
**Complete-cycle rule**  
Only successive **minimum → maximum → minimum** sequences are included in the
results. Incomplete or nonmatching extrema sequences do not contribute to the
reported averages.

**2-step mode**
- **Point A** = first minimum
- **Point B** = maximum
- **Point D** = next minimum
- **Δ1 = B − A**
- **Δ2 = B − D**
- Point C and Δ3 are intentionally blank.

**3-step mode**
- **Point A** = first minimum
- **Point B** = maximum
- **Point C** = transition immediately before the stronger final drop
- **Point D** = next minimum
- **Δ1 = B − A**
- **Δ2 = B − C**
- **Δ3 = C − D**

Point C is found from the strongest negative change in slope within the B → D
segment. It should be visually checked because a noisy trace can occasionally
produce a stronger derivative change than the physically intended transition.

**Extrema `order`**  
The minimum and maximum order values control how much neighboring data a point
must beat to count as a local extremum. Larger values suppress small features;
smaller values make detection more sensitive to local structure and noise.

**Missing-point recovery**  
A **MAX → MAX** sequence is treated as a possible missing minimum, while a
**MIN → MIN** sequence is treated as a possible missing maximum. Recovery is
additive: it searches locally and adds one candidate without deleting the
existing global extrema.
            """
        )


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Thickness Cycle Delta Analyzer", layout="wide")
st.title("Thickness Cycle Delta Analyzer")
st.write(
    "Analyze cyclic thickness-vs-time data with local-extrema detection, "
    "strict complete-cycle validation, optional missing-point recovery, and "
    "2-step or 3-step delta calculations."
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

# -----------------------------------------------------------------------------
# Analysis window and controls
# -----------------------------------------------------------------------------
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

# Use the selected endpoints to mark the active analysis window on the overview.
start_time = float(full_df.iloc[window_start][time_col])
end_time = float(full_df.iloc[window_end][time_col])

st.subheader("Select Analysis Window")
plot_full_dataset(
    full_df,
    time_col,
    thickness_col,
    start_time=start_time,
    end_time=end_time,
)

max_allowed_order = max(1, min(250, (len(full_df) - 1) // 2))
default_order = min(10, max_allowed_order)

st.sidebar.header("Analysis type")
analysis_type = st.sidebar.radio(
    "Cycle analysis", ["2-step", "3-step"], index=1
)

if analysis_type == "2-step":
    st.sidebar.caption("Δ1 = B − A, Δ2 = B − D")
else:
    st.sidebar.caption("Δ1 = B − A, Δ2 = B − C, Δ3 = C − D")

st.sidebar.header("Extrema filters")
min_order = st.sidebar.slider(
    "Minimum filter order",
    min_value=1,
    max_value=max_allowed_order,
    value=default_order,
    step=1,
    key="min_order",
)
max_order = st.sidebar.slider(
    "Maximum filter order",
    min_value=1,
    max_value=max_allowed_order,
    value=default_order,
    step=1,
    key="max_order",
)

window_max_order = max(1, (len(analysis_df) - 1) // 2)
if min_order > window_max_order or max_order > window_max_order:
    st.sidebar.warning(
        "The selected order is large relative to the current analysis window. "
        "Consider reducing the order or expanding the window."
    )

# -----------------------------------------------------------------------------
# Global extrema detection and optional local recovery
# -----------------------------------------------------------------------------
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
                f"Issue {idx + 1}: missing "
                f"{suspect_regions[idx]['Missing Type'].upper()} between "
                f"t={suspect_regions[idx]['Start Time']:.3f} and "
                f"t={suspect_regions[idx]['End Time']:.3f}"
            ),
        )
        selected_issue = suspect_regions[issue_number]
        missing_type = selected_issue["Missing Type"]
        st.sidebar.info(f"Likely missing {missing_type.upper()}")

        recovery_order = st.sidebar.slider(
            f"Local {missing_type} order",
            min_value=1,
            max_value=max_allowed_order,
            value=min_order if missing_type == "min" else max_order,
            step=1,
            key=f"recovery_order_{issue_number}_{missing_type}",
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
            st.sidebar.warning(
                "No local candidate was found in this gap at the selected order."
            )

# Rebuild the final event sequence only after recovery has been applied.
events = build_events(min_indices, max_indices, time_values, thickness_values)

# -----------------------------------------------------------------------------
# Cycle calculations
# -----------------------------------------------------------------------------
result = calculate_cycles(
    events=events,
    time_values=time_values,
    thickness_values=thickness_values,
    analysis_type=analysis_type,
    recovered_min_indices=recovered_min_indices,
    recovered_max_indices=recovered_max_indices,
)
cycle_df = result.cycle_df

st.subheader(f"{analysis_type} Cycle Detection")
plot_cycle_analysis(
    time_values=time_values,
    thickness_values=thickness_values,
    min_indices=min_indices,
    max_indices=max_indices,
    recovered_min_indices=recovered_min_indices,
    recovered_max_indices=recovered_max_indices,
    transition_indices=result.transition_indices,
    recovered_transition_indices=result.recovered_transition_indices,
    analysis_type=analysis_type,
    time_col=time_col,
    thickness_col=thickness_col,
    selected_issue=selected_issue,
)

# -----------------------------------------------------------------------------
# Diagnostics and results
# -----------------------------------------------------------------------------
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
    included_deltas = "Δ1 and Δ2" if analysis_type == "2-step" else "Δ1, Δ2, and Δ3"
    st.success(
        f"{len(cycle_df)} complete minimum → maximum → minimum cycles were identified. "
        f"Only these complete cycles are included in the {included_deltas} calculations."
    )
else:
    st.warning(
        "No complete minimum → maximum → minimum cycles were detected. "
        "Adjust the analysis window and/or extrema filter order."
    )

st.subheader("Δ1 and Δ2 by cycle" if analysis_type == "2-step" else "Δ1, Δ2, and Δ3 by cycle")

if cycle_df.empty:
    st.write("No complete cycles are available in the selected analysis window.")
else:
    st.dataframe(cycle_df, use_container_width=True, hide_index=True)

    st.subheader("Average Δ Values")
    if analysis_type == "2-step":
        cols = st.columns(2)
        cols[0].metric("Average Δ1", f"{cycle_df['Delta 1'].mean():.4f}")
        cols[1].metric("Average Δ2", f"{cycle_df['Delta 2'].mean():.4f}")
    else:
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
