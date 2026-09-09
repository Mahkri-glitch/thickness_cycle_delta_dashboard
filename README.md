# Thickness Cycle Delta Analyzer

A Streamlit dashboard for analyzing cyclic thickness-vs-time data for an **ALD/ALE process**. The program detects repeated extrema, uses each detected maximum as an internal anchor, identifies purge/fall transition boundaries, and reports Δ1, Δ2, and Δ3 for each valid cycle.

The dashboard first identifies a **minimum → maximum → minimum** sequence in **forward physical time**.

## ALD/ALE process definition

```text
A = first minimum
B = entry into the low-slope purge/plateau after the active rise
M = detected maximum anchor inside or at an edge of the purge region
C = onset of the active fall after purge
D = next minimum

Required ordering: A < B <= M <= C < D

Δ1 = B - A
Δ2 = B - C
Δ3 = C - D
```

The maximum is a reference point, not a forced process boundary. It may equal B, C, or both if that is what the sampled trace resolves.

## Purge and Point C detection

For each A → maximum → D cycle the detector:

1. lightly smooths the local thickness trace,
2. calculates the slope between adjacent measurements,
3. uses the active positive rise before the maximum as a reference rate,
4. uses the active negative fall after the maximum as a reference rate,
5. finds B from the low-slope purge region after the active rise, and
6. finds C as either the first sustained two-interval fall or one clearly instantaneous single drop.

The single-drop C path is used only when the drop reaches the representative active-fall rate and is not immediately reversed by a comparably strong rebound. This allows one-sample reaction events while still ignoring ordinary isolated downward excursions during purge.

## Controls

- **Smoothing window (samples)** — default **3**. Smaller smoothing helps preserve short ellipsometry transition events.
- **Purge plateau threshold (% of active slope)** — default **35%**. Lower values require a flatter purge region; higher values allow more positive/negative drift.

The Savitzky-Golay polynomial order is fixed internally at 2 in the dashboard.

## Individual cycle inspector

For long processes, the dashboard includes a second **Inspect Individual Cycle** graph below the main cycle plot.

- Select any accepted cycle by cycle number.
- The graph zooms to that cycle only, with optional padding before A and after D.
- The exact stored **A, B, maximum anchor M, C, and D** are marked and labeled.
- The detected B → C purge region is shaded when it has nonzero width.
- A small table lists each selected point's analysis index, time, and thickness.

The inspector does not re-run the detector. It visualizes the exact points already used to calculate Δ1, Δ2, and Δ3.

## Per-region missing-point recovery

The dashboard flags broken extrema alternation patterns:

```text
MAX → MAX  = possible missing MIN
MIN → MIN  = possible missing MAX
```

Missing-point recovery is now **localized per suspect region** rather than controlled by one recovery order for the entire dataset.

For each suspect region you can:

1. select the approximate broken cycle from the **Missing-point region** selector,
2. change a **local minimum or maximum order** for only that region,
3. preview the candidate on a zoomed **Inspect Missing-Point Region** graph,
4. choose **Save/apply** to keep that local order,
5. move to another suspect region and tune it independently, and
6. use **Reset region** or **Clear all saved local recoveries** when needed.

Saved local orders are stored in Streamlit session state and scoped to the uploaded file, selected analysis window, and selected time/thickness columns. They are reapplied independently during the current Streamlit session and do **not** change the global extrema orders.

Only saved recoveries affect cycle detection. Moving a local-order slider without saving changes the preview only.

When a saved recovered extremum successfully rebuilds an accepted cycle, the missing-point inspector reports the resulting cycle number and immediately shows that cycle's full **A/B/M/C/D** inspection graph.

## Time direction

The uploaded file may be ordered with time increasing or decreasing. The dashboard automatically sorts the selected time column into **ascending chronological order before any extrema or transition calculations**.

Reported point indices refer to the **chronologically sorted analysis window**, not necessarily the original Excel row number. Point times are the recommended reference when comparing datasets.

## Ellipsometer Excel headers

For Excel files, the dashboard scans the first 30 rows for separate time and thickness header cells. A title such as `Thickness vs Time` in one cell is ignored, allowing direct ellipsometer exports where the real column headers appear on a later row.

## Extrema detection

The global extrema anchor sequence is:

```text
minimum → maximum → minimum
```

SciPy relative-extrema `order` controls how many neighboring samples a candidate must beat to count as a local minimum or maximum:

- smaller order = more sensitive to local structure/noise,
- larger order = more selective.

Global minimum and maximum orders establish the baseline event sequence. Per-region recovery overrides are layered on afterward.

## Input

Supported files: `.csv`, `.xlsx`, `.xls`.

## Run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Recommended workflow

1. Upload the dataset and select time/thickness columns.
2. Select the analysis window.
3. Tune global minimum and maximum extrema orders until most cycles follow the physical process.
4. Start with smoothing = 3 and purge plateau threshold = 35%.
5. Enable missing-point recovery and tune only the individual broken regions that remain.
6. Save each useful local recovery order; do not lower the global order just to repair one cycle.
7. Inspect the recovered cycle directly in the missing-point view.
8. Use **Inspect Individual Cycle** to verify A/B/M/C/D on representative cycles across the run.
9. Download the CSV when satisfied.

## Output columns

```text
Cycle,
Point A Index, Point A Time, Point A Thickness,
Point B Index, Point B Time, Point B Thickness,
Max Anchor Index, Max Anchor Time, Max Anchor Thickness,
Point C Index, Point C Time, Point C Thickness,
Point D Index, Point D Time, Point D Thickness,
Delta 1, Delta 2, Delta 3
```

The maximum anchor is stored in the cycle results so the individual-cycle inspector always shows the exact maximum used for that cycle.

## Testing

The analysis test suite verifies purge-entry behavior, sustained and instantaneous Point C behavior, slight purge drift, maximum-as-transition behavior, stored maximum-anchor metadata, and Δ1/Δ2/Δ3 arithmetic.

Run:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```
