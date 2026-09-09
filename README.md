# Thickness Cycle Delta Analyzer

A Streamlit dashboard for analyzing cyclic thickness-vs-time data for an **ALD/ALE process**. The program detects repeated extrema, uses each detected maximum as an internal anchor, finds two transition points around that maximum, and reports Δ1, Δ2, and Δ3 for each valid cycle.

The dashboard first identifies a strict **minimum → maximum → minimum** sequence in **forward physical time**.

## ALD/ALE process definition

```text
A = first minimum
B = rising-side transition between A and the maximum anchor
M = detected maximum anchor (not exported as B or C)
C = falling-side transition between the maximum anchor and D
D = next minimum

Required ordering: A < B < M < C < D

Δ1 = B - A
Δ2 = B - C
Δ3 = C - D
```

The maximum is used only to split the cycle into two transition-search regions. It can never be returned as Point B or Point C.

If either B or C is not distinctly resolved, the cycle is rejected rather than forcing a transition point.

## Transition detection

Each side of the maximum anchor is analyzed independently:

1. lightly smooth the local thickness trace,
2. calculate `dh/dt`,
3. on **A → maximum**, find the onset of the strongest rising regime for Point B,
4. on **maximum → D**, find the onset of the strongest falling regime for Point C,
5. require the selected transition to be strictly inside its search interval, and
6. reject the cycle if either transition cannot be resolved.

This prevents the detected maximum from being mislabeled as a process transition.

## Transition controls

- **Smoothing window (samples)** — default **3**. Smaller smoothing helps preserve short ellipsometry transition events.
- **Transition threshold (% of slope change)** — default **35%**. Lower values detect an earlier onset; higher values place B/C closer to the strongest slope.
- **Transition persistence (samples)** — default **2**. The transition regime must remain connected for at least this many samples.

The Savitzky-Golay polynomial order is fixed internally at 2 in the dashboard.

## Time direction

The uploaded file may be ordered with time increasing or decreasing. The dashboard automatically sorts the selected time column into **ascending chronological order before any extrema or transition calculations**.

If the file is detected as descending in time, the dashboard displays a notice. If the time values are non-monotonic, it displays a warning and sorts them before analysis.

Reported point indices refer to the **chronologically sorted analysis window**, not necessarily the original Excel row number. Point times are the recommended reference when comparing datasets.

## Ellipsometer Excel headers

For Excel files, the dashboard scans the first 30 rows for separate time and thickness header cells. A title such as `Thickness vs Time` in one cell is ignored, allowing direct ellipsometer exports where the real column headers appear on a later row.

## Extrema detection

The extrema anchor sequence is:

```text
minimum → maximum → minimum
```

SciPy relative-extrema `order` controls how many neighboring samples a candidate must beat to count as a local minimum or maximum:

- smaller order = more sensitive to local structure/noise,
- larger order = more selective.

The extrema settings are independent of the transition detector.

## Missing-point recovery

The dashboard can flag broken alternation patterns:

```text
MAX → MAX  = possible missing MIN
MIN → MIN  = possible missing MAX
```

Recovery performs a local extrema search inside the selected gap and adds one candidate without deleting the globally detected extrema.

## Input

Supported files:

- `.csv`
- `.xlsx`
- `.xls`

Select the time and thickness columns in the interface.

## Run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Recommended workflow

1. Upload the dataset and select time/thickness columns.
2. Select the analysis window.
3. Tune minimum and maximum extrema orders until the minima and maximum anchors follow the physical cycles.
4. Start with smoothing = 3, transition threshold = 35%, persistence = 2.
5. Visually inspect B and C on opposite sides of each maximum anchor.
6. If either transition is not resolvable, leave that cycle excluded rather than forcing a point.
7. Download the CSV when satisfied.

## Output columns

```text
Cycle,
Point A Index, Point A Time, Point A Thickness,
Point B Index, Point B Time, Point B Thickness,
Point C Index, Point C Time, Point C Thickness,
Point D Index, Point D Time, Point D Thickness,
Delta 1, Delta 2, Delta 3
```

The maximum anchor is shown on the plot but is not added to the exported A/B/C/D columns.

## Testing

The test suite verifies that:

- B is found only on the rising side of the maximum,
- C is found only on the falling side,
- neither B nor C can equal the maximum anchor,
- unresolved transitions reject the cycle, and
- Δ1/Δ2/Δ3 arithmetic and output columns remain consistent.

Run:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```
