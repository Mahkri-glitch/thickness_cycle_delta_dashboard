# Thickness Cycle Delta Analyzer

A Streamlit dashboard for analyzing cyclic thickness-vs-time data for an **ALD/ALE process**. The program detects repeated extrema, uses each detected maximum as an internal anchor, identifies the low-slope purge/plateau around that maximum, and reports Δ1, Δ2, and Δ3 for each valid cycle.

The dashboard first identifies a **minimum → maximum → minimum** sequence in **forward physical time**.

## ALD/ALE process definition

```text
A = first minimum
B = entry into the low-slope purge/plateau after the active rise
M = detected maximum anchor inside or at an edge of the purge region
C = exit from the low-slope purge/plateau before the active fall
D = next minimum

Required ordering: A < B <= M <= C < D

Δ1 = B - A
Δ2 = B - C
Δ3 = C - D
```

The maximum is a reference point, not a forced process boundary. It may equal B, C, or both if that is what the sampled trace resolves.

## Purge plateau detection

For each A → maximum → D cycle the detector:

1. lightly smooths the local thickness trace,
2. calculates the slope between adjacent measurements,
3. uses the active positive rise before the maximum as a reference rate,
4. uses the active negative fall after the maximum as a reference rate,
5. starts at the maximum and expands left and right through the contiguous **low-slope** region, and
6. returns the left edge as B and the right edge as C.

Because plateau membership is based on **slope magnitude**, the purge region may drift slightly upward or downward instead of being perfectly flat.

## Controls

- **Smoothing window (samples)** — default **3**. Smaller smoothing helps preserve short ellipsometry transition events.
- **Purge plateau threshold (% of active slope)** — default **35%**. Lower values require a flatter purge region; higher values allow more positive/negative drift and widen the B → C region.

The Savitzky-Golay polynomial order is fixed internally at 2 in the dashboard.

## Time direction

The uploaded file may be ordered with time increasing or decreasing. The dashboard automatically sorts the selected time column into **ascending chronological order before any extrema or transition calculations**.

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

The extrema settings are independent of the purge plateau detector.

## Missing-point recovery

The dashboard can flag broken alternation patterns:

```text
MAX → MAX  = possible missing MIN
MIN → MIN  = possible missing MAX
```

Recovery performs a local extrema search inside the selected gap and adds one candidate without deleting the globally detected extrema.

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
3. Tune minimum and maximum extrema orders until the minima and maximum anchors follow the physical cycles.
4. Start with smoothing = 3 and purge plateau threshold = 35%.
5. If B/C are too close to the maximum, increase the plateau threshold.
6. If the purge region becomes too wide, lower the plateau threshold.
7. Visually inspect B and C and download the CSV when satisfied.

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

The test suite verifies plateau-entry/exit detection, slight purge drift, maximum-as-transition behavior, and Δ1/Δ2/Δ3 arithmetic.

Run:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```
