# Thickness Cycle Delta Analyzer

A Streamlit dashboard for analyzing cyclic thickness-vs-time data for an **ALD/ALE process**. The program detects repeated extrema, validates complete cycles, identifies Point C as the **onset of the rapid etch-related thickness decrease**, and reports Δ1, Δ2, and Δ3 for each valid cycle.

The dashboard uses a strict **minimum → maximum → minimum** rule in **forward physical time**.

## ALD/ALE process definition

```text
A = first minimum
B = maximum
C = onset of the sustained rapid etch-related thickness decrease
D = next minimum

Δ1 = B - A
Δ2 = B - C
Δ3 = C - D
```

## Point C: etch-onset detection

Point C is no longer defined as the largest second-derivative response or the center of the strongest inflection. That definition can fail when the physically meaningful etch step is nearly instantaneous, because the largest curvature naturally occurs inside the sharp drop.

For each B → D segment the current detector:

1. lightly smooths thickness with a low-order Savitzky-Golay filter,
2. calculates the first derivative `dh/dt`,
3. finds the most negative slope, which identifies the strongest etch rate,
4. estimates the purge/pre-etch slope before that event,
5. defines an onset threshold between the purge slope and strongest etch slope, and
6. walks backward from the strongest etch point to the start of the connected threshold-crossing region.

The start of that region is Point C. A narrow or nearly instantaneous etch is therefore allowed rather than intentionally filtered out.

## Point C controls

- **Smoothing window (samples)** — light smoothing before `dh/dt` is calculated. The default is 5 samples so a rapid etch is not excessively broadened.
- **Etch onset threshold (% of slope change)** — default 35%. Lower values detect an earlier onset; higher values place C closer to the steepest etch slope.
- **Etch persistence (samples)** — default 2. The etch-rate region must remain connected through the strongest etch point for at least this many samples. Use 1 for an extremely short event.

The Savitzky-Golay polynomial order is fixed internally at 2 for the dashboard rather than exposed as a primary physical control.

## Time direction

The uploaded file may be ordered with time increasing or decreasing. The dashboard automatically sorts the selected time column into **ascending chronological order before any extrema or Point C calculations**.

This is important because Point C is defined as the onset of etching in forward physical time. Without this normalization, a reverse-chronological export would make the algorithm walk toward the wrong side of the etch event and would also reverse the interpretation of A/B/D cycle boundaries.

If the file is detected as descending in time, the dashboard displays a notice. If the time values are non-monotonic, it displays a warning and still sorts them before analysis.

Because of this normalization, reported point indices refer to the **chronologically sorted analysis window**, not necessarily the original Excel row number. Point times are the recommended reference when comparing datasets.

## Extrema detection

A cycle contributes only when extrema occur as:

```text
minimum → maximum → minimum
```

SciPy relative-extrema `order` controls how many neighboring samples a candidate must beat to count as a local minimum or maximum:

- smaller order = more sensitive to local structure/noise,
- larger order = more selective.

Minimum and maximum orders are independent of Point C detection.

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
2. Confirm whether the dashboard reports that time was reversed and normalized.
3. Select the analysis window.
4. Tune minimum and maximum extrema orders so A/B/D markers follow the physical cycles.
5. Start Point C with smoothing = 5, onset threshold = 35%, persistence = 2.
6. If C is too far into the drop, lower the onset threshold.
7. If C is triggered too early by noise, increase the onset threshold and/or persistence.
8. For a truly one-sample etch event, set persistence = 1.
9. Visually inspect Point C markers and download the CSV when satisfied.

## Output columns

```text
Cycle,
Point A Index, Point A Time, Point A Thickness,
Point B Index, Point B Time, Point B Thickness,
Point C Index, Point C Time, Point C Thickness,
Point D Index, Point D Time, Point D Thickness,
Delta 1, Delta 2, Delta 3
```

## Testing

The test suite includes:

- an abrupt etch case where Point C must remain on the purge-side edge of the drop,
- a gradual transition case where Point C must occur before the maximum etch rate,
- a threshold-direction test verifying that a lower onset threshold moves C earlier or leaves it unchanged, and
- cycle arithmetic/output-column checks.

Run:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```
