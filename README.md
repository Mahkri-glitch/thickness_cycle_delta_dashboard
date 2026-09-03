# Thickness Cycle Delta Analyzer

A Streamlit dashboard for analyzing cyclic thickness-vs-time data for an **ALD/ALE process**. The program detects repeated extrema, validates complete cycles, identifies the Point C process transition, and reports Δ1, Δ2, and Δ3 for each valid cycle.

The dashboard uses a strict **minimum → maximum → minimum** cycle rule. Only complete successive cycles with a valid Point C transition are included in the reported deltas and averages.

## What the program does

1. Loads a CSV or Excel thickness-vs-time dataset.
2. Lets the user select an analysis window.
3. Detects local minima and maxima with `scipy.signal.argrelmin` and `scipy.signal.argrelmax`.
4. Keeps only complete successive `MIN → MAX → MIN` cycles.
5. Locates Point C from a broad curvature transition within the B → D segment.
6. Calculates Δ1, Δ2, and Δ3 for the ALD/ALE process.
7. Optionally searches for one missing local extremum in a broken alternation region.
8. Displays diagnostics, the annotated trace, per-cycle results, averages, and a downloadable CSV.

## ALD/ALE process definition

Each valid cycle uses four points:

```text
A = first minimum
B = maximum
C = broad transition into the final ALE-related drop
D = next minimum
```

The reported quantities are:

```text
Δ1 = B - A
Δ2 = B - C
Δ3 = C - D
```

## Point C transition detection

The previous implementation selected the single strongest negative change in slope. That made Point C sensitive to a very sharp C → D step because a nearly instantaneous drop can produce a much larger raw second-derivative response than the broader physical transition.

The current detector is designed to favor a **persistent transition** instead of the sharpest individual step:

1. Extract the B → D thickness segment.
2. Smooth the segment with a low-order Savitzky-Golay polynomial.
3. Calculate the first and second derivatives using the measured time coordinates.
4. Search the negative second derivative for curvature peaks.
5. Reject curvature peaks that are narrower than the selected minimum transition width.
6. Choose the remaining candidate with the greatest prominence as Point C.

### Point C controls

The sidebar contains three transition-filter settings:

- **Smoothing window (samples)** — controls how much local point-to-point structure is suppressed before derivatives are calculated. The window is always odd.
- **Polynomial order** — intentionally restricted to **2 or 3** so the local fit does not follow high-order wiggles or sharp point-to-point features too closely.
- **Minimum transition width (samples)** — rejects narrow curvature peaks. Increase this value when a steep C → D step is still being mistaken for Point C.

The default settings are:

```text
Smoothing window = 11 samples
Polynomial order = 2
Minimum transition width = 5 samples
```

These values are starting points rather than universal physical constants. Point C should still be visually inspected against the measured trace.

## Understanding the extrema `order` settings

The minimum and maximum extrema orders are separate from the Point C polynomial order.

SciPy's relative-extrema `order` determines how many neighboring samples a candidate point must beat to count as a minimum or maximum:

- **Smaller order**: more sensitive; may detect small features or noise.
- **Larger order**: more selective; suppresses smaller local extrema.

The minimum and maximum orders are independent because the two sides of a cycle can have different shapes or noise characteristics.

## Complete-cycle requirement

A cycle contributes to the results only when the detected extrema appear in this exact order:

```text
minimum → maximum → minimum
```

If the sequence is incomplete, interrupted by an extra extremum, or truncated by the selected analysis-window boundary, it is not included in the delta averages.

The dashboard does not force the detected number of cycles to match the programmed number of ALD/ALE cycles.

## Missing-point recovery

The dashboard can flag broken alternation patterns:

```text
MAX → MAX  = possible missing MIN
MIN → MIN  = possible missing MAX
```

When recovery is enabled, the program performs a local extrema search inside the selected suspicious gap and adds one candidate:

- the deepest local minimum for a missing MIN, or
- the highest local maximum for a missing MAX.

Recovery is **additive**. It does not delete globally detected extrema. If the restored extremum creates a valid complete cycle, Point C is recalculated automatically for that cycle.

The broken-alternation method cannot reliably infer an entire missing physical cycle when the remaining extrema still alternate correctly.

## Input format

The program accepts:

- `.csv`
- `.xlsx`
- `.xls`

The file needs at least two numeric columns. You select which column is time and which is thickness in the interface.

Example:

```csv
Time,Thickness
0.0,1.23
0.5,1.25
1.0,1.31
```

A small synthetic example is included at:

```text
examples/synthetic_example.csv
```

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the dashboard

```bash
python -m streamlit run app.py
```

## Recommended workflow

1. Upload the dataset.
2. Select the time and thickness columns.
3. Use **Analysis window** to isolate the cyclic region of interest.
4. Adjust the minimum and maximum extrema orders until A/B/D markers correspond to representative physical cycle points.
5. Start Point C detection with polynomial order 2, smoothing window 11, and minimum transition width 5.
6. Visually inspect Point C markers.
7. If a narrow C → D step is being selected, increase **Minimum transition width** first; increase the smoothing window if additional suppression is needed.
8. Check successful cycles, rejected sequences, and transition failures.
9. Use Missing Point Recovery only when a clear local extremum is absent and broken alternation is detected.
10. Download `cycle_delta_results.csv` for downstream analysis.

## Output columns

```text
Cycle,
Point A Index, Point A Time, Point A Thickness,
Point B Index, Point B Time, Point B Thickness,
Point C Index, Point C Time, Point C Thickness,
Point D Index, Point D Time, Point D Thickness,
Delta 1, Delta 2, Delta 3
```

Indices are relative to the **selected analysis window**, not necessarily the original untrimmed file.

## Diagnostics

The dashboard reports:

- full data points,
- analyzed data points,
- detected extrema,
- successful cycles,
- detected minima,
- detected maxima,
- rejected sequences, and
- transition failures.

A transition failure means a valid `MIN → MAX → MIN` extrema sequence was found but no curvature feature passed the Point C transition filters. This can occur when the B → D segment is too short, the time coordinate is not strictly increasing, or the minimum-width/smoothing settings are too restrictive for that cycle.

## Testing

The test suite includes a regression case containing both a broad process transition and a much larger narrow terminal step. The Point C detector is expected to select the broad transition rather than the narrow step.

Install development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Repository structure

```text
thickness_cycle_delta_dashboard/
├── app.py                      # Streamlit interface and plotting
├── analysis.py                 # Reusable scientific analysis functions
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development/test dependencies
├── README.md                    # Usage and interpretation guide
├── examples/
│   └── synthetic_example.csv
└── tests/
    └── test_analysis.py
```
