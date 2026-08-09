# Thickness Cycle Delta Analyzer

A Streamlit dashboard for analyzing cyclic thickness-vs-time data, especially when the signal is expected to contain repeated local minima and maxima and the scientific quantity of interest is the thickness change between defined points within each complete cycle.

The project uses a strict **minimum → maximum → minimum** validation rule. Only complete successive cycles are included in the reported deltas and averages.

## What the program does

1. Loads a CSV or Excel thickness-vs-time dataset.
2. Lets the user select an analysis window.
3. Detects local minima and maxima with `scipy.signal.argrelmin` and `scipy.signal.argrelmax`.
4. Keeps only complete successive `MIN → MAX → MIN` cycles.
5. Calculates either a 2-step or 3-step delta model.
6. Optionally searches for one missing local extremum in a broken alternation region.
7. Displays diagnostics, the annotated trace, per-cycle results, averages, and a downloadable CSV.

## Scientific interpretation

### Complete-cycle requirement

A cycle contributes to the results only when the detected extrema appear in this exact order:

```text
minimum → maximum → minimum
```

If the sequence is incomplete, interrupted by an extra extremum, or truncated by the selected analysis-window boundary, it is not included in the delta averages.

This is intentional: the dashboard does not force the number of detected cycles to match the programmed number of ALD/ALE cycles.

### 2-step analysis

The 2-step model uses three physical points:

```text
A = first minimum
B = maximum
D = next minimum
```

The reported quantities are:

```text
Δ1 = B - A
Δ2 = B - D
```

Point C and Δ3 are left blank in the output because no transition point is used in 2-step mode.

### 3-step analysis

The 3-step model uses four points:

```text
A = first minimum
B = maximum
C = transition before the stronger final drop
D = next minimum
```

The reported quantities are:

```text
Δ1 = B - A
Δ2 = B - C
Δ3 = C - D
```

Point C is detected inside the B → D segment by:

1. calculating the local first derivative,
2. calculating the change in slope,
3. ignoring the segment boundaries, and
4. selecting the strongest negative interior change in slope.

This transition definition is useful for the intended 3-step shape, but it is not a universal physical law. **Always visually inspect Point C**, especially when the trace is noisy or contains an unrelated sharp feature.

## Understanding the extrema `order` setting

SciPy's relative-extrema `order` determines how many neighboring samples a candidate point must be lower or higher than.

- **Smaller order**: more sensitive; may detect small features or noise.
- **Larger order**: more selective; suppresses smaller local extrema.

The minimum and maximum orders are independent because the two sides of a cycle may have different shapes or noise characteristics.

A good workflow is to adjust `order` while visually checking whether the markers correspond to representative cycle minima and maxima.

## Missing-point recovery

The dashboard can flag broken alternation patterns:

```text
MAX → MAX  = possible missing MIN
MIN → MIN  = possible missing MAX
```

When recovery is enabled, the program performs a local extrema search only inside the selected suspicious gap. It then adds one candidate:

- the deepest local minimum for a missing MIN, or
- the highest local maximum for a missing MAX.

Recovery is **additive**. It does not delete the globally detected extrema. If the restored extremum creates valid complete cycles, those cycles are then processed normally. In 3-step mode, Point C is recalculated automatically for those recovered cycles and plotted with a separate marker.

### Limitation of recovery

The broken-alternation method can identify a likely missing point when the remaining extrema contain `MAX → MAX` or `MIN → MIN`. It cannot reliably infer an entire missing physical cycle when the remaining detected extrema still alternate correctly.

## Input format

The program accepts:

- `.csv`
- `.xlsx`
- `.xls`

The file needs at least two numeric columns. You select which column is time and which is thickness in the interface.

A typical CSV looks like:

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

From the repository root:

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, normally similar to:

```text
http://localhost:8501
```

## Recommended analysis workflow

1. Upload the dataset.
2. Select the time and thickness columns.
3. Use **Analysis window** to isolate the cyclic region of interest.
4. Select **2-step** or **3-step** analysis.
5. Adjust the minimum and maximum `order` values until the detected extrema correspond to representative physical cycle points.
6. Check the number of successful cycles and rejected sequences.
7. In 3-step mode, visually inspect Point C markers.
8. If a clear local extremum is missing and the dashboard identifies a broken alternation region, enable **Missing Point Recovery** and adjust the local order.
9. Review the per-cycle table and average deltas.
10. Download `cycle_delta_results.csv` for downstream analysis.

## Output columns

The downloaded CSV is standardized to:

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

A transition failure means a valid `MIN → MAX → MIN` extrema sequence was found but the 3-step transition calculation could not return a usable Point C, for example if the B → D segment was too short or contained invalid time spacing.

## Performance notes

Scientific calculations always use the full selected-resolution data. Only the rendered line plots are downsampled for faster Streamlit/Matplotlib redraws. Detected extrema, Point C locations, deltas, and averages are never calculated from downsampled plot data.

File parsing is cached with `st.cache_data`, which prevents Excel/CSV files from being reparsed every time a Streamlit control changes.

## Repository structure

```text
thickness_cycle_delta_dashboard/
├── app.py                      # Streamlit interface and plotting
├── analysis.py                 # Reusable scientific analysis functions
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development/test dependencies
├── README.md                    # Usage and interpretation guide
├── .gitignore
├── examples/
│   └── synthetic_example.csv
└── tests/
    └── test_analysis.py
```

Separating `analysis.py` from `app.py` makes the scientific logic easier to test and reduces the chance that UI edits alter the analysis algorithm.

## Run the tests

Install the development requirements:

```bash
python -m pip install -r requirements-dev.txt
```

Then run:

```bash
pytest -q
```

## Important interpretation notes

- The program detects signal features; it does not independently prove their physical origin.
- Do not use `abs()` to force deltas positive. A negative delta can be useful evidence that a detected point does not match the expected physical ordering.
- Analysis-window boundaries can suppress extrema because relative-extrema detection requires neighboring context. Include enough data before the first desired minimum and after the final desired minimum.
- Programmed process-cycle count, detected extrema count, and calculable cycle count do not have to be identical.
- Point C should be treated as a derivative-based operational definition and visually verified against the expected process behavior.
