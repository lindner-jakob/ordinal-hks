# Ordinal HKS experiments

Data and code for the asynchronous simulations in the paper "Ordinal Hegselmann-Krause Systems".
Each update activates a uniformly chosen unstable agent; initial opinions are
uniform on `[0,n/rho]`, with confidence radius 1. `T` counts individual updates.

Files:

- `data/scaling/`: size/density study, including the data for panels (a,b),
  summaries, fitted exponents and the parameter grid in `manifest.json`.
- `data/sweeps/`: rank sweep (`fig3`, panel c) and density sweep (`fig4`, panel d),
  with raw outcomes, integer seeds and summaries.
- `src/hks_extended_experiments.cpp`: size-study simulator and self-tests.
- `src/hks_sim_common.hpp`, `src/replay_sweeps.cpp`: simulator and adapter for
  the NumPy-generated profiles used in the rank/density sweeps.
- `scripts/run_hks_density_scaling.py`: run the complete size/density study.
- `scripts/analyze_hks_density_scaling.py`: compute summaries and scaling fits.
- `scripts/replay_archived_runs.py`: regenerate seeded sweep runs and compare
  stopping times with the archive; without `--all-sweeps`, run a small sample.

Requirements: Python 3.12, NumPy (`pip install -r requirements.txt`) and a C++17
compiler. Exact size-study reproduction was checked with GCC/libstdc++ 13.3.0.

Run from this directory:

```sh
# Regenerate the size/density study, then analyze its results.
python3 scripts/run_hks_density_scaling.py --workers 8
python3 scripts/analyze_hks_density_scaling.py

# Build and reproduce the rank/density sweeps.
mkdir -p build
g++ -O3 -std=c++17 src/hks_extended_experiments.cpp -o build/hks_extended_experiments
g++ -O3 -std=c++17 src/replay_sweeps.cpp -o build/replay_sweeps
python3 scripts/replay_archived_runs.py --all-sweeps
```

Outputs go to `results/`. Full runs can be expensive; the size-study runner
resumes completed batches. To analyze the existing data without new simulations:

```sh
mkdir -p results
cp -r data/scaling results/
python3 scripts/analyze_hks_density_scaling.py
```

Use the archived seeds as integers, without conversion to floating point.
The original bootstrap seeds for the older sweeps were not saved; their
intervals remain in the summaries but are not used in the paper's plots.

AI disclosure: the authors specified the designs; ChatGPT/Codex assisted with
implementation, execution and repository preparation. The authors
are responsible for reviewing the released material.
