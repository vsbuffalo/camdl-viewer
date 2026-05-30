# camdl-viewer

Tabbed [Streamlit](https://streamlit.io) helper-viewers for
[camdl](../camdl) simulation output. Point it at a camdl content-addressable
storage (CAS) `runs/` directory and get:

- **CAS browser** — pick a run, inspect its `run.json` metadata and plot its
  `traj.tsv`.
- **Simulation ensemble** — overlay replicate trajectories as faint same-colour
  spaghetti, with median + 50%/95% prediction-interval ribbons (computed across
  replicates) and observed values on top. Toggle scenarios with checkboxes,
  scrub replicates with a slider, and switch which stream/compartment is plotted.

## Quick start (bundled measles demo)

A self-contained example is included: `he_measles.camdl` — a seasonally forced
measles SEIR in the spirit of He, Ionides & King (2010). Generate simulations
and observed data, then launch the viewer:

```sh
make        # camdl batch run -> 30 seeds x 3 scenarios; + a held-out "truth" obs series
make view   # launch the viewer on the generated CAS + observed data
```

`make` writes a CAS runs tree to `output/sims/` and observed data to
`data/observed/` (both produced by camdl — `camdl batch run` and a single
`camdl simulate --obs-dir` truth run). `make view` opens the ensemble tab with
three scenarios (baseline / strong seasonality / vaccination), 30 replicate
trajectories each, PI ribbons, and the observed `I` points overlaid.

## Run on your own data

```sh
uv run streamlit run app.py -- --runs /path/to/output/sims [--obs file.tsv ...]
```

Everything after `--` is passed to the app:

- `--runs PATH` — a camdl CAS `runs/` directory
  (`runs/{sim_hash8}/{scenario}-{scen_hash8}/seed_{n}/`). Optional; if omitted,
  set it in the sidebar.
- `--obs PATH` — extra observed-data TSV(s) to overlay (repeatable). Observed
  files in each run's `obs/` subdir and next to the runs dir are auto-discovered.

The sidebar also lets you change the runs directory / observed files and
**Rescan** after new runs are written.

Example (camdl-book getting-started SIR output):

```sh
uv run streamlit run app.py -- \
  --runs ../camdl-book/examples/getting-started/output/runs \
  --obs  ../camdl-book/guide/fitting-data/output/synthetic_data.tsv
```

## How it works

| Module | Responsibility |
| --- | --- |
| `camdl_viewer/cas.py` | Walk the CAS tree, parse `run.json` (handles both the old-flat and new-tagged `RunKind` formats), load `traj.tsv`, group sims into coloured scenarios. |
| `camdl_viewer/ensemble.py` | Align replicate trajectories onto a common time axis (`np.interp`, NaN outside each replicate's range) and compute prediction-interval bands with `np.nanquantile(method="linear")` — matching camdl's `quantile_sorted`. |
| `camdl_viewer/observed.py` | Best-effort discovery/matching of observed-data TSVs by stream name. |
| `camdl_viewer/plotting.py` | Plotly figures: spaghetti + ribbons + observed, and the simple Tab-1 trajectory plot. |
| `camdl_viewer/ui_browser.py`, `ui_ensemble.py` | The two tab renderers. |
| `app.py` | Arg parsing, sidebar, caching, tab dispatch. |

### Notes

- **Prediction intervals are computed over *all* replicates**, regardless of the
  replicate slider. The slider and the "highlight seed" selector only change
  which spaghetti lines are drawn — a 95% PI from a handful of replicates would
  mislead. The caption reports `PI over N reps, showing M`.
- **Ragged trajectories** (epidemics that die out early, or non-integer ODE time
  grids) are handled: each replicate is interpolated onto the shared axis and
  contributes NaN outside its own time range, so quantiles stay honest per time
  point.
- Only `kind == "simulate"` runs feed the ensemble tab; other kinds (fits,
  profiles, …) still appear in the CAS browser.
