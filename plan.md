# camdl-viewer — a Streamlit helper-viewer for camdl output

## Context

camdl writes simulation output into a content-addressable storage (CAS) tree:
`runs/{sim_hash8}/{scenario_slug}-{scen_hash8}/seed_{n}/{traj.tsv, run.json}`.
Today the only way to look at that output is `camdl list/show/cat` (Rust CLI) or
ad-hoc polars/matplotlib in notebooks. The user wants a **simple tool that spins
up a set of tabbed helper viewers** for this output:

1. A **CAS file browser** tab — pick a run, see its `run.json` metadata + `traj.tsv`.
2. A **simulation ensemble** tab — overlay many replicate trajectories as faint
   same-color spaghetti, with median + 50%/95% prediction-interval ribbons and
   observed values on top; scrub replicates with a slider and toggle scenarios.

Decisions confirmed with the user: **Python + Streamlit**; slider scrubs
**seeds/replicates** with **scenarios as checkboxes** grouped in one control
panel; "lineages" means **trajectory spaghetti lines** (not camdl genealogy
trees); data source is a **CAS `runs/` directory**; build **both tabs**.

The project lives at `/Users/vsb/projects/work/camdl-viewer` (currently empty).

## Verified data-format facts (checked against real files)

- **`traj.tsv`**: line 1 = `# camdl <ver> (<date>)` comment; line 2 = header
  `t<TAB>S<TAB>I<TAB>R<TAB>flow_infection<TAB>flow_recovery`; then TSV rows.
  Column 0 is `t` (float); rest are numeric streams (compartments, then `flow_*`).
- **`run.json` has TWO formats — must handle both**:
  - *Old flat* (on disk in the book examples): top-level `model, scenario,
    sim_hash, scen_hash, seed, backend, dt, version, created_at, argv`. No
    `kind`/`status`/`hash`.
  - *New tagged* (canonical `run_meta.rs` `Run`/`RunKind`): top-level `hash,
    version, created_at, argv, status, label, kind`; sim fields nested under
    `kind = {"kind":"simulate", model, sim_hash, scen_hash, seed, ...}`.
    `status` is `"running"` or `{"completed":{"wall_time_seconds":...}}`.
- **Observed data** (`synthetic_data.tsv`, `data/in_bed.tsv`): plain wide TSV,
  no comment line, header `time<TAB><stream...>`; `time` may be non-contiguous.
- **Quantiles**: match camdl's `quantile_sorted` (external-harness/summary.rs
  L111–121) exactly with `numpy.quantile(method="linear")` (R type=7).
- **Tooling**: camdl-book uses **uv** (Python 3.13, polars 1.39). Use uv + polars
  + numpy + plotly + streamlit.

Reference files to mirror (read-only):
`/Users/vsb/projects/work/camdl/rust/crates/cli/src/run_meta.rs` (run.json schema),
`/Users/vsb/projects/work/camdl/rust/crates/external-harness/src/summary.rs` (quantiles).

## File / module layout

```
camdl-viewer/
  pyproject.toml          # uv: streamlit, plotly, polars, numpy
  README.md               # launch + verification
  app.py                  # entrypoint: arg parsing (sidebar/--runs), st.tabs dispatch
  camdl_viewer/
    __init__.py
    cas.py                # discovery + run.json/traj loading + RunRecord/CasIndex
    ensemble.py           # time-grid alignment + PI computation (numpy)
    observed.py           # observed-data discovery + stream matching
    plotting.py           # Plotly figure builders
    ui_browser.py         # Tab 1 render
    ui_ensemble.py        # Tab 2 render
```

Init with `uv init` then `uv add streamlit plotly polars numpy`.

## Core data structures & functions

**`cas.py`**
- `RunRecord(scenario, seed, sim_hash, scen_hash, kind, run_dir, traj_path, run_json: dict|None, parse_errors)`
- `ScenarioGroup(scenario, scen_hash, color, records[sorted by seed])`
- `CasIndex(runs_root, records, scenarios, warnings)`
- `normalize_run_json(raw) -> dict` — flatten old & new formats to a common dict;
  new format lifts fields out of nested `raw["kind"]`, sets `kind=raw["kind"]["kind"]`;
  old format defaults `kind="simulate"`. Never assume keys exist.
- `load_run_json(run_dir) -> (dict|None, errors)`
- `discover_runs(runs_root) -> CasIndex` — canonical walk of
  `runs/{sim8}/{slug}-{scen8}/seed_{n}/`; derive fields from run.json when present,
  else from path parts. **Fallback**: if canonical walk is empty, glob `**/traj.tsv`
  and infer best-effort. Group `kind=="simulate"` records by `(scenario, scen_hash)`,
  assign deterministic colors (fixed palette indexed by sorted scenario name so a
  scenario keeps its color regardless of checkbox state). Collect warnings.
- `load_traj(path) -> pl.DataFrame` — `pl.read_csv(sep="\t", comment_prefix="#")`, cast `t` Float64.
- `traj_stream_columns(df) -> list[str]` — all cols except `t`.

**`ensemble.py`** (the tricky alignment + PI math)
- Replicates may have **different time grids / lengths** (epidemics die out early;
  ODE backends emit non-integer `t`). Align via a **union time axis** (sorted unique
  `t` across included reps; `intersection` mode optional), then `np.interp` each
  replicate's stream onto it with `left=right=NaN` (no extrapolation).
- `assemble_ensemble(records, traj_loader, stream, axis_mode="union") -> EnsembleSeries(time, replicates[R,T], seeds)`
- `compute_pi(series, quantiles=(.025,.25,.5,.75,.975)) -> PIBands(time, median, q025, q25, q75, q975, n_per_t)`
  via `np.nanquantile(..., axis=0, method="linear")`. NaN-aware is load-bearing —
  one short replicate must not poison later time points. `n==1` → bands collapse to the line.

**`observed.py`** — best-effort/optional.
- `discover_observed(runs_root, extra_paths) -> list[ObservedSeries]` — search each
  `run_dir/obs/*.tsv`, standalone `--obs` paths, and `runs_root` parent `*.tsv` with a
  time/date column; numeric non-meta columns are streams.
- `match_observed(observed, stream) -> ObservedSeries|None` — by stream-name equality
  (then case-insensitive). No scenario matching (observed = ground truth, plotted once).

**`plotting.py`** — Plotly `go.Figure`.
- `ensemble_figure(groups_data, stream, show_spaghetti, show_median, show_pi50, show_pi95, highlight_seed, max_replicates, observed)`.
  Per scenario, z-order: 95% ribbon (q025→q975, `fill='tonexty'`, alpha ~0.12) →
  50% ribbon (q25→q75, alpha ~0.25) → spaghetti (one Scatter/replicate, scenario color,
  `opacity~0.15`, `width=1`, `connectgaps=False`, no legend) → median (solid, width 2.5,
  one legend entry per scenario) → observed (added once, black markers).
  `highlight_seed` redraws that rep at `opacity=1, width=2`. Layout `plotly_white`,
  `hovermode='x unified'`, axis titles `t` / `stream`.
- `traj_figure(df, columns)` — simple multi-line plot for Tab 1.

## Streamlit UI

`app.py`: `st.set_page_config(layout="wide")`; read `--runs`/`--obs` from `sys.argv`
after `--` (argparse), fallback to sidebar `text_input`; sidebar "rescan" button;
`st.tabs(["CAS browser", "Simulation ensemble"])`. Wrap `discover_runs`, `load_traj`,
`assemble_ensemble`, `compute_pi` in `@st.cache_data` keyed on path + dir-mtime token
(rescan busts cache). Pass `Path` as `str` into cached fns. Give every widget an
explicit `key=`.

**Tab 1 (`ui_browser.py`)**: left column = scenario→seed selectors picking a
`RunRecord`; right column = `st.json(run_json)` + column `multiselect` + `traj_figure`
plotly chart + raw table in an expander (`df.to_pandas()`). Show `CasIndex.warnings`.

**Tab 2 (`ui_ensemble.py`)**: `controls, plot = st.columns([1,3])`.
Controls (one panel): scenario **checkboxes** (default all on, color swatch in label);
**stream** selectbox (union of stream cols); **replicates** slider `1..max_seeds`
(controls `max_replicates`); **highlight seed** selectbox; overlay checkboxes
(spaghetti / median / 50% PI / 95% PI / observed); time-axis radio (union/intersection).
Plot column assembles + computes PI per selected scenario (cached) and renders the figure;
caption reports per-scenario replicate count and `n_per_t` range.

**Seed-slider ↔ PI rule**: PIs are computed over **all** replicates regardless of the
slider; the slider/highlight only change which spaghetti lines are drawn (a 95% PI from
3 of 50 reps would mislead). Caption states "PI over all N reps; showing M lines".

## Launch

```
cd /Users/vsb/projects/work/camdl-viewer
uv run streamlit run app.py -- --runs /path/to/output/runs [--obs file.tsv ...]
```

## Verification

1. **Smoke (single seed)**: `--runs
   /Users/vsb/projects/work/camdl-book/examples/getting-started/output/runs`.
   Tab 1 lists `baseline / seed_1`, shows old-format run.json, plots `S,I,R,flow_*`.
   Tab 2 shows one scenario; single spaghetti line; bands collapse to it (`n_per_t==1`).
   Confirms old-format `normalize_run_json` + `n==1` quantile path.
2. **Observed overlay**: add `--obs
   .../guide/fitting-data/output/synthetic_data.tsv --obs
   .../guide/fitting-data/data/in_bed.tsv`. Select `in_bed` → markers appear;
   select `I` → graceful "no observed match" hint. Confirms discovery + non-contiguous time.
3. **Quantile parity**: in `uv run python`, `compute_pi` on `[1,2,3,4,5]` → q0.25==2.0,
   matching the Rust `quantile_sorted` test vector.
4. **Multi-seed (synthetic)**: copy `seed_1` → `seed_2/3` (edited values) under one and
   two scenario dirs; confirm grouping, coloring, ribbon widening, slider/highlight.
   Manual test step (no code artifact).

## Risks / notes

- run.json schema keeps evolving; `normalize_run_json` must be defensive and Tab 1
  always `st.json`s the raw dict so nothing is hidden.
- Filter non-`simulate` kinds (fits/profiles) out of ensemble groups; they can still
  appear in Tab 1.
- Keep loading in polars; convert to pandas only at the `st.dataframe`/render boundary.
