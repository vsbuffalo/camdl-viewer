"""One cohesive look for the whole app — set up once, used everywhere.

Two pieces, both applied by :func:`setup`:

* **CSS** (`_CSS`) — hides Streamlit's default chrome and restyles the few
  structural elements we lean on (sidebar tree rows, bordered "cards", small
  metadata tables). Injected once per run.
* **A registered Plotly template** (`"camdl"`) so every figure shares the same
  typography, transparent-on-card background, subtle grids, and a
  colourblind-safe (Okabe–Ito) colourway — no per-figure styling at the call
  site.

Plus small HTML helpers (`breadcrumb`, `section_title`, `meta_table`, …) that
the views compose. No emoji anywhere; Unicode marks only.
"""

from __future__ import annotations

from contextlib import contextmanager
from html import escape
from typing import Iterable

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- palette ---------------------------------------------------------------- #

INK = "#1f1e1c"
MUTED = "#6b6862"
PAPER = "#fbfaf7"
PANEL = "#ffffff"
LINE = "#e3e0d8"
ACCENT = "#3d5a80"

# Okabe–Ito qualitative palette (colourblind-safe), reordered so S/I/R read as
# blue / vermillion / green.
COLORWAY = [
    "#0072b2", "#d55e00", "#009e73", "#cc79a7",
    "#e69f00", "#56b4e9", "#7d5ba6", "#999999",
]

_STATUS = {
    "completed": ("✓", "ok", "completed"),
    "running": ("◌", "run", "running"),
    "failed": ("✗", "fail", "failed"),
}


# --- setup ------------------------------------------------------------------ #


def setup() -> None:
    """Inject the CSS and register + default the ``camdl`` Plotly template."""
    st.markdown(_CSS, unsafe_allow_html=True)
    pio.templates["camdl"] = go.layout.Template(
        layout=dict(
            colorway=COLORWAY,
            font=dict(family=_MONO, size=12, color=INK),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True, gridcolor="rgba(31,30,28,0.06)", zeroline=False,
                linecolor="rgba(31,30,28,0.22)", ticks="outside",
                tickcolor="rgba(31,30,28,0.22)", ticklen=4,
            ),
            yaxis=dict(
                showgrid=True, gridcolor="rgba(31,30,28,0.06)", zeroline=False,
                linecolor="rgba(31,30,28,0.22)", ticks="outside",
                tickcolor="rgba(31,30,28,0.22)", ticklen=4,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=11),
            ),
            hoverlabel=dict(
                bgcolor=PANEL, bordercolor="rgba(31,30,28,0.2)",
                font=dict(family=_MONO, size=11),
            ),
            margin=dict(l=8, r=8, t=26, b=8),
        )
    )
    pio.templates.default = "camdl"


# --- small composable HTML helpers ------------------------------------------ #


def section_title(text: str) -> None:
    st.markdown(f"<div class='sec-title'>{escape(text)}</div>", unsafe_allow_html=True)


@contextmanager
def card(title: str | None = None):
    """A restyled bordered container; optional small-caps section title."""
    box = st.container(border=True)
    with box:
        if title:
            section_title(title)
        yield box


def status_mark(status: str) -> tuple[str, str, str]:
    """``(glyph, css_class, word)`` for a run status."""
    return _STATUS.get(status, ("·", "muted", status or "unknown"))


def crumbs_line(parts: list[str], *, right: str = "") -> None:
    """A monospace ``a › b › c`` trail; ``right`` is raw HTML pinned right."""
    sep = "<span class='crumb-sep'>›</span>"
    trail_parts = [f"<span>{escape(p)}</span>" for p in parts[:-1]]
    if parts:
        trail_parts.append(f"<b>{escape(parts[-1])}</b>")
    st.markdown(
        f"<div class='crumbs'><span class='trail'>{sep.join(trail_parts)}</span>"
        f"<span class='runmeta'>{right}</span></div>",
        unsafe_allow_html=True,
    )


def breadcrumb(parts: list[str], *, status: str, run_id8: str) -> None:
    """A run breadcrumb with a status badge + run_id on the right."""
    glyph, cls, word = status_mark(status)
    right = (
        f"<span class='{cls}'>{glyph} {escape(word)}</span>"
        f"<span class='mono dim'>{escape(run_id8)}</span>"
    )
    crumbs_line(parts, right=right)


def scenario_colors(labels: Iterable[str]) -> dict[str, str]:
    """Stable colour per scenario label (sorted, from the shared colourway)."""
    return {lab: COLORWAY[i % len(COLORWAY)] for i, lab in enumerate(sorted(set(labels)))}


def meta_table(headers: list[str], rows: Iterable[Iterable[str]]) -> None:
    """A small, styled key/value-ish table for run metadata (not bulk data)."""
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    st.markdown(
        f"<table class='meta'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>",
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f"<div class='note'>{escape(text)}</div>", unsafe_allow_html=True)


# --- the stylesheet --------------------------------------------------------- #

_SANS = "'Inter',ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif"
_MONO = "'IBM Plex Mono',ui-monospace,'SF Mono',Menlo,monospace"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --ink:{INK}; --muted:{MUTED}; --paper:{PAPER}; --panel:{PANEL};
  --line:{LINE}; --accent:{ACCENT}; --accent-faint:rgba(61,90,128,0.10);
  --guide:rgba(31,30,28,0.30);
}}

/* hide default streamlit chrome. The toolbar holds the mobile sidebar-opener,
   and the header holds the toolbar — so only hide them on desktop (where the
   sidebar is pinned open and no opener is needed). #MainMenu / footer are safe
   to hide everywhere. */
#MainMenu, footer {{ display:none !important; }}
@media (min-width: 769px) {{
  header[data-testid="stHeader"] {{ background:transparent; height:0; }}
  [data-testid="stToolbar"] {{ display:none !important; }}
}}
/* DESKTOP ONLY: pin the sidebar open (hide the collapse control + force it
   visible even if a prior session left it collapsed), so it can't get stuck
   off-screen. On mobile the sidebar is a full-screen overlay that MUST stay
   collapsible, so none of this applies below 769px — Streamlit's normal
   open/close control is left intact there. */
@media (min-width: 769px) {{
  [data-testid="stSidebarCollapseButton"] {{ display:none !important; }}
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"][aria-expanded="false"] {{
    display:block !important; visibility:visible !important;
    transform:none !important; margin-left:0 !important;
    min-width:320px !important; width:320px !important;
  }}
  section[data-testid="stSidebar"][aria-expanded="false"] > div {{ display:block !important; }}
}}

html, body, .stApp, [class*="css"] {{ font-family:{_SANS}; color:var(--ink); }}
.stApp {{ background:var(--paper); }}
code, kbd, .mono {{ font-family:{_MONO}; font-size:0.82em; }}
.block-container {{ padding-top:1.1rem; padding-bottom:2rem; max-width:1500px; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background:var(--secondary,var(--paper)); border-right:1px solid var(--line); }}
section[data-testid="stSidebar"] > div {{ padding-top:1.1rem; }}
.side-title {{ font-family:{_MONO}; font-weight:600; font-size:0.95rem; letter-spacing:0.02em; margin-bottom:.4rem; }}
.side-sub {{ font-family:{_MONO}; font-size:0.72rem; color:var(--muted); margin:.5rem 0 .1rem; word-break:break-all; }}

/* Flush the inter-row gap so the CSS guide lines (drawn per row, below) tile
   into continuous verticals; restore breathing room on the non-tree elements. */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap:0 !important; }}
section[data-testid="stSidebar"] .side-title {{ margin:.1rem 0 .55rem; }}
section[data-testid="stSidebar"] .side-sub {{ margin:.7rem 0 .15rem; }}
section[data-testid="stSidebar"] .sec-title {{ margin:1rem 0 .7rem; }}
section[data-testid="stSidebar"] [class*="st-key-rescan-btn"] {{ margin:.35rem 0 .5rem; }}
section[data-testid="stSidebar"] .stTextInput {{ margin-bottom:.35rem; }}

/* All sidebar buttons: flat, monospace, hard left-aligned (Streamlit nests a
   flex wrapper that re-centers the label — force every level left + full width). */
section[data-testid="stSidebar"] .stButton button {{
  width:100%; text-align:left !important; justify-content:flex-start !important;
  border:none; background:transparent; box-shadow:none;
  min-height:0; line-height:1.6; padding:1px 6px 1px 6px;
  font-family:{_MONO}; font-size:0.80rem; color:var(--ink); border-radius:4px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-weight:400;
}}
section[data-testid="stSidebar"] .stButton button > div,
section[data-testid="stSidebar"] .stButton button > div > span,
section[data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] {{
  width:100% !important; justify-content:flex-start !important; text-align:left !important;
}}
section[data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] p {{
  text-align:left !important; margin:0; width:100%;
}}
section[data-testid="stSidebar"] .stButton button:hover {{ background:rgba(31,30,28,0.06); }}
section[data-testid="stSidebar"] .stButton button:focus {{ box-shadow:none; }}
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] {{
  background:var(--accent-faint); color:var(--accent); font-weight:600;
}}
/* rescan: a small outlined button, not a tree row */
section[data-testid="stSidebar"] [class*="st-key-rescan-btn"] button {{
  border:1px solid var(--line); border-radius:6px; justify-content:center !important;
  padding:2px 8px;
}}
section[data-testid="stSidebar"] [class*="st-key-rescan-btn"] button [data-testid="stMarkdownContainer"] {{
  text-align:center !important;
}}
/* per-row tree guides (background) + indent are injected by views.tree_nav,
   keyed on each row's `.st-key-nav-btn-*` container. */

/* section titles + breadcrumb */
.sec-title {{
  font-family:{_MONO}; font-weight:600; font-size:0.68rem; letter-spacing:0.13em;
  text-transform:uppercase; color:var(--muted); margin:0 0 0.5rem;
}}
.crumbs {{
  display:flex; align-items:baseline; justify-content:space-between; gap:1rem;
  font-family:{_MONO}; font-size:0.82rem; color:var(--muted);
  padding:0 0 0.6rem; flex-wrap:wrap;
}}
.crumbs .trail b {{ color:var(--ink); font-weight:600; }}
.crumbs .crumb-sep {{ color:var(--line); margin:0 0.4rem; }}
.crumbs .runmeta {{ display:inline-flex; gap:0.6rem; align-items:baseline; white-space:nowrap; }}
.dim {{ color:var(--muted); }}
.ok {{ color:#2e7d32; }} .run {{ color:#b8860b; }} .fail {{ color:#c62828; }} .muted {{ color:var(--muted); }}

/* bordered containers -> cards */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border:1px solid var(--line) !important; border-radius:11px;
  background:var(--panel); padding:0.9rem 1.05rem;
}}

/* metadata tables */
table.meta {{ border-collapse:collapse; width:100%; font-size:0.80rem; margin:0.2rem 0 0.7rem; }}
table.meta th {{
  text-align:left; font-family:{_MONO}; font-weight:600; font-size:0.64rem;
  letter-spacing:0.08em; text-transform:uppercase; color:var(--muted);
  border-bottom:1px solid var(--line); padding:8px 32px 8px 2px;
}}
table.meta td {{ padding:8px 32px 8px 2px; border-bottom:1px solid var(--line); font-family:{_MONO}; vertical-align:top; }}
/* first column is the field/key — muted, roomy gutter, never wraps */
table.meta th:first-child, table.meta td:first-child {{ color:var(--muted); padding-right:40px; white-space:nowrap; }}
table.meta td:last-child, table.meta th:last-child {{ padding-right:2px; width:100%; }}
table.meta tr:last-child td {{ border-bottom:none; }}

.note {{
  font-size:0.82rem; color:var(--muted); background:rgba(31,30,28,0.03);
  border:1px solid var(--line); border-radius:8px; padding:0.5rem 0.75rem;
}}

/* tighten expanders + multiselect to match */
[data-testid="stExpander"] details {{ border:1px solid var(--line); border-radius:8px; background:transparent; }}
.stMultiSelect [data-baseweb="tag"] {{ background:var(--accent-faint); color:var(--accent); }}
</style>
"""
