"""
Interface layer.

Holds the stylesheet and the HTML fragments the dashboard reuses. Custom markup
is namespaced with a ``wd-`` prefix so styling does not depend on Streamlit's
internal class names any more than necessary.

Visual direction: a field control interface. Deep canopy green carries the
chrome, a bright crop green marks growth, and a single alert orange is reserved
for weeds — the one thing in this system that demands action.
"""

from __future__ import annotations

import html
import re
from typing import Iterable, Optional, Sequence, Tuple

from utils.config import PALETTE

_LINE_BREAKS = re.compile(r"\s*\n\s*")
_BETWEEN_TAGS = re.compile(r">\s+<")


def compact(markup: str) -> str:
    """
    Flatten markup onto one line.

    Markdown reads any line indented by four spaces as a code block, so indented
    HTML passed to ``st.markdown`` renders as literal source instead of markup.
    Every fragment this module emits goes through here, and any inline HTML
    written elsewhere in the app must do the same.
    """
    single_line = _LINE_BREAKS.sub(" ", markup)
    return _BETWEEN_TAGS.sub("><", single_line).strip()


FONT_IMPORT = (
    "https://fonts.googleapis.com/css2"
    "?family=Archivo:wght@500;600;700;800"
    "&family=Inter:wght@400;500;600"
    "&display=swap"
)

STYLESHEET = f"""
<style>
@import url('{FONT_IMPORT}');

:root {{
    --canvas: {PALETTE['canvas']};
    --surface: {PALETTE['surface']};
    --line: {PALETTE['line']};
    --ink: {PALETTE['ink']};
    --muted: {PALETTE['muted']};
    --faint: {PALETTE['faint']};
    --field: {PALETTE['field']};
    --field-deep: {PALETTE['field_deep']};
    --crop: {PALETTE['crop']};
    --crop-bright: {PALETTE['crop_bright']};
    --crop-soft: {PALETTE['crop_soft']};
    --weed: {PALETTE['weed']};
    --weed-soft: {PALETTE['weed_soft']};
    --tum: {PALETTE['institution']};
    --shadow: 0 1px 2px rgba(11,31,20,.05), 0 14px 32px -20px rgba(11,31,20,.28);
}}

html, body, .stApp, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

.stApp {{ background: var(--canvas); color: var(--ink); }}

.block-container {{
    padding-top: 0.9rem;
    padding-bottom: 2.5rem;
    max-width: 1360px;
}}

h1, h2, h3, h4 {{ font-family: 'Archivo', 'Inter', sans-serif; color: var(--ink); }}

/* ---------- Masthead ---------- */

.wd-masthead {{
    background: var(--field);
    background-image:
        radial-gradient(circle at 88% -30%, rgba(70,185,106,.22), transparent 58%);
    border-radius: 16px;
    padding: 1.45rem 1.8rem 1.35rem;
    margin-bottom: 0.95rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
    color: #FFFFFF;
}}
.wd-masthead h1 {{
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.035em;
    margin: 0 0 0.35rem 0;
    color: #FFFFFF;
    max-width: 19ch;
}}
.wd-masthead p {{
    margin: 0;
    font-size: 0.94rem;
    color: rgba(255,255,255,.74);
    max-width: 52ch;
}}
.wd-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--crop-bright);
    margin-bottom: 0.55rem;
}}
.wd-eyebrow::before {{
    content: "";
    width: 26px;
    height: 3px;
    background: var(--crop-bright);
    border-radius: 2px;
}}

.wd-status {{
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.5rem 0.95rem;
    border-radius: 999px;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.18);
    font-size: 0.85rem;
    font-weight: 500;
    color: #FFFFFF;
    white-space: nowrap;
}}
.wd-dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--crop-bright); }}
.wd-dot.idle {{ background: rgba(255,255,255,.5); }}
.wd-dot.warn {{ background: #F5A524; }}

@media (max-width: 980px) {{
    .wd-masthead {{ flex-direction: column; align-items: flex-start; padding: 1.2rem; }}
    .wd-masthead h1 {{ font-size: 1.7rem; }}
}}

/* ---------- Metric row ---------- */

.wd-metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.7rem;
    margin-bottom: 1rem;
}}
.wd-metric {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.95rem 1.1rem 0.9rem;
    position: relative;
    overflow: hidden;
}}
.wd-metric::before {{
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 5px;
    background: var(--accent, var(--line));
}}
.wd-metric-label {{
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--muted);
    margin-bottom: 0.45rem;
}}
.wd-metric-value {{
    font-family: 'Archivo', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 0.92;
    letter-spacing: -0.045em;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
}}
.wd-metric-value .unit {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--muted);
    margin-left: 0.25rem;
    letter-spacing: 0;
}}
.wd-metric-foot {{
    font-size: 0.78rem;
    color: var(--faint);
    margin-top: 0.5rem;
}}

/* The weed count is the one figure here that demands action, so it is the one
   card that inverts. */
.wd-metric.alert {{ background: var(--field); border-color: var(--field); }}
.wd-metric.alert .wd-metric-label {{ color: rgba(255,255,255,.72); }}
.wd-metric.alert .wd-metric-value {{ color: {PALETTE['weed_bright']}; }}
.wd-metric.alert .wd-metric-value .unit {{ color: rgba(255,255,255,.6); }}
.wd-metric.alert .wd-metric-foot {{ color: rgba(255,255,255,.6); }}

@media (max-width: 980px) {{
    .wd-metrics {{ grid-template-columns: repeat(2, 1fr); }}
    .wd-metric-value {{ font-size: 1.8rem; }}
}}

/* ---------- Section headings ---------- */

.wd-section-title {{
    font-family: 'Archivo', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 1.5rem 0 0.3rem 0;
}}
.wd-section-title::after {{
    content: "";
    display: block;
    width: 34px;
    height: 4px;
    background: var(--crop);
    border-radius: 2px;
    margin-top: 0.45rem;
}}
.wd-section-note {{
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.55;
    max-width: 76ch;
    margin-bottom: 0.7rem;
}}

/* ---------- Panels ---------- */

.wd-panel {{
    border: 1px solid var(--line);
    border-radius: 14px;
    background: var(--surface);
    padding: 1rem 1.15rem;
    margin-bottom: 0.6rem;
    box-shadow: var(--shadow);
}}
.wd-panel h3 {{
    font-size: 1.08rem;
    font-weight: 700;
    letter-spacing: -0.015em;
    margin: 0 0 0.5rem 0;
}}
.wd-panel p {{ color: var(--muted); font-size: 0.92rem; margin: 0; line-height: 1.65; }}
.wd-panel ul {{ margin: 0.5rem 0 0 1.1rem; color: var(--muted); font-size: 0.92rem; }}
.wd-panel li {{ margin-bottom: 0.3rem; line-height: 1.5; }}
.wd-panel code {{
    background: var(--crop-soft);
    color: var(--field);
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    font-size: 0.85em;
}}

/* ---------- Stat band ---------- */

.wd-band {{
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: 1fr;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    margin: 0.1rem 0 0.8rem 0;
}}
.wd-band-cell {{ padding: 0.75rem 1rem; border-left: 1px solid var(--line); }}
.wd-band-cell:first-child {{ border-left: none; }}
.wd-band-label {{ font-size: 0.74rem; color: var(--muted); margin-bottom: 0.3rem; }}
.wd-band-value {{
    font-family: 'Archivo', sans-serif;
    font-size: 1.32rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    font-variant-numeric: tabular-nums;
}}
.wd-band-value.crop {{ color: var(--crop); }}
.wd-band-value.weed {{ color: var(--weed); }}

@media (max-width: 980px) {{
    .wd-band {{ grid-auto-flow: row; grid-auto-columns: auto; }}
    .wd-band-cell {{ border-left: none; border-top: 1px solid var(--line); }}
    .wd-band-cell:first-child {{ border-top: none; }}
}}

/* ---------- Key/value table ---------- */

.wd-table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
.wd-table td {{ padding: 0.46rem 0; border-bottom: 1px solid var(--line); }}
.wd-table tr:last-child td {{ border-bottom: none; }}
.wd-table td:first-child {{ color: var(--muted); }}
.wd-table td:last-child {{
    text-align: right;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}}

/* ---------- Provenance tags ---------- */

.wd-tag {{
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 600;
    border: 1px solid transparent;
}}
.wd-tag.published {{ background: #E9F1FA; color: var(--tum); border-color: #CFE0F2; }}
.wd-tag.measured {{ background: var(--crop-soft); color: var(--field); border-color: #C6E2CF; }}
.wd-tag.prototype {{ background: var(--weed-soft); color: var(--weed); border-color: #F6D6C4; }}

/* ---------- Model cards ---------- */

.wd-model {{
    border: 1px solid var(--line);
    border-radius: 14px;
    background: var(--surface);
    padding: 0.9rem 1rem;
    height: 100%;
    display: flex;
    flex-direction: column;
}}
.wd-model.selected {{
    border-color: var(--field);
    box-shadow: 0 0 0 2px rgba(20,67,42,.12);
}}
.wd-model-head {{ display: flex; justify-content: space-between; align-items: baseline; }}
.wd-model-name {{
    font-family: 'Archivo', sans-serif;
    font-weight: 700;
    font-size: 1.12rem;
    letter-spacing: -0.02em;
}}
.wd-model-year {{ font-size: 0.76rem; color: var(--faint); }}
.wd-model-arch {{
    font-size: 0.84rem;
    color: var(--muted);
    line-height: 1.5;
    margin: 0.4rem 0 0.7rem;
    flex: 1;
}}
.wd-model-metric {{ padding-top: 0.55rem; border-top: 1px solid var(--line); }}
.wd-model-metric-label {{ font-size: 0.74rem; color: var(--muted); margin-bottom: 0.2rem; }}
.wd-model-metric-value {{
    font-family: 'Archivo', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.035em;
    font-variant-numeric: tabular-nums;
    color: var(--field);
}}
.wd-model-missing {{ font-size: 0.78rem; color: var(--faint); line-height: 1.4; }}

/* ---------- Pipeline ---------- */

.wd-pipeline {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.7rem;
    margin: 0.2rem 0 0.9rem 0;
}}
.wd-step {{
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    background: var(--surface);
}}
.wd-step-index {{
    font-family: 'Archivo', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
    color: var(--line);
    margin-bottom: 0.35rem;
}}
.wd-step-name {{ font-weight: 600; font-size: 0.95rem; margin-bottom: 0.22rem; }}
.wd-step-detail {{ font-size: 0.82rem; color: var(--muted); line-height: 1.45; }}
.wd-step.active {{ border-color: var(--crop); }}
.wd-step.active .wd-step-index {{ color: var(--crop-bright); }}
.wd-step.blocked {{ background: #F4F6F2; border-style: dashed; }}
.wd-step.blocked .wd-step-name {{ color: var(--muted); }}

@media (max-width: 980px) {{ .wd-pipeline {{ grid-template-columns: repeat(2, 1fr); }} }}

/* ---------- Legend ---------- */

.wd-legend {{ display: flex; gap: 0.5rem; margin: 0.35rem 0 0.55rem 0; }}
.wd-legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 500;
    border: 1px solid var(--line);
    background: var(--surface);
}}
.wd-legend-box {{ width: 12px; height: 12px; border-radius: 3px; }}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {{ background: var(--field); border-right: none; }}
section[data-testid="stSidebar"] * {{ color: #E6EEE8; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.1rem; }}
section[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 0.42rem 0.6rem;
    border-radius: 9px;
    margin-bottom: 0.1rem;
    transition: background 120ms ease;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: rgba(255,255,255,.09);
}}
section[data-testid="stSidebar"] [role="radiogroup"] label p {{
    font-size: 0.94rem;
    font-weight: 500;
}}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.16); }}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    color: rgba(255,255,255,.58);
    font-size: 0.78rem;
    line-height: 1.55;
}}
.wd-brand {{ display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.9rem; }}
.wd-brand-mark {{
    width: 36px; height: 36px; border-radius: 10px;
    background: var(--crop-bright);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 1.05rem;
    color: var(--field-deep);
}}
.wd-brand-name {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1rem; }}
.wd-brand-sub {{ font-size: 0.76rem; color: rgba(255,255,255,.55); }}
.wd-sidestat {{ font-size: 0.84rem; line-height: 1.85; }}
.wd-sidestat span {{ color: rgba(255,255,255,.6); }}
.wd-sidestat strong {{ float: right; font-weight: 600; }}

/* ---------- Streamlit widgets ---------- */

[data-testid="stFileUploaderDropzone"] {{
    border: 2px dashed #B9C9BC;
    border-radius: 14px;
    background: var(--surface);
    padding: 0.9rem 1rem;
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--crop); }}

.stButton > button {{
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.3rem;
    border: 1px solid var(--line);
}}
.stButton > button[kind="primary"] {{
    background: var(--field);
    border-color: var(--field);
    color: #FFFFFF;
}}
.stButton > button[kind="primary"]:hover {{
    background: var(--field-deep);
    border-color: var(--field-deep);
}}

div[data-testid="stImage"] img {{
    border-radius: 12px;
    border: 1px solid var(--line);
}}

.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}

hr {{ border-color: var(--line); }}
*:focus-visible {{ outline: 2px solid var(--crop); outline-offset: 2px; }}

/* ---------- Vertical rhythm ---------- */
/* Streamlit's default element gap stacks on top of card padding, which is what
   produces the drifts of empty space between blocks. */

[data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}
[data-testid="stHorizontalBlock"] {{ gap: 1rem; }}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.35rem; }}

/* Streamlit leaves empty containers behind; they still occupy a gap. */
[data-testid="stElementContainer"]:empty,
[data-testid="stVerticalBlock"]:empty {{ display: none; }}

hr {{ margin: 0.8rem 0; }}
[data-testid="stExpander"] {{ margin-bottom: 0.2rem; }}
[data-testid="stCaptionContainer"] p {{ margin-bottom: 0.15rem; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 0.5rem; }}
div[data-testid="stImage"] {{ margin-bottom: 0.15rem; }}
[data-testid="stSlider"], [data-testid="stSelectbox"] {{ padding-bottom: 0.15rem; }}

@media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


# --------------------------------------------------------------------------
# HTML fragments
# --------------------------------------------------------------------------

def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def header(title: str, subtitle: str, affiliation: str, status_text: str, tone: str = "ok") -> str:
    dot_class = {"ok": "", "idle": " idle", "warn": " warn"}.get(tone, "")
    return compact(f"""
    <div class="wd-masthead">
        <div>
            <div class="wd-eyebrow">{_esc(affiliation)}</div>
            <h1>{_esc(title)}</h1>
            <p>{_esc(subtitle)}</p>
        </div>
        <div class="wd-status"><span class="wd-dot{dot_class}"></span>{_esc(status_text)}</div>
    </div>
    """)


def metric_strip(metrics: Sequence[dict]) -> str:
    cells = []
    for metric in metrics:
        accent = metric.get("accent", PALETTE["line"])
        alert = " alert" if metric.get("alert") else ""
        unit = metric.get("unit", "")
        unit_html = f'<span class="unit">{_esc(unit)}</span>' if unit else ""
        foot = metric.get("foot", "")
        foot_html = f'<div class="wd-metric-foot">{_esc(foot)}</div>' if foot else ""
        cells.append(
            f'<div class="wd-metric{alert}" style="--accent:{accent};">'
            f'<div class="wd-metric-label">{_esc(metric["label"])}</div>'
            f'<div class="wd-metric-value">{_esc(metric["value"])}{unit_html}</div>'
            f"{foot_html}</div>"
        )
    return compact(f'<div class="wd-metrics">{"".join(cells)}</div>')


def stat_band(cells: Sequence[Tuple[str, str, str]]) -> str:
    """A horizontal row of headline figures: (label, value, tone)."""
    rendered = []
    for label, value, tone in cells:
        tone_class = f" {tone}" if tone else ""
        rendered.append(
            f'<div class="wd-band-cell"><div class="wd-band-label">{_esc(label)}</div>'
            f'<div class="wd-band-value{tone_class}">{_esc(value)}</div></div>'
        )
    return compact(f'<div class="wd-band">{"".join(rendered)}</div>')


def panel(title: str, body_html: str) -> str:
    return compact(f'<div class="wd-panel"><h3>{_esc(title)}</h3>{body_html}</div>')


def kv_table(rows: Iterable[Tuple[str, str]]) -> str:
    body = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows)
    return compact(f'<table class="wd-table">{body}</table>')


def tag(text: str, kind: str = "measured") -> str:
    return f'<span class="wd-tag {kind}">{_esc(text)}</span>'


def legend(crop_label: str = "Soybean plant", weed_label: str = "Weed") -> str:
    return compact(f"""
    <div class="wd-legend">
        <div class="wd-legend-item">
            <span class="wd-legend-box" style="background:{PALETTE['crop']};"></span>{_esc(crop_label)}
        </div>
        <div class="wd-legend-item">
            <span class="wd-legend-box" style="background:{PALETTE['weed']};"></span>{_esc(weed_label)}
        </div>
    </div>
    """)


def pipeline(steps: Sequence[dict]) -> str:
    cells = []
    for index, step in enumerate(steps, start=1):
        state = step.get("state", "")
        state_class = f" {state}" if state else ""
        cells.append(
            f'<div class="wd-step{state_class}">'
            f'<div class="wd-step-index">{index}</div>'
            f'<div class="wd-step-name">{_esc(step["name"])}</div>'
            f'<div class="wd-step-detail">{_esc(step["detail"])}</div></div>'
        )
    return compact(f'<div class="wd-pipeline">{"".join(cells)}</div>')


def model_card(
    name: str,
    year: str,
    architecture: str,
    metric_label: str,
    metric_value: Optional[str],
    selected: bool = False,
) -> str:
    if metric_value:
        metric_html = (
            f'<div class="wd-model-metric">'
            f'<div class="wd-model-metric-label">{_esc(metric_label)}</div>'
            f'<div class="wd-model-metric-value">{_esc(metric_value)}</div></div>'
        )
    else:
        metric_html = (
            '<div class="wd-model-metric"><div class="wd-model-missing">'
            "No published value recorded</div></div>"
        )
    selected_class = " selected" if selected else ""
    return compact(
        f'<div class="wd-model{selected_class}">'
        f'<div class="wd-model-head"><span class="wd-model-name">{_esc(name)}</span>'
        f'<span class="wd-model-year">{_esc(year)}</span></div>'
        f'<div class="wd-model-arch">{_esc(architecture)}</div>'
        f"{metric_html}</div>"
    )


def section_title(title: str, note: str = "") -> str:
    note_html = f'<div class="wd-section-note">{_esc(note)}</div>' if note else ""
    return compact(f'<div class="wd-section-title">{_esc(title)}</div>{note_html}')


def bullet_panel(title: str, items: Sequence[str], intro: str = "") -> str:
    intro_html = f"<p>{_esc(intro)}</p>" if intro else ""
    items_html = "".join(f"<li>{_esc(item)}</li>" for item in items)
    return panel(title, f"{intro_html}<ul>{items_html}</ul>")


def brand(name: str, sub: str, mark: str = "W") -> str:
    return compact(
        f'<div class="wd-brand"><div class="wd-brand-mark">{_esc(mark)}</div>'
        f'<div><div class="wd-brand-name">{_esc(name)}</div>'
        f'<div class="wd-brand-sub">{_esc(sub)}</div></div></div>'
    )


def sidebar_stats(rows: Sequence[Tuple[str, str, str]]) -> str:
    """Sidebar status lines: (label, value, colour)."""
    body = "".join(
        f'<div><span>{_esc(label)}</span>'
        f'<strong style="color:{colour};">{_esc(value)}</strong></div>'
        for label, value, colour in rows
    )
    return compact(f'<div class="wd-sidestat">{body}</div>')
