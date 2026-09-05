"""
Interface layer.

Holds the stylesheet and the small set of HTML fragments the dashboard reuses.
Custom markup is namespaced with a ``wd-`` prefix so the styling does not depend
on Streamlit's internal class names any more than necessary.
"""

from __future__ import annotations

import html
import re
from typing import Iterable, List, Optional, Sequence, Tuple

from utils.config import PALETTE

_LINE_BREAKS = re.compile(r"\s*\n\s*")
_BETWEEN_TAGS = re.compile(r">\s+<")


def compact(markup: str) -> str:
    """
    Flatten markup onto one line.

    Markdown reads any line indented by four spaces as a code block, so indented
    HTML passed to ``st.markdown`` is rendered as literal source instead of
    markup. Every fragment this module emits goes through here, and any inline
    HTML written elsewhere in the app must do the same.
    """
    single_line = _LINE_BREAKS.sub(" ", markup)
    return _BETWEEN_TAGS.sub("><", single_line).strip()

FONT_IMPORT = (
    "https://fonts.googleapis.com/css2"
    "?family=Inter+Tight:wght@500;600;700"
    "&family=Inter:wght@400;500;600"
    "&display=swap"
)

STYLESHEET = f"""
<style>
@import url('{FONT_IMPORT}');

:root {{
    --paper: {PALETTE['paper']};
    --surface: {PALETTE['surface']};
    --border: {PALETTE['border']};
    --border-strong: {PALETTE['border_strong']};
    --ink: {PALETTE['ink']};
    --muted: {PALETTE['muted']};
    --faint: {PALETTE['faint']};
    --institution: {PALETTE['institution']};
    --crop: {PALETTE['crop']};
    --crop-soft: {PALETTE['crop_soft']};
    --weed: {PALETTE['weed']};
    --weed-soft: {PALETTE['weed_soft']};
    --signal: {PALETTE['signal']};
    --warning: {PALETTE['warning']};
}}

html, body, .stApp, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

.stApp {{ background: var(--paper); color: var(--ink); }}

.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1320px;
}}

h1, h2, h3, h4 {{
    font-family: 'Inter Tight', 'Inter', sans-serif;
    color: var(--ink);
    letter-spacing: -0.015em;
    font-weight: 600;
}}

/* ---------- Page header ---------- */

.wd-header {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
}}
.wd-header h1 {{
    font-size: 1.85rem;
    line-height: 1.15;
    margin: 0 0 0.35rem 0;
}}
.wd-header p {{
    margin: 0;
    color: var(--muted);
    font-size: 0.95rem;
}}
.wd-affiliation {{
    color: var(--institution);
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}}

.wd-status {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.8rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--surface);
    font-size: 0.82rem;
    color: var(--muted);
    white-space: nowrap;
}}
.wd-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--signal);
    box-shadow: 0 0 0 3px rgba(31, 157, 85, 0.14);
}}
.wd-dot.idle {{ background: var(--faint); box-shadow: 0 0 0 3px rgba(139,150,142,0.14); }}
.wd-dot.warn {{ background: var(--warning); box-shadow: 0 0 0 3px rgba(180,83,9,0.14); }}

/* ---------- Metric strip ---------- */
/* One instrument readout divided by hairlines, rather than four floating cards. */

.wd-metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    overflow: hidden;
    margin-bottom: 1.5rem;
}}
.wd-metric {{
    padding: 1.05rem 1.25rem;
    border-left: 1px solid var(--border);
}}
.wd-metric:first-child {{ border-left: none; }}
.wd-metric-label {{
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}
.wd-metric-value {{
    font-family: 'Inter Tight', sans-serif;
    font-size: 1.7rem;
    font-weight: 600;
    line-height: 1;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}}
.wd-metric-value .unit {{
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--muted);
    margin-left: 0.2rem;
    letter-spacing: 0;
}}
.wd-metric-foot {{
    font-size: 0.75rem;
    color: var(--faint);
    margin-top: 0.4rem;
}}
.wd-swatch {{ width: 9px; height: 9px; border-radius: 2px; display: inline-block; }}
.wd-swatch.crop {{ background: var(--crop); }}
.wd-swatch.weed {{ background: var(--weed); }}

@media (max-width: 900px) {{
    .wd-metrics {{ grid-template-columns: repeat(2, 1fr); }}
    .wd-metric:nth-child(3) {{ border-left: none; }}
    .wd-metric:nth-child(n+3) {{ border-top: 1px solid var(--border); }}
}}

/* ---------- Panels ---------- */

.wd-panel {{
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 1.2rem 1.35rem;
    margin-bottom: 1.1rem;
}}
.wd-panel h3 {{
    font-size: 1rem;
    margin: 0 0 0.2rem 0;
}}
.wd-panel p {{
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.55;
}}
.wd-panel ul {{ margin: 0.6rem 0 0 1.1rem; color: var(--muted); font-size: 0.9rem; }}
.wd-panel li {{ margin-bottom: 0.3rem; }}

.wd-section-title {{
    font-family: 'Inter Tight', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 1.9rem 0 0.35rem 0;
}}
.wd-section-note {{
    color: var(--muted);
    font-size: 0.88rem;
    margin-bottom: 0.9rem;
    max-width: 68ch;
    line-height: 1.55;
}}

/* ---------- Key/value table ---------- */

.wd-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
.wd-table td {{
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
}}
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
    padding: 0.2rem 0.55rem;
    border-radius: 5px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    border: 1px solid transparent;
}}
.wd-tag.published {{
    background: #EDF3FA;
    color: var(--institution);
    border-color: #D3E2F2;
}}
.wd-tag.measured {{
    background: var(--crop-soft);
    color: var(--crop);
    border-color: #CFE4D8;
}}
.wd-tag.prototype {{
    background: var(--weed-soft);
    color: var(--weed);
    border-color: #F2D8C9;
}}

/* ---------- Model cards ---------- */

.wd-model {{
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 1rem 1.15rem;
    height: 100%;
}}
.wd-model.selected {{ border-color: var(--institution); }}
.wd-model-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 0.15rem;
}}
.wd-model-name {{
    font-family: 'Inter Tight', sans-serif;
    font-weight: 600;
    font-size: 1.02rem;
}}
.wd-model-year {{ font-size: 0.78rem; color: var(--faint); }}
.wd-model-arch {{
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.5;
    margin-bottom: 0.75rem;
    min-height: 2.5rem;
}}
.wd-model-metric {{
    display: flex; justify-content: space-between;
    font-size: 0.85rem;
    padding-top: 0.55rem;
    border-top: 1px solid var(--border);
}}
.wd-model-metric span:last-child {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.wd-model-missing {{ font-size: 0.78rem; color: var(--faint); }}

/* ---------- Pipeline ---------- */

.wd-pipeline {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin: 0.4rem 0 1.2rem 0;
}}
.wd-step {{
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0.9rem 1rem;
    background: var(--surface);
}}
.wd-step-index {{
    font-size: 0.75rem;
    color: var(--faint);
    font-variant-numeric: tabular-nums;
    margin-bottom: 0.3rem;
}}
.wd-step-name {{ font-weight: 600; font-size: 0.92rem; margin-bottom: 0.25rem; }}
.wd-step-detail {{ font-size: 0.8rem; color: var(--muted); line-height: 1.45; }}
.wd-step.active {{ border-color: var(--crop); background: var(--crop-soft); }}
.wd-step.blocked {{ border-color: var(--border-strong); background: #F6F6F4; }}

@media (max-width: 900px) {{
    .wd-pipeline {{ grid-template-columns: repeat(2, 1fr); }}
}}

/* ---------- Legend ---------- */

.wd-legend {{
    display: flex; gap: 1.2rem; align-items: center;
    font-size: 0.85rem; color: var(--muted);
    margin: 0.5rem 0 0.9rem 0;
}}
.wd-legend-item {{ display: flex; align-items: center; gap: 0.45rem; }}
.wd-legend-box {{ width: 14px; height: 14px; border-radius: 3px; border: 2px solid; }}

/* ---------- Streamlit widget adjustments ---------- */

section[data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

[data-testid="stFileUploaderDropzone"] {{
    border: 1.5px dashed var(--border-strong);
    border-radius: 10px;
    background: var(--surface);
    box-shadow: 0 1px 2px rgba(22, 33, 27, 0.04);
}}

.stButton > button {{
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid var(--border-strong);
}}
.stButton > button[kind="primary"] {{
    background: var(--crop);
    border-color: var(--crop);
}}
.stButton > button[kind="primary"]:hover {{
    background: #266641;
    border-color: #266641;
}}

div[data-testid="stImage"] img {{
    border-radius: 8px;
    border: 1px solid var(--border);
}}

hr {{ border-color: var(--border); }}

*:focus-visible {{
    outline: 2px solid var(--institution);
    outline-offset: 2px;
}}

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
    <div class="wd-header">
        <div>
            <div class="wd-affiliation">{_esc(affiliation)}</div>
            <h1>{_esc(title)}</h1>
            <p>{_esc(subtitle)}</p>
        </div>
        <div class="wd-status"><span class="wd-dot{dot_class}"></span>{_esc(status_text)}</div>
    </div>
    """)


def metric_strip(metrics: Sequence[dict]) -> str:
    cells = []
    for metric in metrics:
        swatch = metric.get("swatch")
        swatch_html = f'<span class="wd-swatch {swatch}"></span>' if swatch else ""
        unit = metric.get("unit", "")
        unit_html = f'<span class="unit">{_esc(unit)}</span>' if unit else ""
        foot = metric.get("foot", "")
        foot_html = f'<div class="wd-metric-foot">{_esc(foot)}</div>' if foot else ""
        cells.append(
            f"""
            <div class="wd-metric">
                <div class="wd-metric-label">{swatch_html}{_esc(metric['label'])}</div>
                <div class="wd-metric-value">{_esc(metric['value'])}{unit_html}</div>
                {foot_html}
            </div>
            """
        )
    return compact(f'<div class="wd-metrics">{"".join(cells)}</div>')


def panel(title: str, body_html: str) -> str:
    return compact(f'<div class="wd-panel"><h3>{_esc(title)}</h3>{body_html}</div>')


def kv_table(rows: Iterable[Tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows
    )
    return compact(f'<table class="wd-table">{body}</table>')


def tag(text: str, kind: str = "measured") -> str:
    return f'<span class="wd-tag {kind}">{_esc(text)}</span>'


def legend(crop_label: str = "Soybean plant", weed_label: str = "Weed") -> str:
    return compact(f"""
    <div class="wd-legend">
        <div class="wd-legend-item">
            <span class="wd-legend-box" style="border-color:{PALETTE['crop']};
                background:{PALETTE['crop']}22;"></span>{_esc(crop_label)}
        </div>
        <div class="wd-legend-item">
            <span class="wd-legend-box" style="border-color:{PALETTE['weed']};
                background:{PALETTE['weed']}22;"></span>{_esc(weed_label)}
        </div>
    </div>
    """)


def pipeline(steps: Sequence[dict]) -> str:
    cells = []
    for index, step in enumerate(steps, start=1):
        state = step.get("state", "")
        state_class = f" {state}" if state else ""
        cells.append(
            f"""
            <div class="wd-step{state_class}">
                <div class="wd-step-index">Step {index}</div>
                <div class="wd-step-name">{_esc(step['name'])}</div>
                <div class="wd-step-detail">{_esc(step['detail'])}</div>
            </div>
            """
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
            f'<div class="wd-model-metric"><span>{_esc(metric_label)}</span>'
            f"<span>{_esc(metric_value)}</span></div>"
        )
    else:
        metric_html = (
            '<div class="wd-model-metric"><span class="wd-model-missing">'
            "No published value in benchmarks.json</span><span></span></div>"
        )
    selected_class = " selected" if selected else ""
    return compact(f"""
    <div class="wd-model{selected_class}">
        <div class="wd-model-head">
            <span class="wd-model-name">{_esc(name)}</span>
            <span class="wd-model-year">{_esc(year)}</span>
        </div>
        <div class="wd-model-arch">{_esc(architecture)}</div>
        {metric_html}
    </div>
    """)


def section_title(title: str, note: str = "") -> str:
    note_html = f'<div class="wd-section-note">{_esc(note)}</div>' if note else ""
    return compact(f'<div class="wd-section-title">{_esc(title)}</div>{note_html}')


def bullet_panel(title: str, items: Sequence[str], intro: str = "") -> str:
    intro_html = f"<p>{_esc(intro)}</p>" if intro else ""
    items_html = "".join(f"<li>{_esc(item)}</li>" for item in items)
    return panel(title, f"{intro_html}<ul>{items_html}</ul>")
