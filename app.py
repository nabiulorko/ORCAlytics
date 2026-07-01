"""
ORCAlytics — Frontier Molecular Orbital Analyzer
Parses ORCA .out files and computes Conceptual DFT global reactivity descriptors
from HOMO / LUMO orbital energies.
"""

import re
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="ORCAlytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# DESIGN TOKENS
# --------------------------------------------------------------------------------------
BG_VOID     = "#0a0e14"
BG_PANEL    = "#0e141d"
BG_INSET    = "#0c1119"
GRID_LINE   = "#1b2531"
BORDER      = "#26313f"
TXT_PRIMARY = "#dce4ee"
TXT_SECOND  = "#7d8ba0"
TXT_DIM     = "#48566a"
OCCUPIED    = "#4fc3f7"   # HOMO — cyan
VIRTUAL     = "#ff8a4c"   # LUMO — amber
PHOSPHOR    = "#39ff9d"   # signature readout green
DANGER      = "#ff5f6d"

# --------------------------------------------------------------------------------------
# GLOBAL STYLE
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
    }}
    code, .mono {{ font-family: 'IBM Plex Mono', monospace; }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .stApp {{
        background:
            linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px) 0 0 / 100% 28px,
            linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px) 0 0 / 28px 100%,
            {BG_VOID};
    }}

    /* ---------------- Console header ---------------- */
    .console-header {{
        border: 1px solid {BORDER};
        background: {BG_PANEL};
        padding: 1.1rem 1.5rem;
        margin-bottom: 1.4rem;
        position: relative;
    }}
    .console-header::before {{
        content: "";
        position: absolute; top: 0; left: 0;
        width: 3px; height: 100%;
        background: {PHOSPHOR};
        box-shadow: 0 0 8px {PHOSPHOR};
    }}
    .console-row {{
        display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 0.6rem;
    }}
    .console-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: {TXT_PRIMARY};
        letter-spacing: 0.5px;
    }}
    .console-title .diamond {{ color: {PHOSPHOR}; margin-right: 0.5rem; }}
    .console-tag {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: {TXT_DIM};
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    .console-status {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: {PHOSPHOR};
        display: flex; align-items: center; gap: 0.4rem;
    }}
    .status-dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: {PHOSPHOR};
        box-shadow: 0 0 6px {PHOSPHOR};
        display: inline-block;
        animation: pulse 1.8s infinite ease-in-out;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.35; }}
    }}
    .console-sub {{
        margin-top: 0.55rem;
        font-size: 0.92rem;
        color: {TXT_SECOND};
        max-width: 780px;
        line-height: 1.5;
    }}
    .console-formula-row {{
        margin-top: 0.85rem;
        display: flex; gap: 1.6rem; flex-wrap: wrap;
        border-top: 1px dashed {BORDER};
        padding-top: 0.7rem;
    }}
    .console-formula-row .f {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: {TXT_SECOND};
    }}
    .console-formula-row .f b {{ color: {TXT_PRIMARY}; font-weight: 600; }}

    /* ---------------- Panels ---------------- */
    .panel {{
        border: 1px solid {BORDER};
        background: {BG_PANEL};
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
    }}
    .panel-head {{
        display: flex; align-items: center; justify-content: space-between;
        border-bottom: 1px solid {GRID_LINE};
        padding-bottom: 0.55rem;
        margin-bottom: 0.85rem;
    }}
    .panel-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        color: {TXT_PRIMARY};
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    .panel-title::before {{ content: "▍"; color: {PHOSPHOR}; margin-right: 0.4rem; }}
    .panel-index {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: {TXT_DIM};
    }}
    .panel p, .panel li {{
        color: {TXT_SECOND};
        font-size: 0.88rem;
        line-height: 1.55;
    }}
    .panel code {{ color: {PHOSPHOR}; background: {BG_INSET}; padding: 0.1rem 0.35rem; border: 1px solid {GRID_LINE}; }}

    /* ---------------- Sidebar ---------------- */
    section[data-testid="stSidebar"] {{
        background: {BG_PANEL};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{ color: {TXT_SECOND}; }}
    section[data-testid="stSidebar"] h3 {{
        font-family: 'IBM Plex Mono', monospace;
        color: {TXT_PRIMARY} !important;
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    /* ---------------- File uploader ---------------- */
    section[data-testid="stFileUploaderDropzone"] {{
        background: {BG_INSET};
        border: 1px dashed {BORDER};
        border-radius: 0;
    }}

    /* ---------------- Buttons ---------------- */
    .stButton>button, .stDownloadButton>button {{
        background: {BG_INSET};
        color: {PHOSPHOR};
        border: 1px solid {PHOSPHOR};
        border-radius: 0;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
        padding: 0.5rem 1.2rem;
        transition: background 0.15s ease, color 0.15s ease;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        background: {PHOSPHOR};
        color: {BG_VOID};
    }}

    /* ---------------- Metrics ---------------- */
    div[data-testid="stMetric"] {{
        background: {BG_INSET};
        border: 1px solid {GRID_LINE};
        border-radius: 0;
        padding: 0.8rem 1rem;
        border-left: 2px solid {PHOSPHOR};
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TXT_DIM} !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TXT_PRIMARY} !important;
        font-family: 'IBM Plex Mono', monospace;
    }}

    /* ---------------- Data-sheet table ---------------- */
    .table-wrap {{
        overflow-x: auto;
        border: 1px solid {BORDER};
    }}
    table.results {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        font-family: 'IBM Plex Mono', monospace;
    }}
    table.results thead th {{
        background: {BG_INSET};
        color: {TXT_SECOND};
        font-weight: 600;
        font-size: 0.68rem;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        padding: 0.65rem 0.8rem;
        border: 1px solid {GRID_LINE};
        text-align: center;
        white-space: nowrap;
    }}
    table.results tbody td {{
        padding: 0.55rem 0.8rem;
        text-align: right;
        color: {TXT_PRIMARY};
        border: 1px solid {GRID_LINE};
        white-space: nowrap;
    }}
    table.results tbody td:first-child {{
        text-align: left;
        color: {PHOSPHOR};
        font-weight: 600;
    }}
    table.results tbody tr:hover td {{
        background: rgba(57, 255, 157, 0.05);
    }}

    /* ---------------- Footer ---------------- */
    .device-footer {{
        margin-top: 2rem;
        border: 1px solid {BORDER};
        border-top: 2px solid {PHOSPHOR};
        background: {BG_PANEL};
        padding: 1.1rem 1.5rem;
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 1rem;
    }}
    .footer-brand {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        color: {TXT_PRIMARY};
        letter-spacing: 0.5px;
    }}
    .footer-brand span {{ color: {PHOSPHOR}; }}
    .footer-meta {{ text-align: right; font-family: 'IBM Plex Mono', monospace; }}
    .footer-meta .credit {{ color: {TXT_PRIMARY}; font-size: 0.82rem; }}
    .footer-meta .credit a {{ color: {PHOSPHOR}; text-decoration: none; }}
    .footer-meta .stack {{ color: {TXT_DIM}; font-size: 0.72rem; margin-top: 0.15rem; }}
    .footer-meta .copyright {{ color: {TXT_DIM}; font-size: 0.68rem; margin-top: 0.15rem; opacity: 0.7; }}

    ::-webkit-scrollbar {{ height: 8px; width: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; }}
    ::-webkit-scrollbar-track {{ background: {BG_VOID}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="console-header">
        <div class="console-row">
            <div>
                <div class="console-title"><span class="diamond">◈</span>ORCAlytics</div>
                <div class="console-tag">Frontier Molecular Orbital Analyzer · Conceptual DFT Module</div>
            </div>
            <div class="console-status"><span class="status-dot"></span>PARSER READY</div>
        </div>
        <div class="console-sub">
            Reads the <code>ORBITAL ENERGIES</code> block of an ORCA <code>.out</code> file, resolves E<sub>HOMO</sub>
            and E<sub>LUMO</sub> from the SCF occupation numbers, and derives the full Parr–Pearson set of global
            reactivity descriptors — gap, ionization potential, electron affinity, electronegativity, chemical
            potential, hardness, softness, electrophilicity index.
        </div>
        <div class="console-formula-row">
            <div class="f">ΔE<sub>gap</sub> = <b>E<sub>LUMO</sub> − E<sub>HOMO</sub></b></div>
            <div class="f">I = <b>−E<sub>HOMO</sub></b> &nbsp;·&nbsp; A = <b>−E<sub>LUMO</sub></b></div>
            <div class="f">χ = <b>(I + A) / 2</b> &nbsp;·&nbsp; μ = <b>−χ</b></div>
            <div class="f">η = <b>ΔE<sub>gap</sub> / 2</b> &nbsp;·&nbsp; S = <b>1 / η</b></div>
            <div class="f">ω = <b>χ² / 2η</b></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# PARSING LOGIC
# --------------------------------------------------------------------------------------
ORBITAL_BLOCK_RE = re.compile(
    r"ORBITAL ENERGIES\s*\n-+\s*\n\s*NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s*\n((?:.*\n)+?)\n",
    re.MULTILINE,
)
ORBITAL_BLOCK_ALT_RE = re.compile(
    r"ORBITAL ENERGIES\s*\n-+\s*\n\s*NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s*\n((?:\s*\d+.*\n)+)"
)
ORBITAL_LINE_RE = re.compile(r"^\s*(\d+)\s+([\d.\-]+)\s+([\-\d.]+)\s+([\-\d.]+)\s*$")


def parse_orca_out(text: str):
    """Extract (HOMO_eV, LUMO_eV) from the FINAL ORBITAL ENERGIES block in an ORCA .out file."""
    blocks = list(ORBITAL_BLOCK_RE.finditer(text)) or list(ORBITAL_BLOCK_ALT_RE.finditer(text))
    if not blocks:
        return None

    last_block = blocks[-1].group(1)
    rows = []
    for line in last_block.splitlines():
        m = ORBITAL_LINE_RE.match(line)
        if m:
            idx, occ, e_eh, e_ev = m.groups()
            rows.append((int(idx), float(occ), float(e_eh), float(e_ev)))
    if not rows:
        return None

    homo_e, homo_idx = None, None
    for i, (idx, occ, e_eh, e_ev) in enumerate(rows):
        if occ > 0.0:
            homo_e, homo_idx = e_ev, i
        else:
            break
    if homo_e is None or homo_idx + 1 >= len(rows):
        return None

    lumo_e = rows[homo_idx + 1][3]
    return homo_e, lumo_e


def compute_descriptors(homo, lumo):
    gap = lumo - homo
    ip = -homo
    ea = -lumo
    chi = (ip + ea) / 2
    mu = -chi
    eta = gap / 2
    sigma = 1 / eta if eta != 0 else float("inf")
    omega = (chi ** 2) / (2 * eta) if eta != 0 else float("inf")
    return {
        "HOMO (eV)": homo, "LUMO (eV)": lumo, "L–H gap (eV)": gap,
        "I (eV)": ip, "A (eV)": ea, "μ (eV)": mu, "χ (eV)": chi,
        "η (eV)": eta, "σ (eV)⁻¹": sigma, "ω (eV)": omega,
    }


# --------------------------------------------------------------------------------------
# FMO ENERGY-LEVEL DIAGRAM (matplotlib, instrument-panel styling)
# --------------------------------------------------------------------------------------
def render_fmo_diagram(df: pd.DataFrame):
    n = len(df)
    fig_w = max(6.0, 1.55 * n + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=170)
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)

    mono = {"family": "monospace"}

    all_vals = pd.concat([df["HOMO (eV)"], df["LUMO (eV)"]])
    y_min, y_max = all_vals.min(), all_vals.max()
    pad = max(1.0, (y_max - y_min) * 0.25)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(-0.6, n - 0.4)

    ax.grid(axis="y", color=GRID_LINE, linewidth=0.7, linestyle=(0, (1, 3)))
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    half_w = 0.32
    for i, row in df.reset_index(drop=True).iterrows():
        homo, lumo, gap = row["HOMO (eV)"], row["LUMO (eV)"], row["L–H gap (eV)"]

        # HOMO level (occupied, solid, filled electrons)
        ax.hlines(homo, i - half_w, i + half_w, color=OCCUPIED, linewidth=3.2, zorder=3)
        ax.text(i, homo - pad * 0.14, f"↑↓", ha="center", va="top",
                 color=OCCUPIED, fontsize=9, fontdict=mono)
        ax.text(i + half_w + 0.05, homo, f"{homo:.2f}", va="center", ha="left",
                 color=OCCUPIED, fontsize=8, fontdict=mono)

        # LUMO level (virtual, dashed, empty)
        ax.hlines(lumo, i - half_w, i + half_w, color=VIRTUAL, linewidth=3.2,
                   linestyle=(0, (4, 2)), zorder=3)
        ax.text(i + half_w + 0.05, lumo, f"{lumo:.2f}", va="center", ha="left",
                 color=VIRTUAL, fontsize=8, fontdict=mono)

        # gap bracket
        ax.annotate(
            "", xy=(i - half_w - 0.08, lumo), xytext=(i - half_w - 0.08, homo),
            arrowprops=dict(arrowstyle="<->", color=TXT_DIM, linewidth=0.9),
        )
        ax.text(i - half_w - 0.14, (homo + lumo) / 2, f"{gap:.2f}",
                 va="center", ha="right", color=PHOSPHOR, fontsize=8, fontdict=mono, rotation=90)

    ax.set_xticks(range(n))
    ax.set_xticklabels(df["Molecule"].tolist(), color=TXT_SECOND, fontsize=9, fontdict=mono)
    ax.tick_params(axis="y", colors=TXT_SECOND, labelsize=8)
    ax.set_ylabel("Energy (eV)", color=TXT_SECOND, fontsize=9, fontdict=mono)

    # legend
    ax.hlines([], [], [], color=OCCUPIED, label="HOMO (occupied)")
    ax.hlines([], [], [], color=VIRTUAL, linestyle=(0, (4, 2)), label="LUMO (virtual)")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=TXT_SECOND)
    for text in leg.get_texts():
        text.set_fontfamily("monospace")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# SIDEBAR — CONTROL PANEL
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ▍ Control Panel")
    st.markdown(
        "<p style='font-size:0.82rem; color:#7d8ba0;'>ORCAlytics locates the last "
        "<code>ORBITAL ENERGIES</code> table in the file (final SCF), identifies HOMO as the "
        "highest orbital with non-zero occupation, and LUMO as the next orbital above it.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    decimals = st.slider("Decimal precision", 2, 6, 4)
    show_diagram = st.checkbox("Show FMO energy-level diagram", value=True)
    heatmap_on = st.checkbox("Heatmap gap / ω columns", value=True)
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.72rem; color:#48566a;'>MODULE: fmo_parser.py<br>"
        "STATUS: v1.1 · stable</p>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# UPLOAD PANEL
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="panel">
        <div class="panel-head">
            <div class="panel-title">Sample Input</div>
            <div class="panel-index">01</div>
        </div>
        <p>Load one or more ORCA <code>.out</code> files. Molecule labels default to the file name and can be edited after parsing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
uploaded_files = st.file_uploader(
    "Upload .out files", type=["out", "txt"], accept_multiple_files=True, label_visibility="collapsed"
)

# --------------------------------------------------------------------------------------
# PROCESS
# --------------------------------------------------------------------------------------
if uploaded_files:
    results, failed = [], []
    for f in uploaded_files:
        raw = f.read().decode("utf-8", errors="ignore")
        parsed = parse_orca_out(raw)
        name = f.name.rsplit(".", 1)[0]
        if parsed is None:
            failed.append(f.name)
            continue
        homo, lumo = parsed
        results.append({"Molecule": name, **compute_descriptors(homo, lumo)})

    if failed:
        st.warning(
            f"No ORBITAL ENERGIES block found in: {', '.join(failed)}. "
            "Confirm these are valid ORCA SCF/DFT output files."
        )

    if results:
        df = pd.DataFrame(results)

        with st.expander("✏  Rename molecules", expanded=False):
            for i in range(len(df)):
                df.loc[i, "Molecule"] = st.text_input(
                    f"Molecule {i+1}", value=df.loc[i, "Molecule"], key=f"name_{i}"
                )

        # ---- Summary readout ----
        st.markdown(
            """
            <div class="panel" style="margin-bottom:0.6rem;">
                <div class="panel-head">
                    <div class="panel-title">Batch Readout</div>
                    <div class="panel-index">02</div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N molecules", len(df))
        c2.metric("Mean gap (eV)", f"{df['L–H gap (eV)'].mean():.{decimals}f}")
        c3.metric("Mean η (eV)", f"{df['η (eV)'].mean():.{decimals}f}")
        c4.metric("Mean ω (eV)", f"{df['ω (eV)'].mean():.{decimals}f}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- FMO diagram ----
        if show_diagram:
            st.markdown(
                """
                <div class="panel">
                    <div class="panel-head">
                        <div class="panel-title">Frontier Orbital Energy-Level Diagram</div>
                        <div class="panel-index">03</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            fig = render_fmo_diagram(df)
            st.pyplot(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---- Data-sheet table ----
        panel_n = "04" if show_diagram else "03"
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-head">
                    <div class="panel-title">Descriptor Data Sheet</div>
                    <div class="panel-index">{panel_n}</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        numeric_cols = [c for c in df.columns if c != "Molecule"]

        if heatmap_on:
            styler = (
                df.style
                .format({c: f"{{:.{decimals}f}}" for c in numeric_cols})
                .background_gradient(subset=["L–H gap (eV)"], cmap="Blues")
                .background_gradient(subset=["ω (eV)"], cmap="Oranges")
                .hide(axis="index")
            )
            table_html = styler.to_html(table_attributes='class="results"')
        else:
            df_display = df.copy()
            for c in numeric_cols:
                df_display[c] = df_display[c].map(lambda x: f"{x:.{decimals}f}")
            table_html = df_display.to_html(index=False, classes="results", border=0)

        st.markdown(f'<div class="table-wrap">{table_html}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Export ----
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "↓ EXPORT CSV", data=csv_buf.getvalue(),
            file_name="orcalytics_results.csv", mime="text/csv",
        )
    else:
        st.error("No valid orbital energy data could be parsed from the uploaded file(s).")

else:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-head">
                <div class="panel-title">Procedure</div>
                <div class="panel-index">—</div>
            </div>
            <p>
            1. Run a single-point or geometry calculation in ORCA.<br>
            2. Upload the resulting <code>.out</code> file(s) in the panel above.<br>
            3. ORCAlytics resolves E<sub>HOMO</sub> / E<sub>LUMO</sub> from the final SCF orbital
               occupation table and derives the full descriptor set — no manual input required.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="device-footer">
        <div class="footer-brand">◈ ORCA<span>lytics</span></div>
        <div class="footer-meta">
            <div class="credit">Designed &amp; developed by <a href="#" target="_blank">Nabiul Orko</a></div>
            <div class="stack">Python · Streamlit · Pandas · Matplotlib · ORCA Output Parser</div>
            <div class="copyright">© 2026 All Rights Reserved</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
