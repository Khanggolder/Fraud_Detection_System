import streamlit as st
import pandas as pd
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor

#streamlit run C:\Users\ad\Downloads\codepython\project\fraud_detection_system\app_c.py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.c_features import extract_c_features
from src.c_ai_detector import CAIDetector, C_SIGNALS, _C_TOKEN_SIGNAL_WEIGHT

try:
    from src.ml_detector import MLDetector
    _ML_AVAILABLE_C = MLDetector(lang="c").is_loaded
except Exception:
    _ML_AVAILABLE_C = False

st.set_page_config(
    layout="wide",
    page_title="C AI Code Detector",
    page_icon="",
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetric"] label {
        color: #a8b2d1 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ccd6f6 !important;
    }
</style>
""", unsafe_allow_html=True)

MAX_BATCH_WORKERS = 4

@st.cache_resource
def load_detector(use_ppl):
    return CAIDetector(use_perplexity=use_ppl)

@st.cache_resource
def load_ml_detector():
    return MLDetector(lang="c")


def _analyze_one(detector, name, code, threshold):
    try:
        result = detector.analyze(code, threshold=threshold)
        return {
            "File": name,
            "AI Score": result["score"],
            "p(AI)": f'{result["p_ai"]:.3f}',
            "Flag": " AI" if result["flag"] else " Human",
            "Top Signal": result["signals"][0] if result["signals"] else "—",
            "_result": result,
            "_code": code,
        }
    except Exception as e:
        return {
            "File": name,
            "AI Score": 0,
            "p(AI)": "0.000",
            "Flag": " Error",
            "Top Signal": str(e)[:60],
            "_result": {"score": 0, "p_ai": 0, "flag": False, "signals": [], "details": {}},
            "_code": code,
        }


st.title(" C AI Code Detector")
st.caption("AST-based analysis for detecting AI-generated C code")

st.sidebar.header(" Configuration")
ai_threshold = st.sidebar.slider(
    "AI Detection Threshold",
    min_value=0.0, max_value=1.0, value=0.35, step=0.05,
    help="Files with p(AI) >= threshold are flagged.",
)

st.sidebar.markdown("---")
st.sidebar.subheader(" Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload C Files",
    type=["c", "h"],
    accept_multiple_files=True,
)

use_perplexity = st.sidebar.checkbox(
    "Enable Perplexity Model (Qwen2.5-Coder)",
    value=False,
    help="Uses Qwen/Qwen2.5-Coder-0.5B for token distribution analysis.",
)

st.sidebar.markdown("---")
st.sidebar.subheader(" Detection Mode")
mode_options = ["Rule-Based (Signals)"]
if _ML_AVAILABLE_C:
    mode_options.append("ML Model (XGBoost/RF)")
detection_mode = st.sidebar.radio(
    "Choose detection engine",
    mode_options,
    index=0,
)

use_ml = detection_mode == "ML Model (XGBoost/RF)"
if use_ml:
    detector = load_ml_detector()
    st.sidebar.success(f"ML Model loaded — {detector.get_model_info().get('model_type', 'unknown').upper()}")
else:
    detector = load_detector(use_perplexity)

file_contents = {}
if uploaded_files:
    for f in uploaded_files:
        content = f.read().decode("utf-8", errors="replace")
        file_contents[f.name] = content

if file_contents:
    file_names = list(file_contents.keys())
    n_files = len(file_names)

    st.info(f"Analyzing {n_files} file(s)...")
    progress = st.progress(0)
    start_time = time.time()

    results_list = []

    if n_files <= 10:
        for i, name in enumerate(file_names):
            entry = _analyze_one(detector, name, file_contents[name], ai_threshold)
            results_list.append(entry)
            progress.progress((i + 1) / n_files)
    else:
        st.toast(f"Batch mode: processing {n_files} files...")
        done_count = 0
        for i, name in enumerate(file_names):
            entry = _analyze_one(detector, name, file_contents[name], ai_threshold)
            results_list.append(entry)
            done_count += 1
            if done_count % max(1, n_files // 20) == 0 or done_count == n_files:
                progress.progress(done_count / n_files)

    elapsed = time.time() - start_time
    progress.empty()
    st.caption(f"Processed {n_files} files in {elapsed:.1f}s ({n_files/max(elapsed,0.01):.0f} files/sec)")

    flagged_list = [r for r in results_list if "AI" in r["Flag"] and "Human" not in r["Flag"]]
    clean_list = [r for r in results_list if r not in flagged_list]
    flagged = len(flagged_list)

    tab_overview, tab_ai_list, tab_detail = st.tabs(
        [f" Overview ({n_files})", f" AI Detected ({flagged})", " File Inspector"]
    )

    with tab_overview:
        st.subheader("Analysis Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Files", n_files)
        c2.metric("Flagged as AI", flagged)
        c3.metric("Clean", n_files - flagged)

        if flagged:
            st.warning(f" {flagged}/{n_files} file(s) flagged (threshold >= {ai_threshold:.0%})")
        else:
            st.success(f" No files flagged at threshold {ai_threshold:.0%}")

        st.markdown("---")

        df = pd.DataFrame(results_list)
        cols = ["File", "AI Score", "p(AI)", "Flag", "Top Signal"]

        def _hl(row):
            if "AI" in str(row["Flag"]) and "Human" not in str(row["Flag"]):
                return ["background-color: rgba(239,68,68,0.15)"] * len(row)
            return [""] * len(row)

        st.dataframe(df[cols].style.apply(_hl, axis=1), width='stretch', hide_index=True)

        if n_files > 1:
            st.markdown("#### Score Distribution")
            chart = pd.DataFrame({
                "File": [r["File"] for r in results_list],
                "Score": [r["AI Score"] for r in results_list],
            })
            st.bar_chart(chart.set_index("File"), height=300)

    with tab_ai_list:
        st.subheader("Files Detected as AI-Generated")

        if not flagged_list:
            st.success("No files were flagged as AI-generated.")
        else:
            st.error(f"{flagged} file(s) detected as AI-generated code:")

            ai_df = pd.DataFrame(flagged_list)
            ai_cols = ["File", "AI Score", "p(AI)", "Top Signal"]
            st.dataframe(
                ai_df[ai_cols].style.apply(
                    lambda row: ["background-color: rgba(239,68,68,0.2)"] * len(row), axis=1
                ),
                width='stretch',
                hide_index=True,
            )

            st.markdown("---")
            st.subheader("View AI File Code")
            ai_file_names = [r["File"] for r in flagged_list]
            selected_ai = st.selectbox("Select AI-flagged file to view:", ai_file_names, key="c_ai_sel")

            if selected_ai:
                entry = next(r for r in flagged_list if r["File"] == selected_ai)
                res = entry["_result"]

                c_s, c_p = st.columns(2)
                c_s.metric("AI Score", f'{res["score"]}/100')
                c_p.metric("p(AI)", f'{res["p_ai"]:.3f}')

                st.markdown("**Signals:**")
                for sig in res["signals"]:
                    st.markdown(f"- {sig}")

                st.markdown("---")
                st.code(entry["_code"], language="c", line_numbers=True)

    with tab_detail:
        st.subheader("File Inspector")
        sel = st.selectbox("Select file", file_names, key="c_detail_sel")

        if sel:
            entry = next(r for r in results_list if r["File"] == sel)
            res = entry["_result"]

            c_s, c_p, c_f, c_m = st.columns(4)
            c_s.metric("AI Score", f'{res["score"]}/100')
            c_p.metric("p(AI)", f'{res["p_ai"]:.3f}')
            c_f.metric("Verdict", " AI" if res["flag"] else " Human")
            mi = res["details"].get("maintainability_index", 0)
            c_m.metric("Maintainability", f'{mi:.1f}')

            st.markdown("---")
            col_sig, col_code = st.columns([1, 1])

            with col_sig:
                st.markdown("####  Signal Breakdown")
                bd = res["details"].get("signal_breakdown", {})
                wd = res["details"].get("weight_breakdown", {})
                if bd:
                    rows = []
                    for name, strength in sorted(bd.items(), key=lambda x: x[1], reverse=True):
                        w = wd.get(name, 0.0)
                        icon = "" if strength > 0.15 else ("" if strength < -0.1 else "")
                        rows.append({
                            "": icon,
                            "Signal": name,
                            "Strength": f"{strength:+.1%}",
                            "Weighted": f"{w:+.2f}",
                        })
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

                st.markdown("####  Signals")
                for sig in res["signals"]:
                    st.markdown(f"- {sig}")

            with col_code:
                st.markdown("####  Source Code")
                st.code(entry["_code"], language="c", line_numbers=True)

            st.markdown("---")
            col_feat, col_hal = st.columns(2)

            with col_feat:
                with st.expander(" All Features", expanded=False):
                    details = {k: v for k, v in res["details"].items()
                               if k not in ("signal_breakdown", "weight_breakdown")}
                    feat_rows = []
                    for k, v in sorted(details.items()):
                        if isinstance(v, bool):
                            val = str(v)
                        elif isinstance(v, float):
                            val = f"{v:.4f}"
                        else:
                            val = str(v)
                        feat_rows.append({"Feature": k, "Value": val})
                    st.dataframe(pd.DataFrame(feat_rows), width='stretch', hide_index=True, height=400)

            with col_hal:
                with st.expander(" Halstead & Complexity", expanded=False):
                    h_keys = [
                        ("halstead_vocabulary", "Vocabulary"),
                        ("halstead_length", "Length"),
                        ("halstead_volume", "Volume"),
                        ("halstead_difficulty", "Difficulty"),
                        ("halstead_effort", "Effort"),
                        ("halstead_bugs", "Bug Estimate"),
                        ("maintainability_index", "Maintainability Index"),
                        ("cyclomatic_branches", "Cyclomatic Branches"),
                        ("avg_cyclomatic", "Avg Cyclomatic"),
                        ("max_nesting", "Max Nesting"),
                        ("avg_nesting", "Avg Nesting"),
                    ]
                    h_rows = []
                    for key, label in h_keys:
                        val = res["details"].get(key, "—")
                        if isinstance(val, float):
                            val = f"{val:.2f}"
                        h_rows.append({"Metric": label, "Value": str(val)})
                    st.dataframe(pd.DataFrame(h_rows), width='stretch', hide_index=True)

else:
    st.markdown("###  Welcome")
    st.markdown(
        "Upload C files in the sidebar to start analysis.\n\n"
        "**Features:**\n"
        "-  **AST-based** analysis via Tree-sitter C parser\n"
        "-  **10 signal functions** covering comments, naming, formatting, memory, etc.\n"
        "-  **Halstead metrics** & Maintainability Index\n"
        "-  Sub-second per-file analysis\n"
    )

    st.markdown("---")
    st.markdown("#### Signal Weights")
    all_sigs = list(C_SIGNALS) + [("Token Distribution (Code Model)", None, _C_TOKEN_SIGNAL_WEIGHT)]
    sig_data = [{"Signal": name, "Weight": w} for name, _, w in all_sigs]
    st.dataframe(pd.DataFrame(sig_data), width='stretch', hide_index=True)
