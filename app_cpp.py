import streamlit as st
import pandas as pd
import sys
import os
import time

#streamlit run C:\Users\ad\Downloads\codepython\project\fraud_detection_system\app_cpp.py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cpp_features import extract_cpp_ast_features
from src.cpp_ai_detector import CppAIDetector, SIGNALS, _TOKEN_SIGNAL_WEIGHT

try:
    from src.ml_detector import MLDetector
    _ML_AVAILABLE_CPP = MLDetector(lang="cpp").is_loaded
except Exception:
    _ML_AVAILABLE_CPP = False

st.set_page_config(
    layout="wide",
    page_title="C++ AI Code Detector",
)

st.markdown("""
<style>
    .stApp { }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
    }
    .signal-positive { color: #22c55e; }
    .signal-negative { color: #ef4444; }
    .signal-neutral { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(use_ppl):
    return CppAIDetector(use_perplexity=use_ppl)

@st.cache_resource
def load_ml_detector():
    return MLDetector(lang="cpp")


def _analyze_one(detector, name, code, threshold):
    try:
        result = detector.analyze(code, threshold=threshold)
        return {
            "File": name,
            "AI Score": result["score"],
            "p(AI)": f'{result["p_ai"]:.3f}',
            "Flag": "AI" if result["flag"] else "Human",
            "Top Signal": result["signals"][0] if result["signals"] else "—",
            "_result": result,
            "_code": code,
        }
    except Exception as e:
        return {
            "File": name,
            "AI Score": 0,
            "p(AI)": "0.000",
            "Flag": "Error",
            "Top Signal": str(e)[:60],
            "_result": {"score": 0, "p_ai": 0, "flag": False, "signals": [], "details": {}},
            "_code": code,
        }


st.title("C++ AI Code Detector")
st.caption("AST-based analysis with Qwen2.5-Coder perplexity scoring")

st.sidebar.header("Configuration")
ai_threshold = st.sidebar.slider(
    "AI Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05,
    help="Files with p(AI) >= threshold are flagged.",
)

use_perplexity = st.sidebar.checkbox(
    "Enable Perplexity Model (Qwen2.5-Coder)",
    value=False,
    help="Uses Qwen/Qwen2.5-Coder-0.5B for token distribution analysis.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload C++ Files",
    type=["cpp", "cc", "cxx", "c", "h", "hpp", "hxx"],
    accept_multiple_files=True,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Mode")
mode_options = ["Rule-Based (Signals)"]
if _ML_AVAILABLE_CPP:
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

    ai_results = []

    for i, name in enumerate(file_names):
        entry = _analyze_one(detector, name, file_contents[name], ai_threshold)
        ai_results.append(entry)
        if (i + 1) % max(1, n_files // 20) == 0 or (i + 1) == n_files:
            progress.progress((i + 1) / n_files)

    elapsed = time.time() - start_time
    progress.empty()
    st.caption(f"Processed {n_files} files in {elapsed:.1f}s ({n_files/max(elapsed,0.01):.0f} files/sec)")

    flagged_list = [r for r in ai_results if r["Flag"] == "AI"]
    flagged_count = len(flagged_list)

    tab_overview, tab_ai_list, tab_detail = st.tabs(
        [f"Overview ({n_files})", f"AI Detected ({flagged_count})", "File Inspector"]
    )

    with tab_overview:
        st.subheader("Analysis Results")

        col_total, col_flagged, col_clean = st.columns(3)
        col_total.metric("Total Files", n_files)
        col_flagged.metric("Flagged as AI", flagged_count)
        col_clean.metric("Clean", n_files - flagged_count)

        if flagged_count:
            st.warning(f"{flagged_count}/{n_files} file(s) flagged as likely AI-generated (threshold >= {ai_threshold:.0%})")
        else:
            st.success(f"No files flagged at threshold {ai_threshold:.0%}")

        st.markdown("---")

        df_ai = pd.DataFrame(ai_results)
        display_cols = ["File", "AI Score", "p(AI)", "Flag", "Top Signal"]

        def _highlight(row):
            if row["Flag"] == "AI":
                return ["background-color: rgba(239,68,68,0.15)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_ai[display_cols].style.apply(_highlight, axis=1),
            width='stretch',
            hide_index=True,
        )

        if n_files > 1:
            st.markdown("#### Score Distribution")
            chart_df = pd.DataFrame({
                "File": [r["File"] for r in ai_results],
                "Score": [r["AI Score"] for r in ai_results],
            })
            st.bar_chart(chart_df.set_index("File"), height=300)

    with tab_ai_list:
        st.subheader("Files Detected as AI-Generated")

        if not flagged_list:
            st.success("No files were flagged as AI-generated.")
        else:
            st.error(f"{flagged_count} file(s) detected as AI-generated code:")

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
            selected_ai = st.selectbox("Select AI-flagged file to view:", ai_file_names, key="cpp_ai_sel")

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
                st.code(entry["_code"], language="cpp", line_numbers=True)

    with tab_detail:
        st.subheader("File Inspector")
        selected_file = st.selectbox(
            "Select file to inspect",
            file_names,
            key="detail_file_select",
        )

        if selected_file:
            entry = next(r for r in ai_results if r["File"] == selected_file)
            res = entry["_result"]

            col_score, col_pai, col_flag, col_mi = st.columns(4)
            col_score.metric("AI Score", f'{res["score"]}/100')
            col_pai.metric("p(AI)", f'{res["p_ai"]:.3f}')
            col_flag.metric("Verdict", "AI" if res["flag"] else "Human")
            mi_val = res["details"].get("maintainability_index", 0)
            col_mi.metric("Maintainability", f'{mi_val:.1f}')

            st.markdown("---")

            col_sig, col_code = st.columns([1, 1])

            with col_sig:
                st.markdown("#### Signal Breakdown")
                breakdown = res["details"].get("signal_breakdown", {})
                weights = res["details"].get("weight_breakdown", {})

                if breakdown:
                    sig_rows = []
                    for name, strength in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
                        w = weights.get(name, 0.0)
                        if strength > 0.15:
                            icon = "+"
                        elif strength < -0.1:
                            icon = "-"
                        else:
                            icon = "o"
                        sig_rows.append({
                            "": icon,
                            "Signal": name,
                            "Strength": f"{strength:+.1%}",
                            "Weighted": f"{w:+.2f}",
                        })

                    sig_df = pd.DataFrame(sig_rows)
                    st.dataframe(sig_df, width='stretch', hide_index=True)

                st.markdown("#### Signals")
                for sig in res["signals"]:
                    st.markdown(f"- {sig}")

            with col_code:
                st.markdown("#### Source Code")
                st.code(entry["_code"], language="cpp", line_numbers=True)

            st.markdown("---")

            col_feat, col_halstead = st.columns(2)

            with col_feat:
                with st.expander("All Features", expanded=False):
                    details = {
                        k: v for k, v in res["details"].items()
                        if k not in ("signal_breakdown", "weight_breakdown")
                    }
                    feat_rows = []
                    for k, v in sorted(details.items()):
                        if isinstance(v, bool):
                            val = str(v)
                        elif isinstance(v, float):
                            val = f"{v:.4f}"
                        else:
                            val = str(v)
                        feat_rows.append({"Feature": k, "Value": val})

                    feat_df = pd.DataFrame(feat_rows)
                    st.dataframe(feat_df, width='stretch', hide_index=True, height=400)

            with col_halstead:
                with st.expander("Halstead & Complexity Metrics", expanded=False):
                    h_keys = [
                        ("halstead_vocabulary", "Vocabulary"),
                        ("halstead_length", "Length (N)"),
                        ("halstead_volume", "Volume (V)"),
                        ("halstead_difficulty", "Difficulty (D)"),
                        ("halstead_effort", "Effort (E)"),
                        ("halstead_bugs_estimate", "Bug Estimate (B)"),
                        ("halstead_unique_operators", "Unique Operators"),
                        ("halstead_unique_operands", "Unique Operands"),
                        ("halstead_total_operators", "Total Operators"),
                        ("halstead_total_operands", "Total Operands"),
                        ("maintainability_index", "Maintainability Index"),
                        ("cyclomatic_branch_count", "Cyclomatic Branches"),
                        ("avg_cyclomatic_per_function", "Avg Cyclomatic/Function"),
                        ("max_nesting_depth", "Max Nesting Depth"),
                        ("avg_nesting_depth", "Avg Nesting Depth"),
                    ]
                    h_rows = []
                    for key, label in h_keys:
                        val = res["details"].get(key, "—")
                        if isinstance(val, float):
                            val = f"{val:.2f}"
                        h_rows.append({"Metric": label, "Value": str(val)})

                    h_df = pd.DataFrame(h_rows)
                    st.dataframe(h_df, width='stretch', hide_index=True)

else:
    st.markdown("### Welcome")
    st.markdown(
        "Upload C++ files in the sidebar to start analysis.\n\n"
        "**Features:**\n"
        "- **AST-based** analysis via Tree-sitter (not regex)\n"
        "- **Qwen2.5-Coder** perplexity scoring (optional)\n"
        "- **13 signal functions** covering modernity, RAII, naming, formatting, etc.\n"
        "- **Halstead metrics** & Maintainability Index\n"
        "- Sub-second per-file analysis\n"
    )

    st.markdown("---")
    st.markdown("#### Signal Weights")
    all_sigs = list(SIGNALS) + [("Token Distribution (Code Model)", None, _TOKEN_SIGNAL_WEIGHT)]
    sig_data = [{"Signal": name, "Weight": w} for name, _, w in all_sigs]
    st.dataframe(pd.DataFrame(sig_data), width='stretch', hide_index=True)
