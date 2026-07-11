import streamlit as st
from datetime import datetime
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import time

# Your existing imports
from utils.pdf_reader import read_pdf
from utils.rag import create_vectorstore
from utils.ai_helper import ask_ai
from utils.validator import is_medical_document
from utils.report_classifier import detect_report_type

# 1. NEW IMPORT
from utils.evaluation_metrics import (
    log_performance,
    get_metrics_df,
    get_performance_summary
)

# ====================== SESSION STATE ======================
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_text" not in st.session_state:
    st.session_state.processed_text = None

if "processed_df" not in st.session_state:      
    st.session_state.processed_df = None

if "file_type" not in st.session_state:         
    st.session_state.file_type = None

if "report_type" not in st.session_state:
    st.session_state.report_type = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Groq Llama 3.3 70B"

# NEW: Performance Metrics
if "performance_logs" not in st.session_state:
    st.session_state.performance_logs = []

st.set_page_config(page_title="MediGuide AI", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap');
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 600;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    .upload-zone {
        border: 3px dashed #10b981;
        border-radius: 24px;
        padding: 4rem 2rem;
        text-align: center;
        background: linear-gradient(180deg, #f8fafc, #f0fdf4);
    }
    
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
    }
    
    .chat-bubble-user {
        background: #10b981;
        color: white;
        border-radius: 20px 20px 4px 20px;
        padding: 14px 18px;
        max-width: 75%;
        margin-left: auto;
    }
    
    .chat-bubble-assist {
        background: #f1f5f9;
        color: #0f172a;
        border-radius: 20px 20px 20px 4px;
        padding: 14px 18px;
        max-width: 75%;
    }
</style>
""", unsafe_allow_html=True)

# ====================== HELPER ======================
def get_report_type_display():
    rt = st.session_state.report_type
    if isinstance(rt, tuple):
        return rt[0] if rt and len(rt) > 0 else "Medical Report"
    return rt or "Medical Report"

# ====================== WELCOME PAGE ======================
if st.session_state.page == "welcome":
    st.markdown("<h1 class='main-header'>🏥 MediGuide AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.4rem; color:#64748b;'>AI Powered Medical Report Analysis & Intelligent Health Assistant</p>", unsafe_allow_html=True)
    
    if st.button("🚀 Get Started with MediGuide", type="primary", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()

# ====================== MAIN APP ======================
else:
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="margin:0; background: linear-gradient(90deg, #10b981, #3b82f6); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🏥 MediGuide
            </h2>
            <p style="color: #64748b;">AI Medical Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        menu = st.radio(
            "Navigation",
            ["🏠 Home", "📄 Summary Report", "📊 Report Analysis", "💬 AI Chat", "⚙ Settings"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        
        st.markdown("**🤖 Select AI Model**")
        model_options = ["Groq Llama 3.3 70B", "Qwen 2.5:3B", "DeepSeek R1:1.5B"]
        st.session_state.selected_model = st.selectbox(
            "Model", model_options, 
            index=model_options.index(st.session_state.selected_model),
            label_visibility="collapsed"
        )
        st.caption("Make sure Ollama is running for local models")

    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 2rem;">
        <div style="font-size: 2.2rem;">🏥</div>
        <div><h1 style="margin:0; font-size:2.1rem;">MediGuide AI</h1></div>
    </div>
    """, unsafe_allow_html=True)

    # ==================== HOME PAGE ====================
    if menu == "🏠 Home":
        st.markdown("### Upload Your Medical Document")
        col_center = st.columns([1, 2, 1])[1]
        with col_center:
            st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
            st.markdown("### ☁️ Drop your report here\nPDF • XLSX • XLS", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload", type=["pdf", "xlsx", "xls"], label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            with st.spinner("Validating & Processing..."):
                file_type = uploaded_file.name.split('.')[-1].lower()
                st.session_state.file_type = file_type

                if file_type == "pdf":
                    text = read_pdf(uploaded_file)
                    temp_df = None
                else:
                    temp_df = pd.read_excel(uploaded_file)
                    text = temp_df.to_string()

                is_medical, matched = is_medical_document(text)

                if is_medical:
                    st.session_state.processed_text = text
                    st.session_state.processed_df = temp_df if file_type != "pdf" else None
                    st.session_state.vectorstore = create_vectorstore(text)
                    st.session_state.report_type = detect_report_type(text)

                    st.success("✅ Document processed successfully!")
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("File", uploaded_file.name)
                    with c2: st.metric("Size", f"{uploaded_file.size/1024:.1f} KB")
                    with c3: st.metric("Type", get_report_type_display())
                else:
                    st.error("❌ This is not a valid medical document.")
                    st.session_state.processed_text = None
                    st.session_state.processed_df = None
                    st.session_state.vectorstore = None

    # ==================== SUMMARY REPORT ====================
    elif menu == "📄 Summary Report":
        st.markdown("### 📄 Patient Summary Report")
        if st.session_state.processed_text:
            report_date = datetime.now().strftime('%d %B %Y at %H:%M')
            display_type = get_report_type_display()
            
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f"<div class='metric-card'><p>Report Type</p><h3>{display_type}</h3></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='metric-card'><p>Generated</p><h3>{report_date}</h3></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='metric-card'><p>File Type</p><h3>{st.session_state.file_type.upper()}</h3></div>", unsafe_allow_html=True)

            preview = st.session_state.processed_text[:1500]
            st.text_area("Report Preview", preview + "...", height=400)

            summary_text = f"MEDIGUIDE AI SUMMARY\nGenerated: {report_date}\nType: {display_type}\n\n{preview}"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("Download PDF", summary_text, "summary.pdf", "application/pdf", use_container_width=True)
            with col2:
                st.download_button("Download TXT", summary_text, "summary.txt", "text/plain", use_container_width=True)
            with col3:
                st.download_button("Download Raw", st.session_state.processed_text, f"raw.{st.session_state.file_type}", "text/plain", use_container_width=True)
        else:
            st.warning("Please upload a report first.")

    # ==================== REPORT ANALYSIS ====================
    elif menu == "📊 Report Analysis":
        st.markdown("### 📊 Medical Data Analytics")
        if st.session_state.processed_df is not None:
            df = st.session_state.processed_df
            display_type = get_report_type_display()
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Rows", len(df))
            with c2: st.metric("Columns", len(df.columns))
            with c3: st.metric("Numeric Features", len(df.select_dtypes(include='number').columns))
            with c4: st.metric("Type", display_type)

            st.dataframe(df, use_container_width=True)

            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                chart_type = st.selectbox("Select Chart", ["Bar Chart", "Pie Chart", "Line Chart"])
                fig, ax = plt.subplots(figsize=(10, 6))
                if chart_type == "Bar Chart":
                    df.plot(kind='bar', x=df.columns[0], y=numeric_cols[0], ax=ax, color='#10b981')
                elif chart_type == "Pie Chart":
                    df[numeric_cols[0]].plot(kind='pie', ax=ax, autopct='%1.1f%%')
                else:
                    df.plot(kind='line', ax=ax, color='#10b981')
                st.pyplot(fig)

                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button("⬇️ Download Chart", buf, "chart.png", "image/png", use_container_width=True)
        else:
            st.info("Upload an Excel medical report to see analytics & charts.")

    # ==================== AI CHAT ====================
    elif menu == "💬 AI Chat":
        st.markdown("### 💬 AI Medical Assistant")
        st.caption(f"Current Model: **{st.session_state.selected_model}**")

        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-assist">{msg["content"]}</div>', unsafe_allow_html=True)

        if prompt := st.chat_input("Ask anything about your report..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner(f"Thinking with {st.session_state.selected_model}..."):
                start_time = time.time()
                if st.session_state.vectorstore:
                    response = ask_ai(st.session_state.vectorstore, prompt, st.session_state.selected_model)
                else:
                    response = "Please upload a medical report first."
                latency_ms = round((time.time() - start_time) * 1000, 2)

            # NEW: Log performance using the evaluation module
            log_performance(
                st.session_state.performance_logs,
                model=st.session_state.selected_model,
                question=prompt,
                response_time_ms=latency_ms,
                answer=response
            )

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    # ==================== SETTINGS ====================
    else:
        st.markdown("### ⚙️ Settings")
        tab1, tab2 = st.tabs(["Configuration", "📊 Performance Metrics"])
        
        with tab1:
            st.info("You can change AI Model from the sidebar.")
        
        with tab2:
            # NEW: Enhanced Performance Metrics UI
            df = get_metrics_df(st.session_state.performance_logs)

            if not df.empty:
                summary = get_performance_summary(df)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Average Response", f"{summary['avg_time']} ms")
                c2.metric("Total Queries", summary["total_queries"])
                c3.metric("Fastest Model", summary["fastest_model"])
                c4.metric("Avg Answer Length", f"{summary['avg_answer_length']} chars")

                st.markdown("### Response Time by Query")
                st.line_chart(df.set_index("timestamp")["response_time_ms"])

                st.markdown("### Average Response Time per Model")
                st.bar_chart(df.groupby("model")["response_time_ms"].mean())

                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download Performance Metrics",
                    csv,
                    "performance_metrics.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.info("No performance data available. Use AI Chat to generate logs.")

    st.caption("MediGuide AI v2.1 • Secure Medical Intelligence")
