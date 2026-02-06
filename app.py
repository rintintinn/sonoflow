"""
Acoustic Uroflowmetry - Streamlit App
Audio-based flow curve analysis
"""

import streamlit as st
import io
import base64
import librosa
from processor import AudioProcessor, validate_audio
from visualization import generate_clinical_graph


# Page config
st.set_page_config(
    page_title="Acoustic Uroflowmetry",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    .param-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .param-value {
        font-size: 2rem;
        font-weight: 700;
        color: #06b6d4;
    }
    .param-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
    }
    .disclaimer {
        color: #64748b;
        font-size: 0.85rem;
        text-align: center;
        padding: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
    }
    /* Hide number input spinner controls - enhanced */
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
        appearance: none !important;
        margin: 0 !important;
        display: none !important;
    }
    input[type=number] {
        -moz-appearance: textfield !important;
        appearance: textfield !important;
    }
    /* Additional Streamlit-specific selectors */
    [data-testid="stNumberInput"] input::-webkit-inner-spin-button,
    [data-testid="stNumberInput"] input::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>Acoustic Uroflowmetry</h1>
    <p class="subtitle">Audio-based flow curve analysis</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer at top
st.markdown("""
<div class="disclaimer">
    This tool performs signal processing only. It does not provide medical diagnoses or clinical advice.
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Upload audio recording",
    type=["wav", "mp3"],
    help="Drag and drop a WAV or MP3 file"
)

# Volume input and analyze button in same row
col1, col2 = st.columns([0.5, 0.5])

with col1:
    volume_ml = st.number_input(
        "Total Voided Volume (ml)",
        min_value=1,
        max_value=2000,
        value=None,
        step=1,
        placeholder="Enter volume",
        label_visibility="visible"
    )

with col2:
    # Add spacing to align button with input
    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
    analyze_disabled = uploaded_file is None or volume_ml is None
    analyze_clicked = st.button("🔬 Analyze Recording", disabled=analyze_disabled, use_container_width=True, type="primary")

# Process when button is clicked
if analyze_clicked:
    if uploaded_file and volume_ml:
        with st.spinner("Processing audio..."):
            try:
                # Read audio
                audio_bytes = uploaded_file.read()
                audio_buffer = io.BytesIO(audio_bytes)
                audio_data, sample_rate = librosa.load(audio_buffer, sr=None)
                duration = len(audio_data) / sample_rate
                
                # Validate
                error = validate_audio(duration, sample_rate, volume_ml)
                if error:
                    st.error(f"❌ {error}")
                else:
                    # Process
                    processor = AudioProcessor()
                    result = processor.process(audio_data, sample_rate, volume_ml)
                    
                    # Generate graph
                    graph_bytes, timestamp_str = generate_clinical_graph(result)
                    
                    # Store in session state
                    st.session_state.result = result
                    st.session_state.graph_bytes = graph_bytes
                    st.session_state.timestamp = timestamp_str
                    
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")

# Display results
if "result" in st.session_state and st.session_state.result:
    result = st.session_state.result
    
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Parameters in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="param-card">
            <div class="param-label">Qmax</div>
            <div class="param-value">{result.qmax:.1f}</div>
            <div class="param-label">ml/s</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="param-card">
            <div class="param-label">Qavg</div>
            <div class="param-value">{result.qavg:.1f}</div>
            <div class="param-label">ml/s</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="param-card">
            <div class="param-label">Voiding Time</div>
            <div class="param-value">{result.voiding_time:.1f}</div>
            <div class="param-label">seconds</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="param-card">
            <div class="param-label">Volume</div>
            <div class="param-value">{result.volume_ml:.0f}</div>
            <div class="param-label">ml</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Graph
    st.markdown("### Flow Curve")
    st.image(st.session_state.graph_bytes, use_container_width=True)
    
    # Download button with timestamped filename
    timestamp = st.session_state.get('timestamp', 'analysis')
    st.download_button(
        label="📥 Download Graph",
        data=st.session_state.graph_bytes,
        file_name=f"uroflowmetry_{timestamp}.png",
        mime="image/png",
        use_container_width=True
    )


