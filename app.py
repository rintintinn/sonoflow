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
from version import __version__


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

# Header with logo
import os
logo_path = os.path.join(os.path.dirname(__file__), 'UrologyMYLogo.png')
logo_base64 = ""
if os.path.exists(logo_path):
    with open(logo_path, 'rb') as f:
        logo_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div class="main-header">
    {'<img src="data:image/png;base64,' + logo_base64 + '" style="width: 120px; margin-bottom: 1rem; border-radius: 12px;">' if logo_base64 else ''}
    <h1>Acoustic Uroflowmetry</h1>
    <p class="subtitle">Audio-based flow curve analysis</p>
    <p style="color: #64748b; font-size: 0.8rem;">v{__version__}</p>
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
                    # Process with debug mode for comparison graph
                    processor = AudioProcessor()
                    result = processor.process(audio_data, sample_rate, volume_ml, debug=True)
                    
                    # Generate main flow curve graph
                    graph_bytes, timestamp_str = generate_clinical_graph(result)
                    
                    # Generate detection comparison graph
                    comparison_bytes = None
                    flow_stability_bytes = None
                    if result.debug_data:
                        from debug_plots import plot_detection_comparison, plot_flow_stability
                        d = result.debug_data
                        comparison_bytes = plot_detection_comparison(
                            time_axis=d['time_axis_full'],
                            energy=d['energy'],
                            noise_floor=d['noise_floor'],
                            fixed_start_idx=d['fixed_start_idx'],
                            fixed_end_idx=d['fixed_end_idx'],
                            alt_start_idx=d['alt_start_idx'],
                            alt_end_idx=d['alt_end_idx']
                        )
                        
                        # Generate flow stability graph (slope-stabilized Qmax)
                        if 'stable_mask' in d:
                            flow_stability_bytes = plot_flow_stability(
                                time_axis=d['time_axis_trimmed'],
                                flow_rate=d['flow_rate_minimal'],
                                stable_mask=d['stable_mask'],
                                qmax_standard=result.qmax,
                                qmax_slope_stabilized=d['qmax_slope_stabilized'],
                                slope_threshold=d['slope_threshold']
                            )
                    
                    # Store in session state
                    st.session_state.result = result
                    st.session_state.graph_bytes = graph_bytes
                    st.session_state.comparison_bytes = comparison_bytes
                    st.session_state.flow_stability_bytes = flow_stability_bytes
                    st.session_state.timestamp = timestamp_str
                    
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")

# Display results
if "result" in st.session_state and st.session_state.result:
    result = st.session_state.result
    
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Display quality warnings if any
    if result.quality_warning:
        st.warning(f"⚠️ **Quality Warning**: {result.quality_warning}")
    
    # Display SNR indicator
    snr_color = "#22c55e" if result.snr_db >= 15 else "#f59e0b" if result.snr_db >= 10 else "#ef4444"
    snr_status = "Good" if result.snr_db >= 15 else "Acceptable" if result.snr_db >= 10 else "Low"
    st.markdown(f"""
    <div style="display: flex; gap: 1rem; margin-bottom: 1rem; align-items: center;">
        <span style="color: #94a3b8;">Signal Quality:</span>
        <span style="background: {snr_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem;">
            SNR: {result.snr_db:.1f} dB ({snr_status})
        </span>
        <span style="color: #64748b; font-size: 0.8rem;">Sample Rate: {result.sample_rate/1000:.1f} kHz</span>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Detection comparison graph (research/debug display)
    if "comparison_bytes" in st.session_state and st.session_state.comparison_bytes:
        with st.expander("🔬 Detection Method Comparison", expanded=False):
            st.markdown("""
            <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 10px;">
            Comparison of voiding detection methods: Fixed threshold (green) vs Otsu+Changepoint adaptive (purple).
            </p>
            """, unsafe_allow_html=True)
            st.image(st.session_state.comparison_bytes, use_container_width=True)
            
            # Show timing comparison if available
            if result.alt_voiding_time is not None:
                col_fixed, col_alt = st.columns(2)
                with col_fixed:
                    st.metric("Fixed Threshold", f"{result.voiding_time:.1f}s")
                with col_alt:
                    delta = result.alt_voiding_time - result.voiding_time
                    st.metric("Otsu+Changepoint", f"{result.alt_voiding_time:.1f}s", 
                             delta=f"{delta:+.1f}s")
    
    # Flow stability graph (slope-stabilized Qmax)
    if "flow_stability_bytes" in st.session_state and st.session_state.flow_stability_bytes:
        with st.expander("📈 Slope-Stabilized Qmax Analysis", expanded=False):
            st.markdown("""
            <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 10px;">
            Qmax measured only in stable flow regions (|dQ/dt| below adaptive threshold for ≥200ms).
            Green regions = stable flow, Red regions = unstable (high slope).
            </p>
            """, unsafe_allow_html=True)
            st.image(st.session_state.flow_stability_bytes, use_container_width=True)
            
            # Show Qmax comparison if available
            if result.qmax_slope_stabilized is not None:
                col_std, col_slope = st.columns(2)
                with col_std:
                    st.metric("Qmax Standard", f"{result.qmax:.1f} ml/s")
                with col_slope:
                    delta = result.qmax_slope_stabilized - result.qmax
                    st.metric("Qmax Slope-Stabilized", f"{result.qmax_slope_stabilized:.1f} ml/s",
                             delta=f"{delta:+.1f} ml/s")

# Footer - always display
st.markdown("""
<div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #1e293b; text-align: center; color: #94a3b8; font-size: 0.9rem;">
    <p style="margin-bottom: 5px;">Developed by <strong>Dr Badrulhisham Bahadzor</strong></p>
    <p style="margin: 0;">
        <a href="mailto:drbadrul@urology.my" style="color: #38bdf8; text-decoration: none;">drbadrul@urology.my</a> • 
        <a href="https://www.urology.my" target="_blank" style="color: #38bdf8; text-decoration: none;">www.urology.my</a>
    </p>
</div>
""", unsafe_allow_html=True)
