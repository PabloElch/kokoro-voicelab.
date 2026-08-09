import os
import tempfile
import numpy as np
import soundfile as sf
import streamlit as st
from kokoro_onnx import Kokoro

# 1. Page Configuration
st.set_page_config(
    page_title="Lencho x Latera Audio Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern White-Card UI Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #311042, #1e1b4b, #0284c7, #6366f1);
        background-size: 400% 400%;
        animation: geminiGradient 16s ease infinite;
    }
    @keyframes geminiGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * { color: #0f172a !important; }
    .hero-container {
        text-align: center;
        padding: 2rem 1.5rem;
        background: #ffffff;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 900;
        color: #0f172a;
        margin: 0;
    }
    .lencho-highlight {
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
    }
    .hero-subtitle {
        color: #475569;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background-color: #ffffff !important;
        border-radius: 18px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 1.5rem !important;
    }
    div[data-testid="stVerticalBlock"] > div[style*="border"] * {
        color: #0f172a !important;
    }
    .stTextArea textarea {
        color: #0f172a !important;
        background-color: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Memory-Optimized Engine Initializer
@st.cache_resource(show_spinner=True)
def get_kokoro_engine():
    model_path = "kokoro-v1.0.onnx"
    voices_path = "voices-v1.0.bin"
    
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        st.error("Missing model files! Please ensure 'kokoro-v1.0.onnx' and 'voices-v1.0.bin' are uploaded directly to your GitHub repository root.")
        st.stop()
        
    return Kokoro(model_path, voices_path)

VOICE_MAP = {
    "🇺🇸 Hinsene (American Female - Warm)": "af_heart",
    "🇺🇸 Barashe (American Female - Soft)": "af_bella",
    "🇺🇸 Likitu (American Female - Clear)": "af_nicole",
    "🇺🇸 Lalise (American Female - News)": "af_sarah",
    "🇺🇸 Latu (American Female - Casual)": "af_sky",
    "🇺🇸 Lamessa (American Male - Deep)": "am_adam",
    "🇺🇸 Latera (American Male - Crisp)": "am_michael",
    "🇬🇧 Bontu (British Female - Professional)": "bf_emma",
    "🇬🇧 Buze (British Female - Warm)": "bf_isabella",
    "🇬🇧 Lemi (British Male - Expressive)": "bm_george",
    "🇬🇧 Lencho (British Male - Narration)": "bm_fable"
}

# 4. Sidebar Controls & Preview Feature
with st.sidebar:
    st.title("⚙️ Studio Settings")
    voice_display_name = st.selectbox("🎙️ Voice Persona", options=list(VOICE_MAP.keys()), index=10)

    if st.button("▶️ Preview Voice"):
        voice_key = VOICE_MAP.get(voice_display_name, 'bm_fable')
        preview_text = "Hello! This is a quick preview of this voice persona."
        with st.spinner("Generating preview..."):
            try:
                kokoro_engine = get_kokoro_engine()
                samples, sample_rate = kokoro_engine.create(
                    preview_text, voice=voice_key, speed=1.0, lang="en-us"
                )
                if samples is not None and len(samples) > 0:
                    temp_preview = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sf.write(temp_preview.name, samples, sample_rate)
                    st.audio(temp_preview.name, format="audio/wav")
            except Exception:
                st.error("Could not generate preview.")

    st.markdown("<br>", unsafe_allow_html=True)
    speed = st.slider("⚡ Speed Rate", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    st.divider()
    st.caption("🚀 **Engine:** Streamlit RAM Optimized.")

# 5. Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎙️ <span class="lencho-highlight">LENCHO X LATERA</span> AUDIO STUDIO</div>
    <div class="hero-subtitle">Studio-Grade Text-to-Speech Engine Created by <span class="lencho-highlight">Lencho</span> Lemessa & <span class="lencho-highlight">Latera</span> Lemessa</div>
</div>
""", unsafe_allow_html=True)

# Studio Card Input
with st.container(border=True):
    st.subheader("📝 Script Editor")
    text_input = st.text_area("Input Script", height=180, placeholder="Type or paste your text here...", label_visibility="collapsed")

    char_count = len(text_input)
    word_count = len(text_input.split()) if text_input else 0
    est_sec = round(word_count / (2.5 * speed)) if word_count > 0 else 0

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.caption(f"**Characters:** `{char_count}`")
    with col_stat2:
        st.caption(f"**Words:** `{word_count}`")
    with col_stat3:
        st.caption(f"**Est. Duration:** `~{est_sec}s`")

    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("✨ Generate Audio", type="primary")

# Output Section
if generate_btn:
    if not text_input.strip():
        st.warning("Please enter some text in the script editor first.")
    else:
        st.markdown("<h3 style='color: white;'>🔊 Studio Render Output</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            try:
                voice_key = VOICE_MAP.get(voice_display_name, 'bm_fable')
                kokoro = get_kokoro_engine()
                
                with st.spinner("Synthesizing speech..."):
                    samples, sample_rate = kokoro.create(text_input, voice=voice_key, speed=speed, lang="en-us")

                if samples is not None and len(samples) > 0:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sf.write(temp_file.name, samples, sample_rate)
                    
                    col_audio, col_dl = st.columns([3, 1])
                    with col_audio:
                        st.audio(temp_file.name, format="audio/wav")
                    with col_dl:
                        with open(temp_file.name, "rb") as file:
                            st.download_button(
                                label="📥 Download WAV",
                                data=file,
                                file_name="lencho_latera_voice.wav",
                                mime="audio/wav",
                                use_container_width=True
                            )
                else:
                    st.error("No audio generated.")

            except Exception as e:
                st.error("⚠️ An internal error occurred during synthesis:")
                st.exception(e)
