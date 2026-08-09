import tempfile
import soundfile as sf
import streamlit as st
import numpy as np
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro

# 1. Page Configuration
st.set_page_config(
    page_title="Lencho Voice Lab",
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

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.15) !important;
    }

    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

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
        font-size: 2.8rem;
        font-weight: 900;
        color: #0f172a;
        margin: 0;
        letter-spacing: 1px;
    }

    .lencho-highlight {
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777, #2563eb);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glowingText 4s linear infinite;
        font-weight: 900;
    }

    @keyframes glowingText {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-subtitle {
        color: #475569;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background-color: #ffffff !important;
        border-radius: 18px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15) !important;
        padding: 1.5rem !important;
    }

    div[data-testid="stVerticalBlock"] > div[style*="border"] h1,
    div[data-testid="stVerticalBlock"] > div[style*="border"] h2,
    div[data-testid="stVerticalBlock"] > div[style*="border"] h3,
    div[data-testid="stVerticalBlock"] > div[style*="border"] p,
    div[data-testid="stVerticalBlock"] > div[style*="border"] span,
    div[data-testid="stVerticalBlock"] > div[style*="border"] label {
        color: #0f172a !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #334155 !important;
        font-size: 0.95rem !important;
    }

    .stTextArea textarea {
        color: #0f172a !important;
        background-color: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.3s ease-in-out !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# Replace lines 140-146 in app.py with this:
@st.cache_resource(show_spinner=False)
def load_onnx_kokoro():
    # Automatically downloads model.onnx and voices.bin/voices.json from the correct ONNX repo
    return Kokoro.from_pretrained()
    voices_path = hf_hub_download(
        repo_id="hexgrad/Kokoro-82M", 
        filename="voices.json"
    )
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

# 4. Sidebar Controls
with st.sidebar:
    st.title("⚙️ Studio Settings")
    st.markdown("Customize your voice engine parameters.")
    st.divider()

    voice_display_name = st.selectbox(
        "🎙️ Voice Persona", 
        options=list(VOICE_MAP.keys()),
        index=10
    )

    speed = st.slider(
        "⚡ Speed Rate", 
        min_value=0.5, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="Adjust the pace of speech generation."
    )

    st.divider()
    st.caption("🚀 **ONNX Speed Acceleration:** Active for fast Cloud CPU rendering.")

# 5. Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎧 <span class="lencho-highlight">LENCHOS</span> VOICE LAB</div>
    <div class="hero-subtitle">Studio-Grade Text-to-Speech Engine Created by <span class="lencho-highlight">Lencho</span> Lemessa For <span class="lencho-highlight">Latera</span> Lemessa </div>
</div>
""", unsafe_allow_html=True)

# Studio Card Input
with st.container(border=True):
    st.subheader("📝 Script Editor")
    text_input = st.text_area(
        "Input Script", 
        height=180, 
        placeholder="Type or paste your text here...",
        label_visibility="collapsed"
    )

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
            progress_bar = st.progress(0.0, text="Initializing ONNX Engine...")
            try:
                actual_voice = VOICE_MAP.get(voice_display_name, 'bm_fable')
                lang_code = "en-gb" if actual_voice.startswith("b") else "en-us"
                
                progress_bar.progress(0.2, text="Loading ONNX Runtime...")
                kokoro = load_onnx_kokoro()
                
                progress_bar.progress(0.5, text="Synthesizing audio...")
                samples, sample_rate = kokoro.create(
                    text_input, 
                    voice=actual_voice, 
                    speed=speed, 
                    lang=lang_code
                )

                if len(samples) > 0:
                    progress_bar.progress(0.9, text="Generating WAV file...")
                    
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sf.write(temp_file.name, samples, sample_rate)
                    
                    progress_bar.progress(1.0, text="Complete!")

                    col_audio, col_dl = st.columns([3, 1])
                    with col_audio:
                        st.audio(temp_file.name, format="audio/wav")
                    with col_dl:
                        with open(temp_file.name, "rb") as file:
                            st.download_button(
                                label="📥 Download WAV",
                                data=file,
                                file_name="lencho_voice.wav",
                                mime="audio/wav",
                                use_container_width=True
                            )
                else:
                    progress_bar.empty()
                    st.error("No audio generated.")

            except Exception as e:
                progress_bar.empty()
                st.error("⚠️ An internal error occurred during synthesis:")
                st.exception(e)
