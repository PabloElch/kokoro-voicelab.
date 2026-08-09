import os
import tempfile
import urllib.request
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

# Initialize Session State with a strict history limit to save RAM (capped at 2 items)
if "history" not in st.session_state:
    st.session_state.history = []

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
        font-size: 2.5rem;
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

# 3. Automatic Downloader & Kokoro Engine Initializer
@st.cache_resource(show_spinner=False)
def get_kokoro_engine():
    model_path = "kokoro-v1.0.onnx"
    voices_path = "voices-v1.0.bin"

    if not os.path.exists(model_path):
        with st.spinner("Downloading Kokoro ONNX model (first-time boot setup)..."):
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
                model_path
            )

    if not os.path.exists(voices_path):
        with st.spinner("Downloading voice weights configuration..."):
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                voices_path
            )

    return Kokoro(model_path, voices_path)

@st.cache_resource(show_spinner=False)
def get_background_track():
    bg_path = "ambient_bed.wav"
    if not os.path.exists(bg_path):
        try:
            urllib.request.urlretrieve(
                "https://github.com/rafaelreis-io/rafaelreis-io/raw/main/ambient.wav",
                bg_path
            )
        except Exception:
            pass
    return bg_path

def mix_audio_beds(voice_samples, sample_rate, bg_path, volume=0.15):
    if not os.path.exists(bg_path):
        return voice_samples
    try:
        bg_samples, _ = sf.read(bg_path)
        if len(bg_samples.shape) > 1:
            bg_samples = np.mean(bg_samples, axis=1)
        if len(voice_samples.shape) > 1:
            voice_samples = np.mean(voice_samples, axis=1)
            
        if len(bg_samples) < len(voice_samples):
            repeats = int(np.ceil(len(voice_samples) / len(bg_samples)))
            bg_samples = np.tile(bg_samples, repeats)
            
        bg_samples = bg_samples[:len(voice_samples)]
        mixed = voice_samples + (bg_samples * volume)
        return np.clip(mixed, -1.0, 1.0)
    except Exception:
        return voice_samples

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

# 4. Sidebar Controls & Background Mixer Settings
with st.sidebar:
    st.title("⚙️ Studio Settings")
    st.markdown("Customize your voice engine parameters.")
    st.divider()

    voice_display_name = st.selectbox(
        "🎙️ Voice Persona", 
        options=list(VOICE_MAP.keys()),
        index=10
    )

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
                    st.audio(temp_preview.name, format="audio/wav", autoplay=True)
            except Exception:
                st.error("Could not generate preview.")

    st.markdown("<br>", unsafe_allow_html=True)

    speed = st.slider("⚡ Speed Rate", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    st.divider()
    st.markdown("### 🎵 Background Music Bed")
    enable_bg = st.checkbox("Enable Ambient Bed", value=False)
    bg_volume = st.slider("Music Volume", min_value=0.05, max_value=0.40, value=0.15, step=0.05)

    st.divider()
    st.caption("🚀 **Studio Engine:** Active.")

# 5. Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎙️ <span class="lencho-highlight">LENCHO X LATERA</span> AUDIO STUDIO</div>
    <div class="hero-subtitle">Precision AI Voice Synthesis by <span class="lencho-highlight">Lencho</span> & <span class="lencho-highlight">Latera</span> Lemessa</div>
</div>
""", unsafe_allow_html=True)

# Studio Card Input
with st.container(border=True):
    st.subheader("📝 Script Editor")
    text_input = st.text_area(
        "Input Script", height=180, placeholder="Type or paste your text here...", label_visibility="collapsed"
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
                voice_key = VOICE_MAP.get(voice_display_name, 'bm_fable')
                
                progress_bar.progress(0.3, text="Loading Kokoro Model...")
                kokoro = get_kokoro_engine()
                
                progress_bar.progress(0.6, text="Synthesizing speech...")
                samples, sample_rate = kokoro.create(text_input, voice=voice_key, speed=speed, lang="en-us")

                if samples is not None and len(samples) > 0:
                    progress_bar.progress(0.8, text="Mixing audio tracks...")
                    if enable_bg:
                        bg_path = get_background_track()
                        samples = mix_audio_beds(samples, sample_rate, bg_path, volume=bg_volume)

                    progress_bar.progress(0.9, text="Formatting WAV file...")
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sf.write(temp_file.name, samples, sample_rate)
                    
                    with open(temp_file.name, "rb") as f:
                        audio_bytes = f.read()

                    # Save to history but strictly keep only the latest 2 items to prevent memory overload
                    history_item = {
                        "text": text_input,
                        "voice": voice_display_name,
                        "speed": speed,
                        "audio_bytes": audio_bytes,
                        "filename": f"lencho_latera_voice_{len(st.session_state.history)+1}.wav"
                    }
                    st.session_state.history.insert(0, history_item)
                    if len(st.session_state.history) > 2:
                        st.session_state.history.pop()  # Drop oldest item from RAM

                    progress_bar.progress(1.0, text="Complete!")

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
                    progress_bar.empty()
                    st.error("No audio generated.")

            except Exception as e:
                progress_bar.empty()
                st.error("⚠️ An internal error occurred during synthesis:")
                st.exception(e)

# Session History & Archive Section (Strictly capped at 2 items to prevent memory limits)
if st.session_state.history:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>📜 Recent Session Archive (Last 2)</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        for idx, item in enumerate(st.session_state.history):
            item_num = len(st.session_state.history) - idx
            st.markdown(f"**#{item_num} | Persona:** `{item['voice']}` | **Speed:** `{item['speed']}x`")
            snippet = item['text'][:100] + "..." if len(item['text']) > 100 else item['text']
            st.caption(f"**Script:** {snippet}")
            
            hist_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            hist_temp.write(item['audio_bytes'])
            hist_temp.close()
            
            col_ha, col_hd = st.columns([3, 1])
            with col_ha:
                st.audio(hist_temp.name, format="audio/wav")
            with col_hd:
                st.download_button(
                    label="📥 Download",
                    data=item['audio_bytes'],
                    file_name=item['filename'],
                    mime="audio/wav",
                    key=f"history_download_{idx}",
                    use_container_width=True
                )
            if idx < len(st.session_state.history) - 1:
                st.divider()
