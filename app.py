import tempfile
import gc
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from kokoro import KPipeline

# 1. Page Configuration
st.set_page_config(
    page_title="Lencho Voice Lab",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Restrict PyTorch CPU threads to stop memory/CPU spikes
torch.set_num_threads(1)

# 2. Animated Gemini Neon Gradient & Glassmorphism Styling
st.markdown("""
<style>
    /* Full App Animated Gemini Neon Background */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #0284c7, #4f46e5);
        background-size: 400% 400%;
        animation: geminiGradient 14s ease infinite;
    }

    @keyframes geminiGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Make Streamlit top header transparent */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Hero Banner with Glowing LENCHO Highlight */
    .hero-container {
        text-align: center;
        padding: 2.2rem 1.5rem;
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0;
        letter-spacing: 1px;
    }

    /* Glowing Animated Gradient text specifically for LENCHO */
    .lencho-highlight {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #f472b6, #38bdf8);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glowingText 4s linear infinite;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(129, 140, 248, 0.6);
    }

    @keyframes glowingText {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Studio Input Card Glassmorphism */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    }

    /* Neon Pulse Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%) !important;
        background-size: 200% 200% !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.4) !important;
        transition: all 0.3s ease-in-out !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 0 30px rgba(236, 72, 153, 0.7) !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Loader
@st.cache_resource(max_entries=1)
def load_pipeline():
    return KPipeline(lang_code='a')

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
    st.caption("💡 **Tip:** Keep scripts concise for faster generation.")

# 5. Neon Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎧 <span class="lencho-highlight">LENCHO</span> VOICE LAB</div>
    <div class="hero-subtitle">Studio-Grade Text-to-Speech Engine Powered by AI</div>
</div>
""", unsafe_allow_html=True)

# Main Studio Card
with st.container(border=True):
    st.subheader("📝 Script Editor")
    text_input = st.text_area(
        "Input Script", 
        height=180, 
        placeholder="Type or paste your text here...",
        label_visibility="collapsed"
    )

    # Real-time Analytics Bar
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
        st.markdown("### 🔊 Studio Render Output")
        with st.container(border=True):
            with st.status("Initializing Kokoro Engine...", expanded=True) as status:
                st.write("🧠 Loading model pipeline...")
                pipeline = load_pipeline()
                actual_voice = VOICE_MAP.get(voice_display_name, 'bm_fable')
                
                st.write("🗣️ Synthesizing speech segments...")
                raw_segments = [s for s in text_input.replace('\n', '.').split('.') if s.strip()]
                total_estimated = max(1, len(raw_segments))
                
                progress_bar = st.progress(0.0, text="Starting speech synthesis...")
                audio_chunks = []
                
                with torch.no_grad():
                    generator = pipeline(text_input, voice=actual_voice, speed=speed)
                    for index, (_, _, audio) in enumerate(generator):
                        audio_chunks.append(audio)
                        progress = min(0.95, (index + 1) / total_estimated)
                        progress_bar.progress(
                            progress, 
                            text=f"Rendered segment {index + 1} of ~{total_estimated}"
                        )
                        st.write(f"🔊 Rendered segment {index + 1}...")

                if audio_chunks:
                    st.write("💾 Merging tracks and saving WAV...")
                    progress_bar.progress(0.98, text="Finalizing WAV file...")
                    
                    full_audio = np.concatenate(audio_chunks)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sf.write(temp_file.name, full_audio, 24000)
                    
                    progress_bar.progress(1.0, text="Generation complete!")
                    
                    del audio_chunks, full_audio
                    gc.collect()

                    status.update(label="✨ Audio ready!", state="complete", expanded=False)
                    
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
                    status.update(label="⚠️ Failed to generate speech.", state="error")
