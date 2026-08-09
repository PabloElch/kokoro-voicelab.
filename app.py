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

# 2. Modern UI CSS Styling
st.markdown("""
<style>
    /* Sleek container styling */
    .hero-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem;
        border-radius: 14px;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }
    /* Primary Action Button Customization */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 15px rgba(59, 130, 246, 0.4);
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
    st.markdown("Customize your voice engine parameters below.")
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
    st.caption("💡 **Tip:** Break longer scripts into smaller paragraphs for faster rendering.")

# 5. Main Content Panel
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🎧 Lencho Voice Lab</div>
    <div class="hero-subtitle">High-Fidelity AI Text-to-Speech Engine</div>
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

    # Real-time Stats Row
    char_count = len(text_input)
    word_count = len(text_input.split()) if text_input else 0
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.caption(f"**Characters:** {char_count}")
    with col_stat2:
        st.caption(f"**Words:** {word_count}")
    with col_stat3:
        est_sec = round(word_count / (2.5 * speed)) if word_count > 0 else 0
        st.caption(f"**Est. Duration:** ~{est_sec}s")

    generate_btn = st.button("✨ Generate Audio", type="primary")

# Output Section
if generate_btn:
    if not text_input.strip():
        st.warning("Please enter some text in the script editor first.")
    else:
        st.markdown("### 🔊 Audio Generation Output")
        with st.container(border=True):
            with st.status("Initializing Kokoro TTS...", expanded=True) as status:
                st.write("🧠 Loading model pipeline...")
                pipeline = load_pipeline()
                actual_voice = VOICE_MAP.get(voice_display_name, 'bm_fable')
                
                st.write("🗣️ Generating speech segments...")
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
                    st.write("💾 Finalizing WAV file...")
                    progress_bar.progress(0.98, text="Saving WAV file...")
                    
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
