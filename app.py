import tempfile
import gc
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from kokoro import KPipeline

st.set_page_config(page_title="Lencho Voice Lab", page_icon="🎧")

# Restrict PyTorch CPU threads to stop memory/CPU spikes
torch.set_num_threads(1)

@st.cache_resource(max_entries=1)
def load_pipeline():
    return KPipeline(lang_code='a')

VOICE_MAP = {
    "Hinsene (American Female - Warm)": "af_heart",
    "Barashe (American Female - Soft)": "af_bella",
    "Likitu (American Female - Clear)": "af_nicole",
    "Lalise (American Female - News)": "af_sarah",
    "Latu (American Female - Casual)": "af_sky",
    "Lamessa (American Male - Deep)": "am_adam",
    "Latera (American Male - Crisp)": "am_michael",
    "Bontu (British Female - Professional)": "bf_emma",
    "Buze (British Female - Warm)": "bf_isabella",
    "Lemi (British Male - Expressive)": "bm_george",
    "Lencho (British Male - Narration)": "bm_fable"
}

st.title("🎧 Lencho Voice Lab")

text_input = st.text_area("Input Script", height=150, placeholder="Type your text here...")
voice_display_name = st.selectbox("Voice Persona", options=list(VOICE_MAP.keys()))
speed = st.slider("Speed Rate", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

if st.button("✨ Generate Audio", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.status("Initializing Kokoro TTS...", expanded=True) as status:
            st.write("🧠 Loading model pipeline...")
            pipeline = load_pipeline()
            actual_voice = VOICE_MAP.get(voice_display_name, 'af_heart')
            
            st.write("🗣️ Generating speech segments...")
            audio_chunks = []
            
            # Disable gradient tracking to prevent RAM accumulation
            with torch.no_grad():
                generator = pipeline(text_input, voice=actual_voice, speed=speed)
                for index, (_, _, audio) in enumerate(generator):
                    audio_chunks.append(audio)
                    st.write(f"🔊 Rendered segment {index + 1}...")

            if audio_chunks:
                st.write("💾 Merging segments and saving WAV...")
                full_audio = np.concatenate(audio_chunks)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(temp_file.name, full_audio, 24000)
                
                # Clear memory immediately after write
                del audio_chunks, full_audio
                gc.collect()

                status.update(label="✨ Audio ready!", state="complete", expanded=False)
                st.audio(temp_file.name, format="audio/wav")
                
                with open(temp_file.name, "rb") as file:
                    st.download_button(
                        label="📥 Download WAV File",
                        data=file,
                        file_name="lencho_voice.wav",
                        mime="audio/wav"
                    )
            else:
                status.update(label="⚠️ Failed to generate speech.", state="error")
