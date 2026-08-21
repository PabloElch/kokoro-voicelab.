import gc
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
from kokoro_onnx import Kokoro


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Lenchos Audio Studio"

TARGET_WORDS_PER_CHUNK = 500

OUTPUT_SAMPLE_RATE = 24000

MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)

VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)

MODEL_FILENAME = "kokoro-v1.0.onnx"
VOICES_FILENAME = "voices-v1.0.bin"

# Friendly Voice Map for Single Narration
SINGLE_VOICE_MAP = {
    "🇺🇸 Beza (Warm Female)": "af_heart",
    "🇺🇸 Birikti (Soft Female)": "af_bella",
    "🇺🇸 Demoze (Clear Female)": "af_nicole",
    "🇺🇸 Lalise (News Female)": "af_sarah",
    "🇺🇸 Efrata (Casual Female)": "af_sky",
    "🇺🇸 Lencho (Deep Male)": "am_adam",
    "🇺🇸 Dego (Crisp Male)": "am_michael",
    "🇬🇧 Bontu (Professional Female)": "bf_emma",
    "🇬🇧 Hawi (Warm Female)": "bf_isabella",
    "🇬🇧 Lalisa (Expressive Male)": "bm_george",
    "🇬🇧 Lemi (Narration Male)": "bm_fable",
}

# Raw Real Voice IDs for 2-Person Conversation
RAW_VOICES = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_fable",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Lenchos Audio Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS & DROPDOWN FIXES
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0b0d;
        color: #f2f2f2;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0b0b0d;
    }
    [data-testid="stHeader"] {
        background-color: #0b0b0d;
    }
    [data-testid="stSidebar"] {
        background-color: #08080a;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: #08080a;
    }
    body, p, label, span, div {
        color: #f2f2f2;
    }
    .stMarkdown, .stCaption {
        color: #d6d6d6;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    .hero-container {
        padding: 28px 10px 20px 10px;
        text-align: center;
    }
    
    /* Textareas and Inputs */
    textarea {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
        border: 1px solid #303036 !important;
    }
    input {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
    }

    /* Fix Selectbox Dropdown Options Readability */
    [data-baseweb="select"] > div {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
        border-color: #303036 !important;
    }
    [data-baseweb="select"] span {
        color: #f5f5f5 !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
        border: 1px solid #303036 !important;
    }
    [role="option"] {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
    }
    [role="option"]:hover {
        background-color: #222226 !important;
        color: #ffffff !important;
    }
    [aria-selected="true"] {
        background-color: #2a2a30 !important;
        color: #ffffff !important;
    }

    [data-testid="stMetric"] {
        background-color: #111114;
        border: 1px solid #29292d;
        border-radius: 12px;
        padding: 12px;
    }
    [data-testid="stMetricLabel"] {
        color: #aaaaaf !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    .stButton > button {
        background-color: #18181c;
        color: #ffffff;
        border: 1px solid #35353b;
    }
    .stButton > button:hover {
        background-color: #24242a;
        color: #ffffff;
        border-color: #55555d;
    }
    hr {
        border-color: #29292d !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <style>
    @keyframes softGlow {
        0% { text-shadow: 0 0 4px rgba(79, 70, 229, 0.3), 0 0 10px rgba(79, 70, 229, 0.1); }
        50% { text-shadow: 0 0 12px rgba(79, 70, 229, 0.6), 0 0 20px rgba(79, 70, 229, 0.3); }
        100% { text-shadow: 0 0 4px rgba(79, 70, 229, 0.3), 0 0 10px rgba(79, 70, 229, 0.1); }
    }
    .glowing-name {
        color: #818cf8;
        font-weight: 700;
        animation: softGlow 3s infinite ease-in-out;
    }
    </style>
    <div class="hero-container">
        <div style="font-weight: 800; font-size: 28px; margin-bottom: 8px;">
            🎙️ LENCHOS AUDIO STUDIO
        </div>
        <div style="font-size: 16px; color: #a1a1aa;">
            Built by <span class="glowing-name">Lencho Lemessa</span> to deliver high-quality voice synthesis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def get_base_work_dir():
    base_dir = Path(tempfile.gettempdir()) / "lenchos_audio_studio"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def download_file(url, destination):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return str(destination)


# ============================================================
# KOKORO ENGINE
# ============================================================

@st.cache_resource(show_spinner="Loading Kokoro model...")
def get_kokoro_engine():
    model_dir = get_base_work_dir() / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / MODEL_FILENAME
    voices_path = model_dir / VOICES_FILENAME

    download_file(MODEL_URL, model_path)
    download_file(VOICES_URL, voices_path)

    return Kokoro(str(model_path), str(voices_path))


# ============================================================
# TEXT PROCESSING & CHUNKING UTILS
# ============================================================

def normalize_script(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        cleaned = re.sub(r"\s+", " ", line.strip())
        if cleaned:
            lines.append(cleaned)
    return "\n\n".join(lines)


def split_long_sentence(sentence, target_words):
    words = sentence.split()
    if len(words) <= target_words:
        return [sentence.strip()]
    pieces = []
    for start in range(0, len(words), target_words):
        piece = " ".join(words[start : start + target_words])
        if piece.strip():
            pieces.append(piece.strip())
    return pieces


def split_script_into_chunks(text, target_words=TARGET_WORDS_PER_CHUNK):
    text = normalize_script(text)
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    sentences = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraph_sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in paragraph_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentences.extend(split_long_sentence(sentence, target_words))

    chunks = []
    current = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > target_words:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def parse_dialogue_script(text, default_voice, secondary_voice):
    lines = text.strip().split("\n")
    parsed_turns = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        match = re.match(r"^([\w\-_]+?)\s*:\s*(.+)$", line_str)
        if match:
            speaker_label = match.group(1).strip().lower()
            utterance = match.group(2).strip()
            
            if speaker_label in ["a", "speaker a", "speaker 1", "s1", "voice 1", default_voice.lower()]:
                assigned_voice = default_voice
            elif speaker_label in ["b", "speaker b", "speaker 2", "s2", "voice 2", secondary_voice.lower()]:
                assigned_voice = secondary_voice
            else:
                assigned_voice = default_voice
                
            if utterance:
                parsed_turns.append({"voice": assigned_voice, "text": utterance})
        else:
            parsed_turns.append({"voice": default_voice, "text": line_str})
    return parsed_turns


def make_job_id(script, voice, speed):
    payload = f"{script}|{voice}|{speed:.4f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def create_job_directory(job_id):
    base_dir = get_base_work_dir()
    job_dir = base_dir / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "chunks").mkdir(parents=True, exist_ok=True)
    return job_dir


def get_chunk_path(job_dir, index):
    return Path(job_dir) / "chunks" / f"chunk_{index:03d}.wav"


def chunk_is_complete(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        info = sf.info(str(path))
        return info.frames > 0 and info.samplerate > 0
    except Exception:
        return False


def convert_wav_to_mp3(wav_path, mp3_path):
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is not installed.") from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, "-y", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "128k",
        "-ar", str(OUTPUT_SAMPLE_RATE), str(mp3_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("FFmpeg MP3 conversion failed:\n\n" + result.stderr[-3000:])


def generate_preview_mp3(kokoro, voice, speed=1.0):
    preview_text = f"Hello. This is a voice preview using {voice}."
    samples, sample_rate = kokoro.create(preview_text, voice=voice, speed=float(speed), lang="en-us")
    samples = np.asarray(samples, dtype=np.float32)
    sample_rate = int(sample_rate)

    samples_int16 = np.clip(samples, -1.0, 1.0)
    samples_int16 = (samples_int16 * 32767).astype(np.int16)
    raw_audio = samples_int16.tobytes()

    del samples, samples_int16
    gc.collect()

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is not installed.") from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, "-y", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
        "-i", "pipe:0", "-codec:a", "libmp3lame", "-b:a", "128k",
        "-ar", str(OUTPUT_SAMPLE_RATE), "-f", "mp3", "pipe:1",
    ]
    result = subprocess.run(command, input=raw_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError("Could not create MP3 preview.")
    return result.stdout


# ============================================================
# MAIN TABS & UI LAYOUT
# ============================================================

tab_single, tab_dual = st.tabs(["🎙️ Single Narration", "👥 2-Person Conversation"])

# ------------------------------------------------------------
# TAB 1: SINGLE NARRATION
# ------------------------------------------------------------
with tab_single:
    st.header("📜 Single Narration Studio")

    with st.sidebar:
        st.header("🎙️ Voice Settings")
        selected_friendly_name = st.selectbox(
            "Narrator Voice",
            list(SINGLE_VOICE_MAP.keys()),
            index=0,
            key="single_voice_select"
        )
        single_voice = SINGLE_VOICE_MAP[selected_friendly_name]
        single_speed = st.slider("Speech speed", min_value=0.5, max_value=2.0, value=1.0, step=0.05, key="single_speed")

        if st.button("▶️ Preview Voice", use_container_width=True, key="preview_single_btn"):
            with st.spinner(f"Generating preview for {selected_friendly_name}..."):
                try:
                    kokoro = get_kokoro_engine()
                    preview_mp3 = generate_preview_mp3(kokoro, single_voice, single_speed)
                    st.audio(preview_mp3, format="audio/mpeg")
                except Exception as exc:
                    st.error("Could not generate voice preview.")
                    st.exception(exc)

    single_script_input = st.text_area(
        "Paste your story script here",
        height=380,
        placeholder="Type or paste your long story script here...",
        key="single_script_text"
    )

    norm_single = normalize_script(single_script_input)
    single_chunks = split_script_into_chunks(norm_single, TARGET_WORDS_PER_CHUNK) if norm_single else []
    
    single_words = sum(len(chunk.split()) for chunk in single_chunks)
    estimated_minutes = (single_words / 130.0) / float(single_speed) if single_words else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", f"{single_words:,}")
    with col2:
        st.metric("Estimated Minutes", f"{estimated_minutes:.1f} min" if single_words else "0 min")
    with col3:
        st.metric("Chunks", f"{len(single_chunks)}")

    if st.button("🎙️ Generate Narration", type="primary", use_container_width=True, key="gen_single_btn"):
        if single_words < 3:
            st.error("Please enter a longer script.")
            st.stop()

        job_id = make_job_id(norm_single, single_voice, single_speed)
        job_dir = create_job_directory(job_id)
        final_wav = job_dir / "final_narration.wav"
        final_mp3 = job_dir / "final_narration.mp3"

        if final_mp3.exists() and final_mp3.stat().st_size > 0:
            st.success("Narration already exists in disk cache. Loading output instantly!")
            st.audio(str(final_mp3), format="audio/mpeg")
            st.stop()

        try:
            kokoro = get_kokoro_engine()
        except Exception as exc:
            st.error("Model loading failed.")
            st.exception(exc)
            st.stop()

        progress = st.progress(0, text="Generating audio chunks...")
        chunk_paths = []

        for idx, chunk_text in enumerate(single_chunks, start=1):
            c_path = get_chunk_path(job_dir, idx)
            chunk_paths.append(c_path)
            if not chunk_is_complete(c_path):
                samples, sr = kokoro.create(chunk_text, voice=single_voice, speed=float(single_speed), lang="en-us")
                sf.write(str(c_path), np.asarray(samples, dtype=np.float32), int(sr), subtype="PCM_16")
            progress.progress(idx / len(single_chunks), text=f"Processed chunk {idx}/{len(single_chunks)}")

        audio_data = []
        for cp in chunk_paths:
            data, _ = sf.read(str(cp), dtype="float32")
            audio_data.append(data)
        
        if audio_data:
            sf.write(str(final_wav), np.concatenate(audio_data), OUTPUT_SAMPLE_RATE, subtype="PCM_16")
            convert_wav_to_mp3(final_wav, final_mp3)
            st.success("Narration generated successfully and cached to disk!")
            st.audio(str(final_mp3), format="audio/mpeg")


# ------------------------------------------------------------
# TAB 2: 2-PERSON CONVERSATION (DISK CACHE ENHANCED)
# ------------------------------------------------------------
with tab_dual:
    st.header("👥 2-Person Conversation Studio")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        voice_1 = st.selectbox("Voice 1 Persona (Raw ID)", RAW_VOICES, index=0, key="voice_1_select")
        if st.button("▶️ Preview Voice 1", use_container_width=True, key="preview_v1_btn"):
            with st.spinner(f"Generating preview for {voice_1}..."):
                try:
                    kokoro = get_kokoro_engine()
                    preview_mp3_1 = generate_preview_mp3(kokoro, voice_1, 1.0)
                    st.audio(preview_mp3_1, format="audio/mpeg")
                except Exception as exc:
                    st.error("Preview failed.")
                    st.exception(exc)

    with col_v2:
        voice_2 = st.selectbox("Voice 2 Persona (Raw ID)", RAW_VOICES, index=5, key="voice_2_select")
        if st.button("▶️ Preview Voice 2", use_container_width=True, key="preview_v2_btn"):
            with st.spinner(f"Generating preview for {voice_2}..."):
                try:
                    kokoro = get_kokoro_engine()
                    preview_mp3_2 = generate_preview_mp3(kokoro, voice_2, 1.0)
                    st.audio(preview_mp3_2, format="audio/mpeg")
                except Exception as exc:
                    st.error("Preview failed.")
                    st.exception(exc)

    dual_script_input = st.text_area(
        "Paste dialogue script",
        height=320,
        placeholder="A: Hey, did you test the local synthesis setup?\nB: Yes, it works seamlessly!",
        key="dual_script_text"
    )

    parsed_turns = parse_dialogue_script(dual_script_input, voice_1, voice_2) if dual_script_input else []
    dual_words = sum(len(turn["text"].split()) for turn in parsed_turns)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.metric("Dialogue Turns", f"{len(parsed_turns)}")
    with col_d2:
        st.metric("Total Words", f"{dual_words:,}")

    if st.button("🎙️ Generate Conversation Audio", type="primary", use_container_width=True, key="gen_dual_btn"):
        if dual_words < 3:
            st.error("Please enter a valid dialogue script.")
            st.stop()

        job_id = make_job_id(dual_script_input, f"{voice_1}_{voice_2}", 1.0)
        job_dir = create_job_directory(job_id)
        final_wav = job_dir / "final_conversation.wav"
        final_mp3 = job_dir / "final_conversation.mp3"

        # 🔥 DISK CACHE CHECK: If this exact script was already generated, load instantly!
        if final_mp3.exists() and final_mp3.stat().st_size > 0:
            st.success("Conversation already exists in disk cache. Loading output instantly!")
            st.audio(str(final_mp3), format="audio/mpeg")
            st.stop()

        try:
            kokoro = get_kokoro_engine()
        except Exception as exc:
            st.error("Model loading failed.")
            st.exception(exc)
            st.stop()

        progress = st.progress(0, text="Synthesizing conversation turns...")
        turn_paths = []
        pause_samples = np.zeros(int(OUTPUT_SAMPLE_RATE * 0.35), dtype=np.float32)

        for idx, turn in enumerate(parsed_turns, start=1):
            t_path = get_chunk_path(job_dir, idx)
            turn_paths.append(t_path)
            # If a single turn's cache is already on disk, skip re-rendering it!
            if not chunk_is_complete(t_path):
                samples, sr = kokoro.create(turn["text"], voice=turn["voice"], speed=1.0, lang="en-us")
                sf.write(str(t_path), np.asarray(samples, dtype=np.float32), int(sr), subtype="PCM_16")
            progress.progress(idx / len(parsed_turns), text=f"Processed turn {idx}/{len(parsed_turns)}")

        audio_segments = []
        for tp in turn_paths:
            data, _ = sf.read(str(tp), dtype="float32")
            audio_segments.append(data)
            audio_segments.append(pause_samples)

        if audio_segments:
            sf.write(str(final_wav), np.concatenate(audio_segments), OUTPUT_SAMPLE_RATE, subtype="PCM_16")
            convert_wav_to_mp3(final_wav, final_mp3)
            st.success("Conversation generated successfully and cached to disk!")
            st.audio(str(final_mp3), format="audio/mpeg")
