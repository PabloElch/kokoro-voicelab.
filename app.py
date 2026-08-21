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

TARGET_WORDS_PER_CHUNK = 550

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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       MAIN APP BACKGROUND
       ========================= */

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

    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {
        background-color: #08080a;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #08080a;
    }

    /* =========================
       GENERAL TEXT
       ========================= */

    body,
    p,
    label,
    span,
    div {
        color: #f2f2f2;
    }

    .stMarkdown,
    .stCaption {
        color: #d6d6d6;
    }

    /* =========================
       HEADINGS
       ========================= */

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* =========================
       HERO
       ========================= */

    .hero-container {
        padding: 28px 10px 20px 10px;
        text-align: center;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 15px;
        opacity: 0.75;
        margin-top: 8px;
        color: #d0d0d0;
    }

    /* =========================
       STATUS BOX
       ========================= */

    .status-box {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #29292d;
        background-color: #111114;
        margin-top: 10px;
        color: #f2f2f2;
    }

    /* =========================
       TEXT AREA
       ========================= */

    textarea {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
        border: 1px solid #303036 !important;
    }

    textarea::placeholder {
        color: #88888f !important;
    }

    /* =========================
       INPUTS / SELECTBOXES
       ========================= */

    input {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
    }

    [data-baseweb="select"] > div {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
        border-color: #303036 !important;
    }

    [data-baseweb="popover"] {
        background-color: #111114 !important;
    }

    [role="option"] {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
    }

    [role="option"]:hover {
        background-color: #222226 !important;
    }

    /* =========================
       METRIC BOXES
       ========================= */

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

    /* =========================
       EXPANDERS
       ========================= */

    [data-testid="stExpander"] {
        background-color: #111114;
        border: 1px solid #29292d;
        border-radius: 12px;
    }

    /* =========================
       PROGRESS BAR
       ========================= */

    [data-testid="stProgress"] {
        background-color: #050505 !important;
    }

    [data-testid="stProgress"] > div {
        background-color: #050505 !important;
        border-radius: 10px;
    }

    [data-testid="stProgress"] div[role="progressbar"] {
        background-color: #050505 !important;
        border-radius: 10px;
    }

    [data-testid="stProgress"] div[role="progressbar"] > div {
        background-color: #777777 !important;
        border-radius: 10px;
    }

    /* =========================
       BUTTONS
       ========================= */

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

    /* =========================
       DIVIDERS
       ========================= */

    hr {
        border-color: #29292d !important;
    }

    /* =========================
       AUDIO PLAYER
       ========================= */

    audio {
        background-color: #111114;
    }

    /* =========================
       CODE / PREVIEW AREAS
       ========================= */

    pre,
    code {
        background-color: #111114 !important;
        color: #eeeeee !important;
    }

    /* =========================
       ALERTS / INFO BOXES
       ========================= */

    [data-testid="stAlert"] {
        background-color: #111114;
        color: #eeeeee;
    }

    /* =========================
       SCROLLBAR
       ========================= */

    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #08080a;
    }

    ::-webkit-scrollbar-thumb {
        background: #303036;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #44444c;
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
        <div class="hero-title" style="font-weight: 800; font-size: 28px; margin-bottom: 8px;">
            🎙️ LENCHOS AUDIO STUDIO
        </div>
        <div class="hero-subtitle" style="font-size: 16px; color: #a1a1aa;">
            Built by <span class="glowing-name">Lencho Lemessa</span> to deliver multi-speaker conversation &amp; voice synthesis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "current_job" not in st.session_state:
    st.session_state.current_job = None

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def get_base_work_dir():
    base_dir = (
        Path(tempfile.gettempdir())
        / "lenchos_audio_studio"
    )
    base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return base_dir


def cleanup_old_jobs(keep_job_dirs=None):
    if keep_job_dirs is None:
        keep_job_dirs = set()

    base_dir = get_base_work_dir()

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name in {"model"}:
            continue
        if str(item) in keep_job_dirs:
            continue
        try:
            shutil.rmtree(item)
        except Exception:
            pass


# ============================================================
# FILE DOWNLOAD
# ============================================================

def download_file(url, destination):
    destination = Path(destination)

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):
        return str(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    urllib.request.urlretrieve(
        url,
        destination,
    )

    return str(destination)


# ============================================================
# KOKORO ENGINE
# ============================================================

@st.cache_resource(
    show_spinner="Loading Kokoro model..."
)
def get_kokoro_engine():
    model_dir = (
        get_base_work_dir()
        / "model"
    )
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = model_dir / MODEL_FILENAME
    voices_path = model_dir / VOICES_FILENAME

    download_file(MODEL_URL, model_path)
    download_file(VOICES_URL, voices_path)

    kokoro = Kokoro(
        str(model_path),
        str(voices_path),
    )

    return kokoro


# ============================================================
# VOICE MAP
# ============================================================

VOICE_MAP = {
    "🇺🇸 Beza (American Female - Warm)": "af_heart",
    "🇺🇸 Birikti (American Female - Soft)": "af_bella",
    "🇺🇸 Demoze (American Female - Clear)": "af_nicole",
    "🇺🇸 Lalise (American Female - News)": "af_sarah",
    "🇺🇸 Efrata (American Female - Casual)": "af_sky",
    "🇺🇸 Lencho (American Male - Deep)": "am_adam",
    "🇺🇸 Dego (American Male - Crisp)": "am_michael",
    "🇬🇧 Bontu (British Female - Professional)": "bf_emma",
    "🇬🇧 Hawi (British Female - Warm)": "bf_isabella",
    "🇬🇧 Lalisa (British Male - Expressive)": "bm_george",
    "🇬🇧 Lemi (British Male - Narration)": "bm_fable",
}


# ============================================================
# DIALOGUE SCRIPT PARSER
# ============================================================

def parse_dialogue_script(text, default_voice):
    """
    Parses a conversation script formatted like:
    Speaker 1: Hello there!
    Speaker 2: Hi back!

    Or falls back to assigning lines line-by-line / block-by-block.
    Recognizes patterns like 'Name:' or '[VoiceKey]:' at start of lines.
    """
    lines = text.strip().split("\n")
    parsed_turns = []
    
    # Inverted lookup for matching names/labels if user typed them out
    name_to_key = {
        "speaker 1": "af_heart",
        "speaker 2": "am_adam",
        "bella": "af_bella",
        "michael": "am_michael",
    }

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        # Check if line contains speaker marker (e.g. "Beza: Hello")
        match = re.match(r"^([\w\s]+?)\s*:\s*(.+)$", line_str)
        if match:
            speaker_label = match.group(1).strip().lower()
            utterance = match.group(2).strip()
            
            # Map label to actual voice key if possible, else use default
            assigned_voice = name_to_key.get(speaker_label, default_voice)
            if utterance:
                parsed_turns.append({"voice": assigned_voice, "text": utterance})
        else:
            # If no explicit marker, treat the entire line as spoken by default voice
            parsed_turns.append({"voice": default_voice, "text": line_str})
            
    return parsed_turns


# ============================================================
# JOB ID & DIRECTORY
# ============================================================

def make_job_id(script, default_voice, speed):
    payload = f"{script}|{default_voice}|{speed:.4f}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


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


# ============================================================
# CHUNK GENERATION (MULTI-SPEAKER SUPPORTS TURNS)
# ============================================================

def generate_dialogue_chunk(
    kokoro,
    turn_text,
    voice_key,
    speed,
    output_path,
):
    samples, sample_rate = kokoro.create(
        turn_text,
        voice=voice_key,
        speed=float(speed),
        lang="en-us",
    )

    samples = np.asarray(samples, dtype=np.float32)
    sample_rate = int(sample_rate)

    sf.write(
        str(output_path),
        samples,
        sample_rate,
        subtype="PCM_16",
    )

    del samples
    gc.collect()
    return sample_rate


# ============================================================
# MP3 CONVERSION
# ============================================================

def convert_wav_to_mp3(wav_path, mp3_path):
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is not installed.") from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "128k",
        "-ar",
        str(OUTPUT_SAMPLE_RATE),
        str(mp3_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError("FFmpeg MP3 conversion failed:\n\n" + result.stderr[-3000:])


# ============================================================
# PREVIEW VOICE AS MP3
# ============================================================

def generate_preview_mp3(kokoro, voice, speed=1.0):
    preview_text = "Hello. This is a quick preview of this multi-person voice persona."
    samples, sample_rate = kokoro.create(
        preview_text,
        voice=voice,
        speed=float(speed),
        lang="en-us",
    )

    samples = np.asarray(samples, dtype=np.float32)
    sample_rate = int(sample_rate)

    samples_int16 = np.clip(samples, -1.0, 1.0)
    samples_int16 = (samples_int16 * 32767).astype(np.int16)
    raw_audio = samples_int16.tobytes()

    del samples
    del samples_int16
    gc.collect()

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is not installed.") from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "128k",
        "-ar",
        str(OUTPUT_SAMPLE_RATE),
        "-f",
        "mp3",
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        input=raw_audio,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError("Could not create MP3 preview.")

    return result.stdout


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🎙️ Voice & Dialogue Settings")

    default_voice_name = st.selectbox(
        "Default / Speaker 1 Voice",
        list(VOICE_MAP.keys()),
        index=0,
    )
    default_voice_key = VOICE_MAP[default_voice_name]

    secondary_voice_name = st.selectbox(
        "Speaker 2 Voice (For alternating turns)",
        list(VOICE_MAP.keys()),
        index=5,
    )
    secondary_voice_key = VOICE_MAP[secondary_voice_name]

    speed = st.slider(
        "Speech speed",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
    )

    st.divider()

    st.subheader("🎧 Voice Preview")
    if st.button("▶️ Preview Default Voice", use_container_width=True):
        with st.spinner("Generating preview..."):
            try:
                kokoro = get_kokoro_engine()
                preview_mp3 = generate_preview_mp3(kokoro, default_voice_key, speed)
                st.audio(preview_mp3, format="audio/mpeg")
            except Exception as exc:
                st.error("Could not generate voice preview.")
                st.exception(exc)

    st.divider()

    st.subheader("🧹 Job Controls")
    if st.button("Start Fresh", use_container_width=True):
        st.session_state.current_job = None
        st.rerun()


# ============================================================
# MAIN SCRIPT AREA
# ============================================================

st.header("📜 Dialogue Script Area")

script = st.text_area(
    "Paste your conversation script here",
    height=420,
    placeholder=(
        "Format your script with speaker tags like:\n\n"
        "af_heart: Hey, did you test the multi-person local synthesis setup?\n"
        "am_adam: Yes! It merges turns seamlessly without online rate limits."
    ),
    label_visibility="collapsed",
)

# Parse script turns
parsed_turns = parse_dialogue_script(script, default_voice_key) if script else []
total_turns = len(parsed_turns)
total_words = sum(len(turn["text"].split()) for turn in parsed_turns)
estimated_minutes = total_words / 120 if total_words else 0


# ============================================================
# SCRIPT STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Words", f"{total_words:,}")
with col2:
    st.metric("Dialogue Turns", f"{total_turns:,}")
with col3:
    st.metric("Estimated Duration", f"{estimated_minutes:.1f} min")
with col4:
    st.metric("Speakers Mode", "Dual-Voice")


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    "🎙️ Generate Conversation Audio",
    type="primary",
    use_container_width=True,
    disabled=not bool(parsed_turns),
)


# ============================================================
# GENERATION PIPELINE
# ============================================================

if generate_button:
    if total_words < 3:
        st.error("Please enter a longer script.")
        st.stop()

    job_id = make_job_id(script, default_voice_key, speed)
    job_dir = create_job_directory(job_id)

    final_wav_path = job_dir / "final_narration.wav"
    final_mp3_path = job_dir / "final_narration.mp3"

    if final_mp3_path.exists() and final_mp3_path.stat().st_size > 0:
        st.session_state.current_job = {
            "job_id": job_id,
            "work_dir": str(job_dir),
            "mp3_path": str(final_mp3_path),
        }
        st.success("Conversation audio already exists. Loading cached output.")
        st.rerun()

    st.session_state.current_job = {
        "job_id": job_id,
        "work_dir": str(job_dir),
        "mp3_path": None,
    }

    try:
        kokoro = get_kokoro_engine()
    except Exception as exc:
        st.error("Could not load the Kokoro model.")
        st.exception(exc)
        st.stop()

    progress = st.progress(0, text="Initializing multi-speaker synthesis...")
    status_box = st.empty()

    try:
        chunk_paths = []
        pause_duration = 0.35  # Natural pause between speakers in seconds

        for index, turn in enumerate(parsed_turns, start=1):
            chunk_path = get_chunk_path(job_dir, index)
            chunk_paths.append(chunk_path)

            if chunk_is_complete(chunk_path):
                progress.progress(index / total_turns, text=f"Recovered turn {index}/{total_turns}")
                continue

            status_box.markdown(
                f"""<div class="status-box">
                🎙️ Synthesizing turn <b>{index}</b> / <b>{total_turns}</b> using voice: <code>{turn['voice']}</code>
                </div>""",
                unsafe_allow_html=True,
            )

            # Generate individual turn audio
            sample_rate = generate_dialogue_chunk(
                kokoro,
                turn["text"],
                turn["voice"],
                speed,
                chunk_path,
            )

            progress.progress(index / total_turns, text=f"Completed turn {index}/{total_turns}")

        # Combine all sentence/turn files sequentially with minor breath/pacing silences
        status_box.markdown(
            f"""<div class="status-box">🔗 Merging dialogue audio tracks into final file...</div>""",
            unsafe_allow_html=True,
        )

        audio_segments = []
        silence_samples = np.zeros(int(OUTPUT_SAMPLE_RATE * pause_duration), dtype=np.float32)

        for cp in chunk_paths:
            data, sr = sf.read(str(cp), dtype="float32")
            audio_segments.append(data)
            audio_segments.append(silence_samples)

        if audio_segments:
            combined_audio = np.concatenate(audio_segments)
            sf.write(str(final_wav_path), combined_audio, OUTPUT_SAMPLE_RATE, subtype="PCM_16")

        # Convert to MP3
        convert_wav_to_mp3(final_wav_path, final_mp3_path)

        st.session_state.current_job["mp3_path"] = str(final_mp3_path)
        status_box.empty()
        progress.empty()
        st.success("✨ Multi-person conversation generated successfully!")
        st.rerun()

    except Exception as exc:
        st.error("An error occurred during multi-speaker audio generation.")
        st.exception(exc)


# ============================================================
# RENDER COMPLETED JOB OUTPUT
# ============================================================

current_job = st.session_state.current_job
if current_job and current_job.get("mp3_path"):
    mp3_file = Path(current_job["mp3_path"])
    if mp3_file.exists():
        st.divider()
        st.subheader("🎧 Generated Conversation Output")
        
        with open(mp3_file, "rb") as f:
            mp3_bytes = f.read()

        st.audio(mp3_bytes, format="audio/mpeg")
        
        st.download_button(
            label="📥 Download Conversation MP3",
            data=mp3_bytes,
            file_name="lenchos_studio_conversation.mp3",
            mime="audio/mpeg",
            use_container_width=True,
        )
