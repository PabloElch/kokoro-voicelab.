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
    .hero-container {
        padding: 28px 10px 20px 10px;
        text-align: center;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .hero-subtitle {
        font-size: 15px;
        opacity: 0.75;
        margin-top: 8px;
    }

    .status-box {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.25);
        margin-top: 10px;
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
    <div class="hero-container">
        <div class="hero-title">
            🎙️ LENCHOS AUDIO STUDIO
        </div>
        <div class="hero-subtitle">
            Open-source AI narration for calm, long-form audio.
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

    model_path = (
        model_dir
        / MODEL_FILENAME
    )

    voices_path = (
        model_dir
        / VOICES_FILENAME
    )

    download_file(
        MODEL_URL,
        model_path,
    )

    download_file(
        VOICES_URL,
        voices_path,
    )

    kokoro = Kokoro(
        str(model_path),
        str(voices_path),
    )

    return kokoro


# ============================================================
# TEXT PROCESSING
# ============================================================

def normalize_script(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):

        cleaned = re.sub(
            r"\s+",
            " ",
            line.strip(),
        )

        if cleaned:
            lines.append(cleaned)

    return "\n\n".join(lines)


def split_long_sentence(
    sentence,
    target_words,
):
    words = sentence.split()

    if len(words) <= target_words:
        return [sentence.strip()]

    pieces = []

    for start in range(
        0,
        len(words),
        target_words,
    ):

        piece = " ".join(
            words[
                start:start + target_words
            ]
        )

        if piece.strip():
            pieces.append(
                piece.strip()
            )

    return pieces


def split_script_into_chunks(
    text,
    target_words=TARGET_WORDS_PER_CHUNK,
):

    text = normalize_script(text)

    if not text:
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    sentences = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        paragraph_sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph,
        )

        for sentence in paragraph_sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentences.extend(
                split_long_sentence(
                    sentence,
                    target_words,
                )
            )

    chunks = []

    current = []
    current_words = 0

    for sentence in sentences:

        sentence_words = len(
            sentence.split()
        )

        if (
            current
            and current_words + sentence_words
            > target_words
        ):

            chunks.append(
                " ".join(current).strip()
            )

            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:

        chunks.append(
            " ".join(current).strip()
        )

    return chunks


# ============================================================
# JOB ID
# ============================================================

def make_job_id(
    script,
    voice,
    speed,
):

    payload = (
        f"{script}|"
        f"{voice}|"
        f"{speed:.4f}"
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return digest[:16]


# ============================================================
# JOB DIRECTORY
# ============================================================

def create_job_directory(job_id):

    base_dir = get_base_work_dir()

    job_dir = (
        base_dir
        / f"job_{job_id}"
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_dir = (
        job_dir / "chunks"
    )

    chunks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return job_dir


# ============================================================
# CHUNK PATH
# ============================================================

def get_chunk_path(
    job_dir,
    index,
):

    return (
        Path(job_dir)
        / "chunks"
        / f"chunk_{index:03d}.wav"
    )


# ============================================================
# CHUNK VALIDATION
# ============================================================

def chunk_is_complete(path):

    path = Path(path)

    if not path.exists():
        return False

    if path.stat().st_size < 1000:
        return False

    try:

        info = sf.info(
            str(path)
        )

        return (
            info.frames > 0
            and info.samplerate > 0
        )

    except Exception:
        return False


# ============================================================
# GENERATE ONE CHUNK
# ============================================================

def generate_chunk(
    kokoro,
    text,
    voice,
    speed,
    output_path,
):

    samples, sample_rate = (
        kokoro.create(
            text,
            voice=voice,
            speed=float(speed),
            lang="en-us",
        )
    )

    samples = np.asarray(
        samples,
        dtype=np.float32,
    )

    sample_rate = int(
        sample_rate
    )

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
# COMBINE WAV CHUNKS
# ============================================================

def combine_wav_files(
    chunk_paths,
    output_path,
):

    if not chunk_paths:
        raise ValueError(
            "No audio chunks were found."
        )

    first_info = sf.info(
        str(chunk_paths[0])
    )

    sample_rate = (
        first_info.samplerate
    )

    channels = (
        first_info.channels
    )

    with sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        subtype="PCM_16",
        format="WAV",
    ) as output_file:

        for path in chunk_paths:

            with sf.SoundFile(
                str(path),
                mode="r",
            ) as input_file:

                if (
                    input_file.samplerate
                    != sample_rate
                ):
                    raise ValueError(
                        "Chunk sample rates do not match."
                    )

                if (
                    input_file.channels
                    != channels
                ):
                    raise ValueError(
                        "Chunk channel counts do not match."
                    )

                while True:

                    block = (
                        input_file.read(
                            65536,
                            dtype="float32",
                        )
                    )

                    if len(block) == 0:
                        break

                    output_file.write(
                        block
                    )

                    del block

    gc.collect()


# ============================================================
# MP3 CONVERSION
# ============================================================

def convert_wav_to_mp3(
    wav_path,
    mp3_path,
):

    try:
        import imageio_ffmpeg

    except ImportError as exc:

        raise RuntimeError(
            "imageio-ffmpeg is not installed. "
            "Add imageio-ffmpeg to requirements.txt."
        ) from exc

    ffmpeg_exe = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )

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

        raise RuntimeError(
            "FFmpeg MP3 conversion failed:\n\n"
            + result.stderr[-3000:]
        )

    if (
        not mp3_path.exists()
        or mp3_path.stat().st_size == 0
    ):

        raise RuntimeError(
            "MP3 conversion completed but "
            "the MP3 file was not created."
        )


# ============================================================
# DELETE ALL WAV FILES
# ============================================================

def delete_all_wav_files(
    job_dir,
):

    job_dir = Path(job_dir)

    final_wav = (
        job_dir
        / "final_narration.wav"
    )

    if final_wav.exists():

        try:
            final_wav.unlink()
        except Exception:
            pass

    chunks_dir = (
        job_dir / "chunks"
    )

    if chunks_dir.exists():

        for wav_file in chunks_dir.glob(
            "*.wav"
        ):

            try:
                wav_file.unlink()
            except Exception:
                pass

        try:

            if not any(
                chunks_dir.iterdir()
            ):
                chunks_dir.rmdir()

        except Exception:
            pass

    gc.collect()


# ============================================================
# PREVIEW VOICE AS MP3
# ============================================================

def generate_preview_mp3(
    kokoro,
    voice,
    speed=1.0,
):

    preview_text = (
        "Hello. This is a quick preview "
        "of this voice persona. "
        "I hope you enjoy listening."
    )

    samples, sample_rate = (
        kokoro.create(
            preview_text,
            voice=voice,
            speed=float(speed),
            lang="en-us",
        )
    )

    samples = np.asarray(
        samples,
        dtype=np.float32,
    )

    sample_rate = int(
        sample_rate
    )

    # Convert float audio to signed 16-bit PCM.
    samples_int16 = np.clip(
        samples,
        -1.0,
        1.0,
    )

    samples_int16 = (
        samples_int16 * 32767
    ).astype(
        np.int16
    )

    raw_audio = (
        samples_int16
        .tobytes()
    )

    del samples
    del samples_int16

    gc.collect()

    try:
        import imageio_ffmpeg

    except ImportError as exc:

        raise RuntimeError(
            "imageio-ffmpeg is not installed."
        ) from exc

    ffmpeg_exe = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )

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

    del raw_audio

    gc.collect()

    if result.returncode != 0:

        raise RuntimeError(
            "Could not create MP3 preview:\n\n"
            + result.stderr.decode(
                "utf-8",
                errors="ignore",
            )[-3000:]
        )

    if not result.stdout:

        raise RuntimeError(
            "MP3 preview was empty."
        )

    return result.stdout


# ============================================================
# JOB STATUS
# ============================================================

def count_completed_chunks(
    job_dir,
    total_chunks,
):

    count = 0

    for index in range(
        1,
        total_chunks + 1,
    ):

        path = get_chunk_path(
            job_dir,
            index,
        )

        if chunk_is_complete(path):
            count += 1

    return count


# ============================================================
# HISTORY
# ============================================================

def add_history_item(item):

    history = (
        st.session_state.history
    )

    history = [
        old
        for old in history
        if old.get("job_id")
        != item.get("job_id")
    ]

    history.insert(
        0,
        item,
    )

    st.session_state.history = (
        history[:2]
    )


def cleanup_history_dirs():

    keep_dirs = set()

    current_job = (
        st.session_state.current_job
    )

    if current_job:

        work_dir = current_job.get(
            "work_dir"
        )

        if work_dir:
            keep_dirs.add(
                str(work_dir)
            )

    for item in (
        st.session_state.history
    ):

        work_dir = item.get(
            "work_dir"
        )

        if work_dir:
            keep_dirs.add(
                str(work_dir)
            )

    base_dir = get_base_work_dir()

    for item in base_dir.iterdir():

        if not item.is_dir():
            continue

        if item.name == "model":
            continue

        if str(item) not in keep_dirs:

            try:
                shutil.rmtree(item)
            except Exception:
                pass


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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎙️ Voice Settings")

    voice_name = st.selectbox(
        "Narrator",
        list(VOICE_MAP.keys()),
        index=9,
    )

    voice_key = VOICE_MAP[
        voice_name
    ]

    speed = st.slider(
        "Speech speed",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
    )

    st.caption(
        "For Slumber Tales English, "
        "around 0.90–1.00 is a good starting point."
    )

    st.divider()

    st.subheader("🎧 Voice Preview")

    if st.button(
        "▶️ Preview Voice",
        use_container_width=True,
    ):

        with st.spinner(
            "Generating MP3 preview..."
        ):

            try:

                kokoro = (
                    get_kokoro_engine()
                )

                preview_mp3 = (
                    generate_preview_mp3(
                        kokoro,
                        voice_key,
                        speed,
                    )
                )

                st.success(
                    "Preview ready."
                )

                st.audio(
                    preview_mp3,
                    format="audio/mpeg",
                )

            except Exception as exc:

                st.error(
                    "Could not generate voice preview."
                )

                st.exception(exc)

    st.divider()

    st.subheader("⚙️ Chunking")

    st.write(
        f"Target chunk size: "
        f"**{TARGET_WORDS_PER_CHUNK} words**"
    )

    st.caption(
        "Approximately 4–5 minutes of narration "
        "at a calm speaking pace."
    )

    st.divider()

    st.subheader("🧹 Job Controls")

    if st.button(
        "Start Fresh",
        use_container_width=True,
    ):

        current_job = (
            st.session_state.current_job
        )

        if current_job:

            work_dir = current_job.get(
                "work_dir"
            )

            if work_dir:

                try:
                    shutil.rmtree(
                        work_dir
                    )
                except Exception:
                    pass

        st.session_state.current_job = None

        st.rerun()


# ============================================================
# MAIN SCRIPT AREA
# ============================================================

st.header("📜 Narration Script")

script = st.text_area(
    "Paste your story here",
    height=420,
    placeholder=(
        "Paste your bedtime story here...\n\n"
        "For example:\n"
        "Welcome to Slumber Tales English. "
        "Tonight, we are going to visit a little "
        "cottage at the edge of a quiet forest..."
    ),
    label_visibility="collapsed",
)

normalized_script = normalize_script(
    script
)

word_count = (
    len(normalized_script.split())
    if normalized_script
    else 0
)

character_count = len(
    normalized_script
)

estimated_minutes = (
    word_count / 120
    if word_count
    else 0
)

chunks_preview = (
    split_script_into_chunks(
        normalized_script
    )
    if normalized_script
    else []
)

chunk_count = len(
    chunks_preview
)


# ============================================================
# SCRIPT STATISTICS
# ============================================================

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Words",
        f"{word_count:,}",
    )

with col2:

    st.metric(
        "Characters",
        f"{character_count:,}",
    )

with col3:

    st.metric(
        "Estimated duration",
        f"{estimated_minutes:.1f} min",
    )

with col4:

    st.metric(
        "Audio chunks",
        f"{chunk_count}",
    )


if chunk_count > 0:

    st.caption(
        f"Your script will be processed "
        f"as approximately {chunk_count} "
        f"independent chunks."
    )


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    "🎙️ Generate Narration",
    type="primary",
    use_container_width=True,
    disabled=not bool(
        normalized_script
    ),
)


# ============================================================
# GENERATION PIPELINE
# ============================================================

if generate_button:

    if word_count < 5:

        st.error(
            "Please enter a longer script."
        )

        st.stop()

    # --------------------------------------------------------
    # JOB ID
    # --------------------------------------------------------

    job_id = make_job_id(
        normalized_script,
        voice_key,
        speed,
    )

    job_dir = (
        create_job_directory(
            job_id
        )
    )

    final_wav_path = (
        job_dir
        / "final_narration.wav"
    )

    final_mp3_path = (
        job_dir
        / "final_narration.mp3"
    )

    # --------------------------------------------------------
    # CHECK IF FINAL MP3 ALREADY EXISTS
    # --------------------------------------------------------

    if (
        final_mp3_path.exists()
        and final_mp3_path.stat().st_size > 0
    ):

        st.session_state.current_job = {
            "job_id": job_id,
            "work_dir": str(job_dir),
            "voice": voice_key,
            "voice_name": voice_name,
            "speed": speed,
            "word_count": word_count,
            "total_chunks": chunk_count,
            "completed_chunks": chunk_count,
            "mp3_path": str(
                final_mp3_path
            ),
        }

        st.success(
            "This narration already exists. "
            "Using the completed MP3."
        )

        st.rerun()

    # --------------------------------------------------------
    # CURRENT JOB METADATA
    # --------------------------------------------------------

    st.session_state.current_job = {
        "job_id": job_id,
        "work_dir": str(job_dir),
        "voice": voice_key,
        "voice_name": voice_name,
        "speed": speed,
        "word_count": word_count,
        "total_chunks": chunk_count,
        "completed_chunks": 0,
        "mp3_path": None,
    }

    # --------------------------------------------------------
    # LOAD KOKORO
    # --------------------------------------------------------

    try:

        kokoro = (
            get_kokoro_engine()
        )

    except Exception as exc:

        st.error(
            "Could not load the Kokoro model."
        )

        st.exception(exc)

        st.stop()

    # --------------------------------------------------------
    # PROGRESS UI
    # --------------------------------------------------------

    progress = st.progress(
        0,
        text="Preparing narration...",
    )

    status_box = st.empty()

    # --------------------------------------------------------
    # GENERATE CHUNKS
    # --------------------------------------------------------

    try:

        for index, chunk_text in enumerate(
            chunks_preview,
            start=1,
        ):

            chunk_path = (
                get_chunk_path(
                    job_dir,
                    index,
                )
            )

            # --------------------------------------------
            # RECOVERY
            # --------------------------------------------

            if chunk_is_complete(
                chunk_path
            ):

                status_box.markdown(
                    f"""
                    <div class="status-box">
                    ♻️ Chunk <b>{index}</b> /
                    <b>{chunk_count}</b> already exists.
                    Skipping regeneration.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                progress.progress(
                    index / chunk_count,
                    text=(
                        f"Recovered chunk "
                        f"{index}/{chunk_count}"
                    ),
                )

                continue

            # --------------------------------------------
            # GENERATE
            # --------------------------------------------

            status_box.markdown(
                f"""
                <div class="status-box">
                🎙️ Generating chunk
                <b>{index}</b> /
                <b>{chunk_count}</b>...
                </div>
                """,
                unsafe_allow_html=True,
            )

            progress.progress(
                (index - 1)
                / chunk_count,
                text=(
                    f"Generating chunk "
                    f"{index}/{chunk_count}"
                ),
            )

            generate_chunk(
                kokoro=kokoro,
                text=chunk_text,
                voice=voice_key,
                speed=speed,
                output_path=chunk_path,
            )

            gc.collect()

            progress.progress(
                index / chunk_count,
                text=(
                    f"Completed chunk "
                    f"{index}/{chunk_count}"
                ),
            )

        # ----------------------------------------------------
        # VERIFY CHUNKS
        # ----------------------------------------------------

        chunk_paths = []

        for index in range(
            1,
            chunk_count + 1,
        ):

            path = get_chunk_path(
                job_dir,
                index,
            )

            if not chunk_is_complete(
                path
            ):

                raise RuntimeError(
                    f"Chunk {index} is missing "
                    f"or invalid."
                )

            chunk_paths.append(
                path
            )

        # ----------------------------------------------------
        # COMBINE WAV
        # ----------------------------------------------------

        status_box.markdown(
            """
            <div class="status-box">
            🔗 Combining completed chunks...
            </div>
            """,
            unsafe_allow_html=True,
        )

        progress.progress(
            0.90,
            text="Combining audio chunks...",
        )

        combine_wav_files(
            chunk_paths,
            final_wav_path,
        )

        gc.collect()

        # ----------------------------------------------------
        # CONVERT TO MP3
        # ----------------------------------------------------

        status_box.markdown(
            """
            <div class="status-box">
            🎧 Creating final MP3...
            </div>
            """,
            unsafe_allow_html=True,
        )

        progress.progress(
            0.96,
            text="Creating MP3...",
        )

        convert_wav_to_mp3(
            final_wav_path,
            final_mp3_path,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # DELETE EVERY WAV AFTER MP3 SUCCEEDS
        # ----------------------------------------------------

        delete_all_wav_files(
            job_dir
        )

        # ----------------------------------------------------
        # FINISHED
        # ----------------------------------------------------

        progress.progress(
            1.0,
            text="Narration complete!",
        )

        status_box.success(
            f"✅ Finished {chunk_count} chunks. "
            f"Final MP3 is ready."
        )

        completed_job = {
            "job_id": job_id,
            "work_dir": str(job_dir),
            "voice": voice_key,
            "voice_name": voice_name,
            "speed": speed,
            "word_count": word_count,
            "total_chunks": chunk_count,
            "completed_chunks": chunk_count,
            "mp3_path": str(
                final_mp3_path
            ),
        }

        st.session_state.current_job = (
            completed_job
        )

        add_history_item(
            completed_job
        )

        cleanup_history_dirs()

    except Exception as exc:

        st.error(
            "Generation stopped."
        )

        st.exception(exc)

        completed = (
            count_completed_chunks(
                job_dir,
                chunk_count,
            )
        )

        st.info(
            f"Recovery information: "
            f"{completed}/{chunk_count} "
            f"chunks are already complete."
        )

        st.warning(
            "You can press Generate Narration "
            "again with the same settings. "
            "Existing completed chunks will be skipped."
        )


# ============================================================
# CURRENT RESULT
# ============================================================

current_job = (
    st.session_state.current_job
)

if current_job:

    mp3_path = current_job.get(
        "mp3_path"
    )

    work_dir = current_job.get(
        "work_dir"
    )

    st.divider()

    st.header("🎧 Current Narration")

    if (
        mp3_path
        and os.path.exists(mp3_path)
    ):

        st.subheader("MP3")

        st.audio(
            mp3_path,
            format="audio/mpeg",
        )

        mp3_size_mb = (
            os.path.getsize(mp3_path)
            / (1024 * 1024)
        )

        st.caption(
            f"MP3 size: "
            f"{mp3_size_mb:.1f} MB"
        )

        with open(
            mp3_path,
            "rb",
        ) as mp3_file:

            st.download_button(
                "⬇️ Download MP3",
                data=mp3_file,
                file_name=(
                    "slumber_tales_narration.mp3"
                ),
                mime="audio/mpeg",
                use_container_width=True,
                key="download_mp3_current",
            )

        st.success(
            "Your final MP3 is ready. "
            "All WAV files have been deleted."
        )

    if work_dir:

        st.caption(
            "Only the final MP3 is kept after "
            "successful conversion."
        )


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.divider()

    st.header("🕘 Recent Jobs")

    for number, item in enumerate(
        st.session_state.history,
        start=1,
    ):

        job_mp3 = item.get(
            "mp3_path"
        )

        label = (
            f"{number}. "
            f"{item.get('voice_name', 'Kokoro')} — "
            f"{item.get('word_count', 0):,} words — "
            f"{item.get('total_chunks', 0)} chunks"
        )

        with st.expander(label):

            st.write(
                f"**Voice:** "
                f"{item.get('voice_name', '-')}"
            )

            st.write(
                f"**Speed:** "
                f"{item.get('speed', '-')}"
            )

            st.write(
                f"**Words:** "
                f"{item.get('word_count', 0):,}"
            )

            st.write(
                f"**Chunks:** "
                f"{item.get('total_chunks', 0)}"
            )

            if (
                job_mp3
                and os.path.exists(job_mp3)
            ):

                st.audio(
                    job_mp3,
                    format="audio/mpeg",
                )

                with open(
                    job_mp3,
                    "rb",
                ) as mp3_file:

                    st.download_button(
                        "⬇️ Download MP3",
                        data=mp3_file,
                        file_name=(
                            "slumber_tales_narration.mp3"
                        ),
                        mime="audio/mpeg",
                        key=(
                            f"history_mp3_{number}_"
                            f"{item.get('job_id')}"
                        ),
                    )

            else:

                st.caption(
                    "MP3 file is no longer available "
                    "in the current Streamlit session."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Lenchos Audio Studio • Kokoro ONNX • "
    "Chunked MP3 generation with disk-based recovery"
)
