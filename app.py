import os
import tempfile
import numpy as np
import soundfile as sf
import gradio as gr
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')

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

def generate_speech(text, voice_display_name, speed):
    if not text.strip():
        return None, None
    actual_voice = VOICE_MAP.get(voice_display_name, 'af_heart')
    generator = pipeline(text, voice=actual_voice, speed=speed)
    audio_chunks = [audio for _, _, audio in generator]
    if not audio_chunks:
        return None, None
    full_audio = np.concatenate(audio_chunks)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, full_audio, 24000)
    return temp_file.name, temp_file.name

with gr.Blocks(title="Lencho Voice Lab") as demo:
    gr.Markdown("# 🎧 Lencho Voice Lab")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Input Script", lines=6, placeholder="Type your text here...")
            voice_select = gr.Dropdown(choices=list(VOICE_MAP.keys()), value="Hinsene (American Female - Warm)", label="Voice Persona")
            speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed Rate")
            generate_btn = gr.Button("✨ Generate Audio", variant="primary")
        with gr.Column():
            audio_output = gr.Audio(label="Rendered Speech", type="filepath")
            download_file = gr.File(label="Download WAV File")

    generate_btn.click(
        fn=generate_speech, 
        inputs=[text_input, voice_select, speed_slider], 
        outputs=[audio_output, download_file]
    )

if __name__ == "__main__":
    # Render provides an environment variable named PORT
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
