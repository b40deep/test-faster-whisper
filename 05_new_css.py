import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import gradio as gr
import torch
import torchaudio.functional as F

# Settings
samplerate = 16000
block_duration = 1  # seconds - length of blocks of audio captured
chunk_duration = 2    # seconds - if you don't speak, then it will process
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# Model setup: medium.en + float16 (optimized for 3080)
model = WhisperModel("base", device="cuda", compute_type="float32")  # use model size "medium.en" for faster results but slightly less accuracy

def resample_audio_torchaudio(audio_data, orig_sr, target_sr):
    """Resample audio using torchaudio for high quality and speed"""
    if orig_sr == target_sr:
        return audio_data
    
    # Convert numpy to torch tensor
    audio_tensor = torch.from_numpy(audio_data).float()
    
    # Resample using torchaudio
    resampled_tensor = F.resample(audio_tensor, orig_sr, target_sr)
    
    return resampled_tensor.numpy().astype(np.float32)

def get_transcription(stream):
    # Transcription without timestamps
    for lang in ["en"]:
        segments, _ = model.transcribe(
            stream,
            language=lang,
            beam_size=1  # Max speed
        )

        text_output = ""
        for segment in segments:
            text_output += segment.text + " "
        print(f"\tTranscription: {text_output.strip()}")
        return text_output.strip()

def transcribe(stream, new_chunk):
    print(f"transcribe...\n\t{stream}\n\t{new_chunk}")
    if new_chunk is None:
        return stream, ""
    
    sr, y = new_chunk
    target_sr = 16000
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Convert to float32 and normalize    
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Resample to 16kHz using torchaudio (high quality)
    if sr != target_sr:
        y = resample_audio_torchaudio(y, sr, target_sr)
        sr = target_sr

    # Concatenate with previous audio stream
    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    # Only transcribe if we have enough audio (equivalent to old chunk_duration)
    min_samples = int(sr * block_duration)  # seconds minimum
    if len(stream) < min_samples:
        return stream, "", "", ""
    else:
        res = get_transcription(stream)
        return stream, res, res, res

# Gradio interface
def gradio_interface():
    print("Creating Gradio css...")
    gradio_css = """
    .txtbox textarea{
        height: 27vh !important;
        font-size: 4em !important;
    }
    """

    print("Creating Gradio parts...")
    mic_audio =  gr.Audio(elem_classes="mic", show_label=False, sources=["microphone"], type="numpy", streaming=True)
    txt_output_1 = gr.Textbox(elem_classes="txtbox", show_label=False, max_lines=3, autoscroll=True)
    txt_output_2 = gr.Textbox(elem_classes="txtbox", show_label=False, max_lines=3, autoscroll=True)
    txt_output_3 = gr.Textbox(elem_classes="txtbox", show_label=False, max_lines=3, autoscroll=True)

    print("Creating Gradio interface...")
    with gr.Blocks(css=gradio_css) as demo:
        state = gr.State()  # For internal state tracking
        with gr.Row():
            with gr.Column():
                mic_audio.render()
        with gr.Row():
            with gr.Column():
                txt_output_1.render()
        with gr.Row():
            with gr.Column():
                txt_output_2.render()
        with gr.Row():
            with gr.Column():
                txt_output_3.render()

        mic_audio.change(show_progress='hidden',
            fn=transcribe,
            inputs=[state, mic_audio],
            outputs=[state, txt_output_1, txt_output_2, txt_output_3],
            # live=True,
        )
    return demo

if __name__ == "__main__":
    gradio_app = gradio_interface()
    gradio_app.launch(share=False, debug=True)
