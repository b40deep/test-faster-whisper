import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import gradio as gr

# Settings
samplerate = 16000
block_duration = 3  # seconds - length of blocks of audio captured
chunk_duration = 2    # seconds - if you don't speak, then it will process
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# Model setup: medium.en + float16 (optimized for 3080)
model = WhisperModel("base", device="cuda", compute_type="float32")  # use model size "medium.en" for faster results but slightly less accuracy


def get_transcription(stream):
    print("get transcription...")
    # Transcription without timestamps
    for lang in ["en"]:
        segments, _ = model.transcribe(
            stream,
            language=lang,
            beam_size=2  # Max speed
        )

        text_output = ""
        for segment in segments:
            text_output += segment.text + " "
        return text_output.strip()

def transcribe(stream, new_chunk):
    print("transcribe...")
    sr, y = new_chunk
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Convert to float32 and normalize    
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # Concatenate with previous audio stream
    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    # Only transcribe if we have enough audio (equivalent to old chunk_duration)
    min_samples = int(sr * 5)  # 2 seconds minimum
    if len(stream) < min_samples:
        return stream, ""

    return stream, get_transcription(stream)  

print("Creating Gradio interface...")
demo = gr.Interface(
    transcribe,
    ["state", gr.Audio(sources=["microphone"], streaming=True)],
    ["state", "text"],
    live=True,
)

demo.launch(share=True, debug=True)


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    print("Starting recorder with input:")
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        callback=audio_callback, blocksize=frames_per_block):
        print("🎙 Listening... Press Ctrl+C to stop.")
        while True:
            sd.sleep(100)

def transcriber():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)

        total_frames = sum(len(b) for b in audio_buffer)
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = []  # Clear buffer

            audio_data = audio_data.flatten().astype(np.float32)

            # Transcription without timestamps
            for lang in ["en"]:
                segments, _ = model.transcribe(
                    audio_data,
                    language=lang,
                    beam_size=2  # Max speed
                )

                for segment in segments:
                    print(f"{segment.text}")  # Just print text, no timestamps

# Start threads
def start_threads(mic_input):
    print("Starting transcription...")
    threading.Thread(target=recorder, daemon=True).start()
    transcriber()