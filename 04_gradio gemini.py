import numpy as np
import gradio as gr
from faster_whisper import WhisperModel
import random

# --- Settings ---
# Sample rate for the audio input, standard for speech
samplerate = 16000

# Length of audio chunks to process in seconds
# A larger chunk_duration can improve accuracy but increase latency
chunk_duration = 3

# Calculate the number of frames per chunk based on the sample rate
frames_per_chunk = int(samplerate * chunk_duration)

# Model setup: We're using a smaller model for faster processing.
# 'base' is a good balance for real-time.
# 'float32' is used for compute type, but you can change this
# depending on your GPU and memory (e.g., 'float16' for better performance on newer GPUs).
try:
    print("CUDA available. Using it for transcription.")
    model = WhisperModel("base", device="cuda", compute_type="float32")
except ValueError:
    print("CUDA not available. Using CPU for transcription. This may be slower.")
    model = WhisperModel("base", device="cpu", compute_type="float32")

def transcribe_and_format(audio_chunk, buffer_state, full_transcript):
    """
    This single function now handles both transcription and formatting.
    It takes the old transcript and appends the new one.
    """
    print(f"running transcribe_and_format {random.randint(10, 20)}")
    _, audio_data = audio_chunk
    buffer_state.append(audio_data)
    total_frames = sum(len(b) for b in buffer_state)

    # Initialize new_text to an empty string
    new_text = ""

    if total_frames >= frames_per_chunk:
        combined_audio = np.concatenate(buffer_state)
        audio_to_process = combined_audio[:frames_per_chunk]
        buffer_state = [combined_audio[frames_per_chunk:]] # Keep leftover audio

        audio_to_process = audio_to_process.flatten().astype(np.float32)

        segments, _ = model.transcribe(
            audio_to_process,
            language="en",
            beam_size=2
        )
        new_text = " ".join(segment.text for segment in segments).strip()
        print(f"\tNT__{new_text}")

    # Append the new text to the existing full transcript
    updated_transcript = f"{full_transcript} {new_text}" if new_text else full_transcript
    updated_transcript = updated_transcript.strip()
    print(f"\tBF__{buffer_state}")
    print(f"\tTR__{updated_transcript}")
    # Return the updated buffer and the updated full transcript
    return buffer_state, updated_transcript

# Gradio Interface
with gr.Blocks() as demo:
    print("Setting up Gradio interface...")
    gr.Markdown("### Real-time Audio Transcription with Gradio")
    gr.Markdown("Start speaking into the microphone, and the transcription will appear below as you talk.")

    # State variables 
    audio_buffer_state = gr.State(value=[])
    # The textbox itself will hold the transcription history.

    # Microphone component for real-time streaming
    mic_stream = gr.Microphone(
        sources="microphone",
        type="numpy",
        streaming=True,
        label="Microphone Input"
    )

    # Textbox to display the output
    output_textbox = gr.Textbox(label="Transcription Output", lines=10)
    
    # Connect the microphone directly to a single, combined function.
    mic_stream.stream(
        transcribe_and_format,
        inputs=[mic_stream, audio_buffer_state, output_textbox],
        outputs=[audio_buffer_state, output_textbox]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
