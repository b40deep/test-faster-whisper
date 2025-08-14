import numpy as np
from faster_whisper import WhisperModel
import gradio as gr

# Model setup: base + float32 for CUDA
model = WhisperModel("base", device="cuda", compute_type="float32")

def transcribe(stream, new_chunk):
    if new_chunk is None:
        return stream, ""
    
    sr, y = new_chunk
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
    
    # Convert to float32 and normalize
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
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
    
    # Transcription without timestamps
    try:
        segments, _ = model.transcribe(
            stream,
            language="en",
            beam_size=2  # Max speed
        )
        
        text_output = ""
        for segment in segments:
            text_output += segment.text + " "
        
        return stream, text_output.strip()
    
    except Exception as e:
        return stream, f"Transcription error: {str(e)}"

# Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### Real-time Audio Transcription")
        gr.Markdown("Speak into your microphone for live transcription using Whisper")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone"], 
                    streaming=True,
                    label="Microphone Input"
                )
            with gr.Column():
                output = gr.Textbox(
                    label="Live Transcription", 
                    lines=10,
                    max_lines=20
                )
        
        # Hidden state to store audio stream
        audio_state = gr.State()
    
    # Set up the streaming interface
    demo = gr.Interface(
        fn=transcribe,
        inputs=[audio_state, audio_input],
        outputs=[audio_state, output],
        live=True,
        title="Real-time Audio Transcription",
        description="Speak into your microphone for live transcription",
        stream_every=0.1,  # Process every 0.5 seconds
        # time_limit=60     # 60 second limit per session
    )
    
    return demo

if __name__ == "__main__":
    gradio_app = gradio_interface()
    gradio_app.launch(share=True)