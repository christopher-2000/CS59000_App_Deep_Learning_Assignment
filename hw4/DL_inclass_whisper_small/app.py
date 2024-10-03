import gradio as gr
from transformers import pipeline

# Load the Whisper model for speech recognition
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
    device="cpu",  # Force to run on CPU
)

# Define the function to process audio
def transcribe_audio(audio_file):
    # Use Whisper model to transcribe audio
    prediction = pipe(audio_file)["text"]
    return prediction

# Define Gradio interface
interface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),  # Remove 'source' argument
    outputs="text",
    title="Speech-to-Text with Whisper",
    description="Upload an audio file to transcribe speech to text using the Whisper model."
)

# Launch the Gradio app
interface.launch()
