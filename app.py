import gradio as gr
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
import torch

# Load the model once
model = ChatterboxTTS.from_pretrained(device="cpu")  # or "cuda" if GPU fits

def clone_voice(audio, text, exaggeration, cfg_weight):
    # Save uploaded file
    input_path = "input_prompt.wav"
    # ta.save(input_path, audio[1], audio[0])
    # ta.save(input_path, audio[1].unsqueeze(0), audio[0])
    

    waveform = torch.tensor(audio[1]).unsqueeze(0)  # convert NumPy → Torch & add channel
    ta.save(input_path, waveform, audio[0])


    # Generate speech
    output = model.generate(
        text,
        audio_prompt_path=input_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight
    )

    # Save output
    output_path = "cloned_output.wav"
    ta.save(output_path, output, model.sr)

    return output_path

# Create the interface
app = gr.Interface(
    fn=clone_voice,
    inputs=[
        gr.Audio(label="Upload Voice Prompt (.wav)", type="numpy"),
        gr.Textbox(label="Text to Synthesize"),
        gr.Slider(0, 1, value=0.5, step=0.1, label="Exaggeration"),
        gr.Slider(0, 1, value=0.5, step=0.1, label="CFG Weight")
    ],
    outputs=gr.Audio(label="Cloned Voice"),
    title="🗣️ Chatterbox Voice Cloner",
    description="Upload a short voice sample (20–60s) and input your message to hear the cloned voice."
)

# Launch the app
if __name__ == "__main__":
    app.launch()
