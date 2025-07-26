import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cpu")  # or "cpu" if needed
# model = ChatterboxTTS.from_pretrained(device="cuda")


print("Using GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


text = "Chinmay is a great guy and he is very good at coding.I love him. Please fuck me chinmay"

# Your short voice sample path (must be a WAV file)
AUDIO_PROMPT_PATH = "divya.wav"

# Generate speech
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)

# Save to file
ta.save("cloned_output.wav", wav, model.sr)

print("✅ Output saved to cloned_output.wav")
