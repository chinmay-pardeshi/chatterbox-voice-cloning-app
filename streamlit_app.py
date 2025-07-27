import streamlit as st
import os
import sys
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from st_audiorec import st_audiorec  # 🧿 streamlit-audio-recorder

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# App UI
st.title("🎙️ Chatterbox Voice Cloning")
st.markdown("Upload or record your voice, enter text, and clone the voice!")

# Load model
@st.cache_resource
def load_model():
    return ChatterboxTTS.from_pretrained(device="cpu")  # or "cuda" if working

model = load_model()

# --- RECORDING SECTION ---
st.subheader("Step 1: Record your voice or upload a WAV file")

use_rec = st.toggle("🎤 Use microphone to record")
recorded_audio = None

if use_rec:
    audio_bytes = st_audiorec()
    if audio_bytes:
        with open("input_prompt.wav", "wb") as f:
            f.write(audio_bytes)
        st.audio("input_prompt.wav", format="audio/wav")
        recorded_audio = "input_prompt.wav"
        st.success("✅ Voice recorded successfully!")
else:
    uploaded_file = st.file_uploader("Or upload a .wav file", type=["wav"])
    if uploaded_file:
        with open("input_prompt.wav", "wb") as f:
            f.write(uploaded_file.read())
        st.audio("input_prompt.wav", format="audio/wav")
        recorded_audio = "input_prompt.wav"
        st.success("✅ Voice uploaded successfully!")

# --- TEXT & GENERATE ---
st.subheader("Step 2: Enter text and clone your voice")

text = st.text_area("📝 Enter the text you want to speak:", "Hello, I am your cloned voice!")

if st.button("🔊 Generate Voice") and recorded_audio and text.strip():
    wav = model.generate(text, audio_prompt_path=recorded_audio)
    ta.save("cloned_output.wav", wav, model.sr)
    st.audio("cloned_output.wav")
    st.success("✅ Cloned voice generated!")
