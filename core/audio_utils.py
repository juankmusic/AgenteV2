# core/audio_utils.py
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
from core.chatbot_logic import OPENAI_CLIENT  # ✅ reutilizamos cliente OpenAI

def grabar_audio(duracion=5, fs=44100):
    print(f"🎤 Habla ahora (duración {duracion}s)...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(tmp_file.name, fs, audio)
    return tmp_file.name

def transcribir_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = OPENAI_CLIENT.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            language="es"
        )
    return transcript.text
