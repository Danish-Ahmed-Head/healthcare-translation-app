import streamlit as st
import os, tempfile, io
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
from pydub import AudioSegment
import ffmpeg_static
import speech_recognition as sr
import openai
from deep_translator import GoogleTranslator

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Healthcare Translation Assistant", layout="centered")
st.title("🩺 Healthcare Translation Assistant (Patient ↔ Doctor)")
st.markdown("Prototype — AI-enhanced transcription, translation & audio playback. For demo only.")

# =========================
# SETUP
# =========================
AudioSegment.converter = ffmpeg_static.get_ffmpeg_binary()
recognizer = sr.Recognizer()

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    st.warning("⚠️ No OpenAI key found. Falling back to Deep Translator for translations.")

# =========================
# HELPERS
# =========================
def tts_to_bytes(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()

def transcribe_audio(wav_bytes, input_lang="ur"):
    """Try OpenAI Whisper first, fallback to Google Web Speech."""
    text = ""
    if OPENAI_API_KEY:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmpfile.write(wav_bytes)
                tmpfile.flush()
                audio_file = open(tmpfile.name, "rb")
                result = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                text = result.text
        except Exception as e:
            st.error(f"OpenAI STT failed: {e}")
    if not text:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmpfile.write(wav_bytes)
                tmpfile.flush()
                with sr.AudioFile(tmpfile.name) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio, language=input_lang)
        except Exception as e:
            st.error(f"Google STT failed: {e}")
    return text

def translate_text(text, src="ur", dest="en"):
    """Try OpenAI GPT first, fallback to Deep Translator."""
    if not text:
        return ""
    if OPENAI_API_KEY:
        try:
            prompt = f"Translate the following {src} text into {dest}:\n\n{text}"
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a medical translator."},
                          {"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"OpenAI translation failed: {e}")
    try:
        return GoogleTranslator(source=src, target=dest).translate(text)
    except Exception as e:
        st.error(f"Deep Translator failed: {e}")
        return ""

# =========================
# APP UI
# =========================
st.sidebar.header("Settings & Privacy")
input_lang = st.sidebar.selectbox("Input language (patient)", ["Urdu", "English"])
clinician_lang = st.sidebar.selectbox("Clinician language", ["English", "Urdu"])

input_lang_code = "ur" if input_lang == "Urdu" else "en"
output_lang_code = "en" if clinician_lang == "English" else "ur"

st.header("Live Translation Console")
st.write("🎙 Record short phrases; each recording becomes a conversation turn.")

# Patient → Clinician
st.subheader("🧑 Patient → Clinician")

patient_audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop", key="patient")
if patient_audio:
    st.write("Recorded — processing...")
    wav_bytes = patient_audio["bytes"]

    detected_text = transcribe_audio(wav_bytes, input_lang_code)
    st.write("Detected (raw):", detected_text)

    translated_text = translate_text(detected_text, src=input_lang_code, dest=output_lang_code)
    st.write("Clinician view (translated / cleaned):", translated_text)

    if translated_text:
        tts_bytes = tts_to_bytes(translated_text, lang=output_lang_code)
        st.audio(tts_bytes, format="audio/mp3")

# Doctor → Patient
st.subheader("👩‍⚕️ Clinician → Patient")
doctor_input = st.text_input("Type clinician message:")

if doctor_input:
    st.write("Original:", doctor_input)
    translated_text = translate_text(doctor_input, src=output_lang_code, dest=input_lang_code)
    st.write("Translated:", translated_text)

    if translated_text:
        tts_bytes = tts_to_bytes(translated_text, lang=input_lang_code)
        st.audio(tts_bytes, format="audio/mp3")
