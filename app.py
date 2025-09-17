# app.py
"""
Healthcare Translation Assistant - Prototype for pre-interview assignment
Features:
- Browser mic recorder (streamlit_mic_recorder)
- OpenAI (v1) AI-enhanced transcription & translation (optional, if API key present)
- Fallback: Google Web Speech (speech_recognition) for STT + googletrans for translation
- Conversation history + Speak buttons (gTTS)
- Language selectors, privacy toggle, mobile-friendly layout
"""

import streamlit as st
import tempfile
import os
import io
import base64
import time
from pathlib import Path
from pydub import AudioSegment


# Third-party imports that might be optional
try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

# googletrans fallback translator
try:
    from googletrans import Translator as GoogleTranslator
except Exception:
    GoogleTranslator = None

# OpenAI v1 client (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
import shutil, subprocess, streamlit as st

def check_ffmpeg():
    ff = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ff:
        try:
            out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
            return True, out.stdout.splitlines()[0]
        except Exception as e:
            return True, f"ffmpeg present but error running it: {e}"
    else:
        return False, "ffmpeg not found"

ff_ok, ff_status = check_ffmpeg()
st.sidebar.markdown("**ffmpeg status:** " + ("✅ " + ff_status if ff_ok else "❌ " + ff_status))


# -------------------------
# Configuration & secrets
# -------------------------
st.set_page_config(page_title="Healthcare Translation Assistant", layout="centered")
st.title("🩺 Healthcare Translation Assistant (Patient ↔ Doctor)")
st.markdown("Prototype — AI-enhanced transcription, translation & audio playback. For demo only.")

# Get OpenAI API key (optional)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.warning(f"OpenAI client init failed: {e}")
        openai_client = None

# Translator fallback
google_translator = None
if GoogleTranslator is not None:
    google_translator = GoogleTranslator()

# Microphone recorder availability check
if mic_recorder is None:
    st.warning("`streamlit_mic_recorder` not found. Install it (pip install streamlit-mic-recorder) to use browser recording.")
    st.stop()

# SpeechRecognition availability note (fallback STT)
if sr is None:
    st.info("`speech_recognition` not installed — Google Web Speech fallback may not be available. Install if you want fallback STT.")

# gTTS availability
if gTTS is None:
    st.warning("gTTS not installed. Install gTTS to enable TTS audio playback.")

# -------------------------
# Utility helper functions
# -------------------------

def save_bytes_to_file(b: bytes, suffix: str = ".wav") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(b)
    tmp.flush()
    tmp.close()

    # Convert to PCM WAV if needed
    if suffix == ".wav":
        try:
            sound = AudioSegment.from_file(tmp.name)
            pcm_path = tmp.name.replace(".wav", "_pcm.wav")
            sound.export(pcm_path, format="wav", codec="pcm_s16le")
            return pcm_path
        except Exception as e:
            st.warning(f"Audio conversion failed: {e}")
            return tmp.name

    return tmp.name


def extract_wav_bytes(record_value):
    """Handle multiple return shapes from mic_recorder component."""
    if record_value is None:
        return None
    if isinstance(record_value, dict):
        # patterns we've seen
        for k in ("wav", "data", "bytes", "wavBytes"):
            if k in record_value and record_value[k]:
                val = record_value[k]
                # base64 data URI
                if isinstance(val, str) and val.startswith("data:"):
                    try:
                        return base64.b64decode(val.split(",", 1)[1])
                    except Exception:
                        pass
                # base64 raw string
                if isinstance(val, str):
                    try:
                        return base64.b64decode(val)
                    except Exception:
                        pass
                # raw bytes
                if isinstance(val, (bytes, bytearray)):
                    return bytes(val)
    # direct bytes
    if isinstance(record_value, (bytes, bytearray)):
        return bytes(record_value)
    return None

# -------------------------
# Transcription functions
# -------------------------
def transcribe_with_openai(file_path: str, language: str = None) -> str:
    """Use OpenAI v1 audio transcription (whisper-1) via OpenAI client."""
    if openai_client is None:
        raise RuntimeError("OpenAI client not configured")
    try:
        # note: openai_client.audio.transcriptions.create(...) is the v1 pattern
        resp = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=open(file_path, "rb"),
            language=language or None
        )
        # resp.text is expected
        return getattr(resp, "text", "") or ""
    except Exception as e:
        # propagate message for upper-level fallback
        raise

def transcribe_with_google_speech(file_path: str, language="en-US") -> str:
    """Fallback: use speech_recognition with Google Web Speech API on saved file."""
    if sr is None:
        raise RuntimeError("speech_recognition not installed")
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        raise

# -------------------------
# AI postprocessing & translation
# -------------------------
def ai_postprocess_and_translate(raw_text: str, source_lang_name: str, target_lang_name: str = "English"):
    """
    If OpenAI is available, ask the model to:
      - clean transcript, correct errors (esp medical terms)
      - produce clinician-friendly summary or translate
    Returns (cleaned_source_text, translated_text)
    """
    if not raw_text:
        return "", ""

    # If OpenAI available, use it for both cleaning and translation
    if openai_client:
        system_prompt = (
            "You are a clinical translation assistant. "
            "Given a raw speech transcript that may include recognition errors, "
            "1) correct mistakes while preserving medical terms, "
            "2) produce a concise clinician-ready English version (1-2 sentences), "
            "3) also provide a simple patient-language cleaned transcript if requested."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Source language name: {source_lang_name}\n"
                    f"Target language name: {target_lang_name}\n\n"
                    f"Raw transcript: {raw_text}\n\n"
                    "Please return a JSON-like reply with two fields:\n"
                    "CLEANED: <cleaned transcript in original language>\n"
                    "TRANSLATION: <concise clinician-facing English translation (1-2 sentences)>"
                )
            }
        ]
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.0
            )
            out = resp.choices[0].message.content.strip()
            # The model may return plain text; try to heuristically split CLEANED and TRANSLATION
            cleaned = ""
            translation = ""
            if "CLEANED:" in out and "TRANSLATION:" in out:
                try:
                    cleaned_part = out.split("CLEANED:")[1].split("TRANSLATION:")[0].strip()
                    trans_part = out.split("TRANSLATION:")[1].strip()
                    cleaned = cleaned_part
                    translation = trans_part
                except Exception:
                    translation = out
            else:
                # fallback: treat output as translation
                translation = out
            return cleaned or raw_text, translation or raw_text
        except Exception as e:
            st.warning(f"OpenAI postprocessing failed: {e}")
            # fallback to raw_text and empty translation (will use googletrans later)
            return raw_text, ""
    else:
        # No OpenAI: do a light cleanup and return raw_text
        cleaned = raw_text.replace(" um ", " ").replace(" uh ", " ").strip()
        return cleaned, ""

def translate_with_google(text: str, src="auto", dest="en") -> str:
    if not google_translator:
        return ""
    try:
        return google_translator.translate(text, src=src, dest=dest).text
    except Exception:
        return ""

# -------------------------
# TTS helper
# -------------------------
def tts_save_mp3(text: str, lang_code: str = "en") -> str:
    """Return path to generated mp3 file using gTTS."""
    if gTTS is None:
        raise RuntimeError("gTTS not installed")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    try:
        gTTS(text=text, lang=lang_code).save(tmp.name)
        return tmp.name
    except Exception as e:
        raise

def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# -------------------------
# Session state initialization
# -------------------------
if "conv" not in st.session_state:
    # conv = list of dicts: {role: "Patient"/"Doctor", original:str, cleaned:str, translated:str, tts_path:str, ts:float}
    st.session_state.conv = []

# -------------------------
# Top-level UI
# -------------------------
st.sidebar.header("Settings & Privacy")
st.sidebar.markdown("Choose backend for transcription/translation. For best AI-enhanced results, provide an OpenAI API key in Streamlit secrets or environment variable `OPENAI_API_KEY`.")
backend_choice = st.sidebar.selectbox("Backend (transcription/translation)", ["OpenAI (if available)", "Google (cloud)", "Vosk (offline, local)"])
store_locally = st.sidebar.checkbox("Store transcripts locally (for debugging)", value=False)
st.sidebar.markdown("Privacy note: This prototype may send audio to third-party services for transcription. Do not use real patient data without agreements.")

# Language selectors
col_a, col_b = st.columns(2)
with col_a:
    input_lang_name = st.selectbox("Input language (patient)", ["Urdu", "English", "Hindi", "Arabic"], index=0)
with col_b:
    target_lang_name = st.selectbox("Clinician language", ["English", "Urdu", "Hindi", "Arabic"], index=0)

# Map names to language codes (used by various services)
LANG_CODES = {"Urdu": "ur", "English": "en", "Hindi": "hi", "Arabic": "ar"}
# Google speech uses en-US, ur-PK etc — build reasonable defaults:
STT_LANG_CODES = {"Urdu": "ur-PK", "English": "en-US", "Hindi": "hi-IN", "Arabic": "ar-SA"}

st.markdown("## Live Translation Console")
st.caption("Record short phrases; each recording becomes a conversation turn. Use the Speak buttons to replay translated audio.")

# Two big buttons side-by-side for patient and doctor recording/entry
left, right = st.columns([1, 1])

with left:
    st.subheader("🧑 Patient → Clinician")
    st.write(f"Input language: **{input_lang_name}**")
    patient_record = mic_recorder(key="patient_rec")
    if patient_record:
        st.success("Recorded — processing...")
        wav_bytes = extract_wav_bytes(patient_record)
        if wav_bytes:
            tmp_wav = save_bytes_to_file(wav_bytes, suffix=".wav")
            raw_transcript = ""
            # Try OpenAI transcription if chosen and available
            if backend_choice.startswith("OpenAI") and openai_client:
                try:
                    st.info("Transcribing (OpenAI Whisper)...")
                    raw_transcript = transcribe_with_openai(tmp_wav, language=LANG_CODES.get(input_lang_name))
                except Exception as e:
                    st.warning(f"OpenAI transcription failed: {e}. Falling back to Google STT if available.")
                    raw_transcript = ""
            # Fallback: Google STT via speech_recognition
            if not raw_transcript:
                if sr is not None:
                    try:
                        st.info("Transcribing (Google Web Speech)...")
                        raw_transcript = transcribe_with_google_speech(tmp_wav, language=STT_LANG_CODES.get(input_lang_name, "en-US"))
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        raw_transcript = ""
                else:
                    st.error("No transcription backend available. Install speech_recognition or provide OpenAI key.")
                    raw_transcript = ""
            # Optional local save
            if store_locally:
                local_dir = Path("transcripts")
                local_dir.mkdir(exist_ok=True)
                idx = int(time.time())
                Path(local_dir / f"{idx}_raw.txt").write_text(raw_transcript or "")
            # AI postprocess (clean + translate) if possible
            cleaned, ai_translation = ai_postprocess_and_translate(raw_transcript or "", source_lang_name=input_lang_name, target_lang_name=target_lang_name)
            # If AI didn't produce a translation (no OpenAI), fallback to googletrans
            if not ai_translation:
                # translate raw/clean to target using googletrans if present
                if google_translator:
                    try:
                        ai_translation = translate_with_google(cleaned or raw_transcript or "", src=LANG_CODES.get(input_lang_name, "auto"), dest=LANG_CODES.get(target_lang_name, "en"))
                    except Exception:
                        ai_translation = ""
                else:
                    ai_translation = cleaned or raw_transcript or ""
            # TTS generation: use target lang code
            tts_path = None
            try:
                tts_lang = LANG_CODES.get(target_lang_name, "en")
                if gTTS is not None and ai_translation:
                    tts_path = tts_save_mp3(ai_translation, lang_code=tts_lang)
                else:
                    tts_path = None
            except Exception as e:
                st.warning(f"TTS failed: {e}")
                tts_path = None
            # Append to conversation history
            st.session_state.conv.append({
                "role": "Patient",
                "original": raw_transcript or "",
                "cleaned": cleaned or "",
                "translated": ai_translation or "",
                "tts": tts_path,
                "ts": time.time()
            })
            # show immediate results
            st.markdown("**Detected (raw):**")
            st.info(raw_transcript or "—")
            st.markdown("**Clinician view (translated / cleaned):**")
            st.success(ai_translation or "—")
            # cleanup temp wav
            try:
                os.remove(tmp_wav)
            except Exception:
                pass
        else:
            st.warning("Could not parse recorded audio; update streamlit_mic_recorder if problem persists.")

with right:
    st.subheader("👩‍⚕️ Clinician → Patient")
    st.write(f"Output language for patient: **{target_lang_name}**")
    # Doctor input mode: type or record
    doctor_mode = st.radio("Doctor input mode", ["Type English text", "🎙 Record speech"], index=0)
    doctor_input_text = ""
    if doctor_mode == "Type English text":
        doctor_input_text = st.text_area("Type clinician message:", height=120)
        if st.button("➡ Translate & Add to Conversation"):
            if not doctor_input_text.strip():
                st.warning("Please type a message first.")
            else:
                # translate clinician text to patient language
                translated = ""
                if openai_client and backend_choice.startswith("OpenAI"):
                    # ask OpenAI to produce patient-friendly translation
                    try:
                        msgs = [
                            {"role": "system", "content": "You are a clinical translator. Translate the message preserving meaning and medical terms."},
                            {"role": "user", "content": f"Translate this to {target_lang_name}: {doctor_input_text}"}
                        ]
                        resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=msgs, max_tokens=200, temperature=0.0)
                        translated = resp.choices[0].message.content.strip()
                    except Exception as e:
                        st.warning(f"OpenAI translation failed: {e}")
                        translated = ""
                if not translated:
                    if google_translator:
                        try:
                            translated = translate_with_google(doctor_input_text, src="en", dest=LANG_CODES.get(target_lang_name, "ur"))
                        except Exception:
                            translated = doctor_input_text
                    else:
                        translated = doctor_input_text
                # TTS
                tts_path = None
                try:
                    if gTTS is not None:
                        tts_path = tts_save_mp3(translated, lang_code=LANG_CODES.get(target_lang_name, "ur"))
                except Exception as e:
                    st.warning(f"TTS failed: {e}")
                    tts_path = None
                st.session_state.conv.append({
                    "role": "Doctor",
                    "original": doctor_input_text,
                    "cleaned": doctor_input_text,
                    "translated": translated,
                    "tts": tts_path,
                    "ts": time.time()
                })
                st.success("Added to conversation.")
    else:
        # record doctor speech
        doc_record = mic_recorder(key="doctor_rec")
        if doc_record:
            st.success("Recorded — processing...")
            wav_bytes = extract_wav_bytes(doc_record)
            if wav_bytes:
                tmp_wav = save_bytes_to_file(wav_bytes, suffix=".wav")
                # transcribe via OpenAI or Google
                raw_transcript = ""
                if backend_choice.startswith("OpenAI") and openai_client:
                    try:
                        raw_transcript = transcribe_with_openai(tmp_wav, language=STT_LANG_CODES.get("English", "en-US"))
                    except Exception as e:
                        st.warning(f"OpenAI transcription failed: {e}")
                        raw_transcript = ""
                if not raw_transcript:
                    if sr is not None:
                        try:
                            raw_transcript = transcribe_with_google_speech(tmp_wav, language=STT_LANG_CODES.get("English", "en-US"))
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")
                            raw_transcript = ""
                    else:
                        st.error("No STT backend available.")
                        raw_transcript = ""
                # translate to patient language
                translated = ""
                if openai_client and backend_choice.startswith("OpenAI"):
                    try:
                        msgs = [
                            {"role": "system", "content": "You are a clinical translator."},
                            {"role": "user", "content": f"Translate this to {target_lang_name}: {raw_transcript}"}
                        ]
                        resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=msgs, max_tokens=200, temperature=0.0)
                        translated = resp.choices[0].message.content.strip()
                    except Exception as e:
                        st.warning(f"OpenAI translate failed: {e}")
                        translated = ""
                if not translated:
                    if google_translator:
                        try:
                            translated = translate_with_google(raw_transcript, src="en", dest=LANG_CODES.get(target_lang_name, "ur"))
                        except Exception:
                            translated = raw_transcript
                    else:
                        translated = raw_transcript
                # TTS
                tts_path = None
                try:
                    if gTTS is not None and translated:
                        tts_path = tts_save_mp3(translated, lang_code=LANG_CODES.get(target_lang_name, "ur"))
                except Exception as e:
                    st.warning(f"TTS failed: {e}")
                st.session_state.conv.append({
                    "role": "Doctor",
                    "original": raw_transcript,
                    "cleaned": raw_transcript,
                    "translated": translated,
                    "tts": tts_path,
                    "ts": time.time()
                })
                st.markdown("**Detected Doctor (raw):**")
                st.info(raw_transcript or "")
                st.markdown("**Translated for patient:**")
                st.success(translated or "")
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass
            else:
                st.warning("Could not parse recorded audio; update mic recorder or check browser support.")

# -------------------------
# Conversation history panel (bottom, with Speak buttons)
# -------------------------
st.markdown("---")
st.header("Conversation History")
if not st.session_state.conv:
    st.info("No conversation turns yet. Record a short phrase to start.")
else:
    # show newest first
    for idx, msg in enumerate(reversed(st.session_state.conv)):
        role = msg.get("role", "User")
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.get("ts", time.time())))
        st.markdown(f"**{role}** — _{ts}_")
        st.write("Original:", msg.get("original", ""))
        st.write("Cleaned:", msg.get("cleaned", ""))
        st.write("Translated:", msg.get("translated", ""))
        tts_path = msg.get("tts")
        if tts_path and Path(tts_path).exists():
            # provide play button and download link
            play_key = f"play_{idx}_{int(msg.get('ts',0))}"
            if st.button("🔊 Speak", key=play_key):
                audio_bytes = read_file_bytes(tts_path)
                st.audio(audio_bytes, format="audio/mp3")
            # download
            try:
                b = read_file_bytes(tts_path)
                b64 = base64.b64encode(b).decode()
                href = f'<a download="tts_{idx}.mp3" href="data:audio/mp3;base64,{b64}">Download audio</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception:
                pass
        st.markdown("---")

# -------------------------
# Footer / tips
# -------------------------
st.caption("Tip: For best results provide an OpenAI API key in Streamlit secrets or set OPENAI_API_KEY in your environment. "
           "This enables AI-enhanced transcription and translation (Whisper + GPT). "
           "If you don't have it, the app will use Google Web Speech + googletrans as fallback.")
