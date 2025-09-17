# app.py
"""
Healthcare Translation Assistant - Prototype for pre-interview assignment
Features:
- Browser mic recorder (streamlit_mic_recorder)
- OpenAI (v1) AI-enhanced transcription & translation (optional, if API key present)
- Fallback: Google Web Speech (speech_recognition) + googletrans
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
import shutil, subprocess

# -------------------------
# Third-party imports (optional)
# -------------------------
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

try:
    from googletrans import Translator as GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -------------------------
# FFmpeg check (required for pydub)
# -------------------------
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
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Healthcare Translation Assistant", layout="centered")
st.title("🩺 Healthcare Translation Assistant (Patient ↔ Doctor)")
st.markdown("Prototype — AI-enhanced transcription, translation & audio playback. For demo only.")

# -------------------------
# OpenAI client
# -------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.warning(f"OpenAI client init failed: {e}")
        openai_client = None

# -------------------------
# Google translator fallback
# -------------------------
google_translator = GoogleTranslator() if GoogleTranslator else None

# -------------------------
# Check mic recorder
# -------------------------
if mic_recorder is None:
    st.warning("`streamlit_mic_recorder` not found. Install it (pip install streamlit-mic-recorder) to use browser recording.")
    st.stop()
if sr is None:
    st.info("`speech_recognition` not installed — Google STT fallback not available.")
if gTTS is None:
    st.warning("gTTS not installed. Install gTTS to enable TTS audio playback.")

# -------------------------
# Utility functions
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
    if record_value is None:
        return None
    if isinstance(record_value, dict):
        for k in ("wav", "data", "bytes", "wavBytes"):
            if k in record_value and record_value[k]:
                val = record_value[k]
                if isinstance(val, str) and val.startswith("data:"):
                    try:
                        return base64.b64decode(val.split(",", 1)[1])
                    except Exception:
                        pass
                if isinstance(val, str):
                    try:
                        return base64.b64decode(val)
                    except Exception:
                        pass
                if isinstance(val, (bytes, bytearray)):
                    return bytes(val)
    if isinstance(record_value, (bytes, bytearray)):
        return bytes(record_value)
    return None

def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# -------------------------
# Transcription functions
# -------------------------
def transcribe_with_openai(file_path: str, language: str = None) -> str:
    if openai_client is None:
        raise RuntimeError("OpenAI client not configured")
    try:
        resp = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=open(file_path, "rb"),
            language=language or None
        )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        raise

def transcribe_with_google_speech(file_path: str, language="en-US") -> str:
    if sr is None:
        raise RuntimeError("speech_recognition not installed")
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        raise

# -------------------------
# Translation / AI cleanup
# -------------------------
def ai_postprocess_and_translate(raw_text: str, source_lang_name: str, target_lang_name: str = "English"):
    if not raw_text:
        return "", ""
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
            {"role": "user",
             "content": (
                 f"Source language name: {source_lang_name}\n"
                 f"Target language name: {target_lang_name}\n\n"
                 f"Raw transcript: {raw_text}\n\n"
                 "Please return a JSON-like reply with two fields:\n"
                 "CLEANED: <cleaned transcript in original language>\n"
                 "TRANSLATION: <concise clinician-facing English translation (1-2 sentences)>"
             )}
        ]
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.0
            )
            out = resp.choices[0].message.content.strip()
            cleaned, translation = "", ""
            if "CLEANED:" in out and "TRANSLATION:" in out:
                try:
                    cleaned = out.split("CLEANED:")[1].split("TRANSLATION:")[0].strip()
                    translation = out.split("TRANSLATION:")[1].strip()
                except Exception:
                    translation = out
            else:
                translation = out
            return cleaned or raw_text, translation or raw_text
        except Exception as e:
            st.warning(f"OpenAI postprocessing failed: {e}")
            return raw_text, ""
    else:
        cleaned = raw_text.replace(" um ", " ").replace(" uh ", " ").strip()
        return cleaned, ""

def translate_with_google(text: str, src="auto", dest="en") -> str:
    if not google_translator:
        return ""
    try:
        return google_translator.translate(text, src=src, dest=dest).text
    except Exception:
        return text

# -------------------------
# TTS helpers
# -------------------------
def save_tts(text: str, lang_code: str = "en") -> str:
    if gTTS is None or not text.strip():
        return None
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(tmp_file.name)
        return tmp_file.name
    except Exception as e:
        st.warning(f"TTS generation failed: {e}")
        return None

def safe_tts(text, lang):
    for _ in range(3):
        try:
            return save_tts(text, lang_code=lang)
        except Exception:
            time.sleep(1)
    st.warning("TTS failed after 3 attempts.")
    return None

# -------------------------
# Session state
# -------------------------
if "conv" not in st.session_state:
    st.session_state.conv = []

# -------------------------
# Language codes
# -------------------------
LANG_CODES = {"Urdu": "ur", "English": "en", "Hindi": "hi", "Arabic": "ar"}
STT_LANG_CODES = {"Urdu": "ur-PK", "English": "en-US", "Hindi": "hi-IN", "Arabic": "ar-SA"}

# -------------------------
# Sidebar: Settings & Privacy
# -------------------------
st.sidebar.header("Settings & Privacy")
backend_choice = st.sidebar.selectbox("Backend (transcription/translation)", ["OpenAI (if available)", "Google (cloud)", "Vosk (offline, local)"])
store_locally = st.sidebar.checkbox("Store transcripts locally (for debugging)", value=False)
agree_privacy = st.sidebar.checkbox("I confirm this audio does NOT contain real patient data", value=True)
if not agree_privacy:
    st.stop()

# -------------------------
# Top-level UI: Language selectors
# -------------------------
col_a, col_b = st.columns(2)
with col_a:
    input_lang_name = st.selectbox("Input language (patient)", ["Urdu", "English", "Hindi", "Arabic"], index=0)
with col_b:
    target_lang_name = st.selectbox("Clinician language", ["English", "Urdu", "Hindi", "Arabic"], index=0)

st.markdown("## Live Translation Console")
st.caption("Record short phrases; each recording becomes a conversation turn. Use the Speak buttons to replay translated audio.")

# -------------------------
# Patient → Clinician
# -------------------------
left, right = st.columns([1,1])
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
            # Transcription
            if backend_choice.startswith("OpenAI") and openai_client:
                try:
                    st.info("Transcribing (OpenAI Whisper)...")
                    raw_transcript = transcribe_with_openai(tmp_wav, language=LANG_CODES.get(input_lang_name))
                except Exception as e:
                    st.warning(f"OpenAI transcription failed: {e}")
            if not raw_transcript:
                if sr:
                    try:
                        st.info("Transcribing (Google Web Speech)...")
                        raw_transcript = transcribe_with_google_speech(tmp_wav, language=STT_LANG_CODES.get(input_lang_name))
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                else:
                    st.error("No transcription backend available.")
            # Local store
            if store_locally:
                local_dir = Path("transcripts")
                local_dir.mkdir(exist_ok=True)
                Path(local_dir / f"{int(time.time())}_raw.txt").write_text(raw_transcript or "")
            # AI postprocess
            cleaned, ai_translation = ai_postprocess_and_translate(raw_transcript or "", input_lang_name, target_lang_name)
            if not ai_translation and google_translator:
                ai_translation = translate_with_google(cleaned or raw_transcript or "", src=LANG_CODES.get(input_lang_name), dest=LANG_CODES.get(target_lang_name, "en"))
            # TTS
            tts_path = None
            if gTTS and ai_translation:
                tts_path = safe_tts(ai_translation, LANG_CODES.get(target_lang_name, "en"))
            st.session_state.conv.append({
                "role": "Patient",
                "original": raw_transcript or "",
                "cleaned": cleaned or "",
                "translated": ai_translation or "",
                "tts": tts_path,
                "ts": time.time()
            })
            st.markdown("**Detected (raw):**")
            st.info(raw_transcript or "—")
            st.markdown("**Clinician view (translated / cleaned):**")
            st.success(ai_translation or "—")
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

# -------------------------
# Clinician → Patient
# -------------------------
with right:
    st.subheader("👩‍⚕️ Clinician → Patient")
    st.write(f"Output language for patient: **{target_lang_name}**")
    doctor_mode = st.radio("Doctor input mode", ["Type English text", "🎙 Record speech"], index=0)
    doctor_input_text = ""
    if doctor_mode == "Type English text":
        doctor_input_text = st.text_area("Type clinician message:", height=120)
        if st.button("➡ Translate & Add to Conversation"):
            if doctor_input_text.strip():
                translated = ""
                if openai_client and backend_choice.startswith("OpenAI"):
                    try:
                        msgs = [
                            {"role": "system", "content": "You are a clinical translator."},
                            {"role": "user", "content": f"Translate this to {target_lang_name}: {doctor_input_text}"}
                        ]
                        resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=msgs, max_tokens=200, temperature=0.0)
                        translated = resp.choices[0].message.content.strip()
                    except Exception as e:
                        st.warning(f"OpenAI translation failed: {e}")
                if not translated and google_translator:
                    translated = translate_with_google(doctor_input_text, src="en", dest=LANG_CODES.get(target_lang_name, "ur"))
                tts_path = safe_tts(translated, LANG_CODES.get(target_lang_name, "ur"))
                st.session_state.conv.append({
                    "role": "Doctor",
                    "original": doctor_input_text,
                    "cleaned": doctor_input_text,
                    "translated": translated,
                    "tts": tts_path,
                    "ts": time.time()
                })
                st.success("Added to conversation.")
    # -------------------------
# Clinician → Patient (speech recording)
# -------------------------
    else:  # Doctor record speech
        doc_record = mic_recorder(key="doctor_rec")
        if doc_record:
            st.success("Recorded — processing...")
            wav_bytes = extract_wav_bytes(doc_record)
            if wav_bytes:
                tmp_wav = save_bytes_to_file(wav_bytes, suffix=".wav")
                raw_transcript = ""
                # Transcription
                if backend_choice.startswith("OpenAI") and openai_client:
                    try:
                        raw_transcript = transcribe_with_openai(tmp_wav, language=STT_LANG_CODES.get("English", "en-US"))
                    except Exception as e:
                        st.warning(f"OpenAI transcription failed: {e}")
                if not raw_transcript:
                    if sr:
                        try:
                            raw_transcript = transcribe_with_google_speech(tmp_wav, language=STT_LANG_CODES.get("English", "en-US"))
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")
                    else:
                        st.error("No STT backend available.")
                # Translation to patient language
                translated = ""
                if openai_client and backend_choice.startswith("OpenAI"):
                    try:
                        msgs = [
                            {"role": "system", "content": "You are a clinical translator."},
                            {"role": "user", "content": f"Translate this to {target_lang_name}: {raw_transcript}"}
                        ]
                        resp = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=msgs,
                            max_tokens=200,
                            temperature=0.0
                        )
                        translated = resp.choices[0].message.content.strip()
                    except Exception as e:
                        st.warning(f"OpenAI translation failed: {e}")
                if not translated and google_translator:
                    translated = translate_with_google(raw_transcript, src="en", dest=LANG_CODES.get(target_lang_name, "ur"))
                # TTS
                tts_path = safe_tts(translated, LANG_CODES.get(target_lang_name, "ur"))
                # Add to conversation state
                st.session_state.conv.append({
                    "role": "Doctor",
                    "original": raw_transcript,
                    "cleaned": raw_transcript,
                    "translated": translated,
                    "tts": tts_path,
                    "ts": time.time()
                })
                # Display immediate results
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
# Conversation history panel
# -------------------------
st.markdown("---")
st.header("Conversation History")
if not st.session_state.conv:
    st.info("No conversation turns yet. Record a short phrase to start.")
else:
    with st.expander("View full conversation"):
        for idx, msg in enumerate(reversed(st.session_state.conv)):
            role = msg.get("role", "User")
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.get("ts", time.time())))
            st.markdown(f"**{role}** — _{ts}_")
            st.write("Original:", msg.get("original", ""))
            st.write("Cleaned:", msg.get("cleaned", ""))
            st.write("Translated:", msg.get("translated", ""))

            # TTS playback
            tts_path = msg.get("tts")
            if tts_path and Path(tts_path).exists():
                play_key = f"play_{idx}_{int(msg.get('ts',0))}"
                if st.button("🔊 Speak", key=play_key):
                    audio_bytes = read_file_bytes(tts_path)
                    st.audio(audio_bytes, format="audio/mp3")
                # Download link
                try:
                    b = read_file_bytes(tts_path)
                    b64 = base64.b64encode(b).decode()
                    href = f'<a download="tts_{idx}.mp3" href="data:audio/mp3;base64,{b64}">Download audio</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception:
                    pass
            st.markdown("---")

# -------------------------
# Footer tips
# -------------------------
st.caption(
    "Tip: For best results provide an OpenAI API key in Streamlit secrets or set OPENAI_API_KEY in your environment. "
    "This enables AI-enhanced transcription and translation (Whisper + GPT). "
    "If you don't have it, the app will use Google Web Speech + googletrans as fallback."
        )
