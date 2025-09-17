# app.py
"""
Healthcare Translation Assistant - Streamlit prototype (fixed for Streamlit Cloud)
Features:
- Browser mic recorder (streamlit_mic_recorder)
- OpenAI (optional, if OPENAI_API_KEY provided in Streamlit Secrets)
- Fallback: speech_recognition (Google Web Speech) + googletrans
- In-memory TTS playback (gTTS)
- pydub + ffmpeg-static for audio conversion (works on Streamlit Cloud)
"""

import os
import time
import base64
import glob
import json
import re
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st

# ----- audio libs -----
try:
    import pydub
    from pydub import AudioSegment
except Exception:
    pydub = None
    AudioSegment = None

# prefer ffmpeg-static so ffmpeg binary is available on Cloud
try:
    import ffmpeg_static
    pydub.AudioSegment.converter = ffmpeg_static.get_ffmpeg_binary()
except Exception:
    # fallback: hope ffmpeg available in PATH (may fail on some cloud setups)
    pass

# ----- optional third-party libs -----
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

# OpenAI client import will be attempted only if key present
openai_client = None
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.warning(f"OpenAI client init failed: {e}")
        openai_client = None

# ----- helper utilities -----
def cleanup_temp_files(tmp_dir=None):
    """Remove old temp pcm/mp3 files created by the app."""
    td = tmp_dir or tempfile.gettempdir()
    for pattern in ("*_pcm.wav", "*.mp3"):
        for f in glob.glob(os.path.join(td, pattern)):
            try:
                os.remove(f)
            except Exception:
                pass

cleanup_temp_files()

def parse_gpt_output(output_text: str):
    """Try to parse a JSON-like response from GPT, fallback to text-splitting."""
    cleaned, translation = "", ""
    try:
        json_like = re.sub(r"(\w+):", r'"\1":', output_text)
        json_like = json_like.replace("'", '"')
        data = json.loads(json_like)
        cleaned = data.get("CLEANED", "").strip()
        translation = data.get("TRANSLATION", "").strip()
    except Exception:
        if "CLEANED:" in output_text and "TRANSLATION:" in output_text:
            try:
                cleaned = output_text.split("CLEANED:")[1].split("TRANSLATION:")[0].strip()
                translation = output_text.split("TRANSLATION:")[1].strip()
            except Exception:
                translation = output_text.strip()
        else:
            translation = output_text.strip()
    return cleaned or "", translation or ""

def save_bytes_to_temp_wav(b: bytes) -> str:
    """Save bytes to a temp file and return the path (original container format)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(b)
    tmp.flush()
    tmp.close()
    return tmp.name

def convert_to_pcm_wav(in_bytes: bytes) -> str:
    """
    Convert arbitrary audio bytes to PCM 16-bit WAV file path using pydub.
    Returns path to temporary PCM WAV file.
    """
    if AudioSegment is None:
        raise RuntimeError("pydub not available for audio conversion")
    # load in-memory
    bio = BytesIO(in_bytes)
    try:
        seg = AudioSegment.from_file(bio)
    except Exception as e:
        # try treating as raw wav
        raise RuntimeError(f"pydub failed to read audio bytes: {e}")
    pcm_tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_pcm.wav")
    pcm_tmp.close()
    seg.export(pcm_tmp.name, format="wav", codec="pcm_s16le")
    return pcm_tmp.name

def tts_to_bytes(text: str, lang_code: str = "en") -> BytesIO:
    """Return a BytesIO mp3 buffer for text using gTTS (in-memory)."""
    if gTTS is None:
        raise RuntimeError("gTTS not installed")
    out = BytesIO()
    gTTS(text=text, lang=lang_code).write_to_fp(out)
    out.seek(0)
    return out

def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# ----- STT + translation functions -----
def transcribe_with_openai(file_path: str, language: str = None) -> str:
    """Transcribe audio file using OpenAI Whisper (via OpenAI client)."""
    if openai_client is None:
        raise RuntimeError("OpenAI client not configured")
    try:
        # openai_client expects a file-like object for transcription
        resp = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=open(file_path, "rb"),
            language=language or None
        )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        raise

def transcribe_with_google_speech(file_path: str, language="en-US") -> str:
    """Use speech_recognition with Google Web Speech on a local PCM WAV file."""
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

def translate_with_google(text: str, src="auto", dest="en") -> str:
    if GoogleTranslator is None:
        return ""
    try:
        translator = GoogleTranslator()
        return translator.translate(text, src=src, dest=dest).text
    except Exception:
        return ""

def ai_postprocess_and_translate(raw_text: str, source_lang_name: str, target_lang_name: str = "English"):
    """Optional: use OpenAI chat to clean and translate the raw transcript (if OpenAI available)."""
    if not raw_text:
        return "", ""
    if openai_client:
        system_prompt = (
            "You are a clinical translation assistant. Given a raw speech transcript that may contain recognition errors, "
            "1) correct mistakes while preserving medical terms, "
            "2) produce a concise clinician-ready English version (1-2 sentences), "
            "3) return two fields: CLEANED and TRANSLATION."
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
            cleaned, translation = parse_gpt_output(out)
            return cleaned or raw_text, translation or raw_text
        except Exception:
            return raw_text, ""
    else:
        # simple cleanup fallback
        cleaned = raw_text.replace(" um ", " ").replace(" uh ", " ").strip()
        return cleaned, ""

# ----- Streamlit UI and flow -----
st.set_page_config(page_title="Healthcare Translation Assistant", layout="centered")
st.title("🩺 Healthcare Translation Assistant (Patient ↔ Clinician)")
st.markdown("Prototype — AI-enhanced transcription, translation & audio playback. For demo only.")

# Sidebar
st.sidebar.header("Settings & Privacy")
backend_choice = st.sidebar.selectbox("Backend (transcription/translation)",
                                      ["OpenAI (if available)", "Google (cloud)", "Vosk (offline, local)"])
store_locally = st.sidebar.checkbox("Store transcripts locally (for debugging)", value=False)
st.sidebar.markdown("Privacy note: This prototype may send audio to third-party services for transcription. Do not use real patient data without agreements.")

# Language selectors
col_a, col_b = st.columns(2)
with col_a:
    input_lang_name = st.selectbox("Input language (patient)", ["Urdu", "English", "Hindi", "Arabic"], index=0)
with col_b:
    target_lang_name = st.selectbox("Clinician language", ["English", "Urdu", "Hindi", "Arabic"], index=0)

LANG_CODES = {"Urdu": "ur", "English": "en", "Hindi": "hi", "Arabic": "ar"}
STT_LANG_CODES = {"Urdu": "ur-PK", "English": "en-US", "Hindi": "hi-IN", "Arabic": "ar-SA"}

st.markdown("## Live Translation Console")
st.caption("Record short phrases; each recording becomes a conversation turn. Use the Speak buttons to replay translated audio.")

if "conv" not in st.session_state:
    st.session_state.conv = []

# Patient -> Clinician
left, right = st.columns([1, 1])
with left:
    st.subheader("🧑 Patient → Clinician")
    st.write(f"Input language: **{input_lang_name}**")
    patient_record = mic_recorder(key="patient_rec")
    if patient_record:
        st.success("Recorded — processing...")
        wav_bytes = None
        try:
            wav_bytes = None
            # attempt to extract bytes (component returns different shapes)
            if isinstance(patient_record, dict):
                for k in ("wav", "data", "bytes", "wavBytes"):
                    if k in patient_record and patient_record[k]:
                        val = patient_record[k]
                        if isinstance(val, str) and val.startswith("data:"):
                            wav_bytes = base64.b64decode(val.split(",", 1)[1])
                            break
                        if isinstance(val, str):
                            try:
                                wav_bytes = base64.b64decode(val)
                                break
                            except Exception:
                                pass
                        if isinstance(val, (bytes, bytearray)):
                            wav_bytes = bytes(val)
                            break
            elif isinstance(patient_record, (bytes, bytearray)):
                wav_bytes = bytes(patient_record)
        except Exception as e:
            st.warning(f"Could not parse recorder output: {e}")

        st.write("DEBUG: wav_bytes length =", len(wav_bytes) if wav_bytes else "None")
        if not wav_bytes:
            st.warning("No audio bytes found in recorder output.")
        else:
            # Convert to PCM WAV (temp file) for STT backends that expect WAV path
            try:
                pcm_path = convert_to_pcm_wav(wav_bytes)
            except Exception as e:
                st.warning(f"Audio conversion failed: {e}")
                pcm_path = save_bytes_to_temp_wav(wav_bytes)

            raw_transcript = ""
            # Try OpenAI Whisper if chosen and client available
            if backend_choice.startswith("OpenAI") and openai_client:
                try:
                    st.info("Transcribing (OpenAI Whisper)...")
                    raw_transcript = transcribe_with_openai(pcm_path, language=LANG_CODES.get(input_lang_name))
                except Exception as e:
                    st.warning(f"OpenAI transcription failed: {e}")
                    raw_transcript = ""

            # Fallback: Google Web Speech via speech_recognition
            if not raw_transcript and sr is not None:
                try:
                    st.info("Transcribing (Google Web Speech)...")
                    raw_transcript = transcribe_with_google_speech(pcm_path, language=STT_LANG_CODES.get(input_lang_name, "en-US"))
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    raw_transcript = ""

            if store_locally:
                try:
                    local_dir = Path("transcripts")
                    local_dir.mkdir(exist_ok=True)
                    idx = int(time.time())
                    Path(local_dir / f"{idx}_raw.txt").write_text(raw_transcript or "")
                except Exception:
                    pass

            # AI postprocess (clean + translate)
            cleaned, ai_translation = ai_postprocess_and_translate(raw_transcript or "", source_lang_name=input_lang_name, target_lang_name=target_lang_name)

            # if AI didn't produce a translation, use googletrans fallback (translate to clinician language)
            if not ai_translation:
                if GoogleTranslator is not None:
                    try:
                        ai_translation = translate_with_google(cleaned or raw_transcript or "", src=LANG_CODES.get(input_lang_name, "auto"), dest=LANG_CODES.get(target_lang_name, "en"))
                    except Exception:
                        ai_translation = cleaned or raw_transcript or ""
                else:
                    ai_translation = cleaned or raw_transcript or ""

            # TTS playback in-memory
            if gTTS is not None and ai_translation:
                try:
                    tts_bytesio = tts_to_bytes(ai_translation, lang_code=LANG_CODES.get(target_lang_name, "en"))
                    st.audio(tts_bytesio, format="audio/mp3")
                except Exception as e:
                    st.warning(f"TTS failed: {e}")

            # Append to conversation
            st.session_state.conv.append({
                "role": "Patient",
                "original": raw_transcript or "",
                "cleaned": cleaned or "",
                "translated": ai_translation or "",
                "tts": None,
                "ts": time.time()
            })

            st.markdown("**Detected (raw):**")
            st.info(raw_transcript or "—")
            st.markdown("**Clinician view (translated / cleaned):**")
            st.success(ai_translation or "—")

            # cleanup pcm temp
            try:
                if Path(pcm_path).exists():
                    os.remove(pcm_path)
            except Exception:
                pass

# Clinician -> Patient
with right:
    st.subheader("👩‍⚕️ Clinician → Patient")
    st.write(f"Output language for patient: **{target_lang_name}**")
    doctor_mode = st.radio("Doctor input mode", ["Type English text", "🎙 Record speech"], index=0)

    if doctor_mode == "Type English text":
        doctor_input_text = st.text_area("Type clinician message:", height=120)
        if st.button("➡ Translate & Add to Conversation"):
            if not doctor_input_text.strip():
                st.warning("Please type a message first.")
            else:
                translated = ""
                # prefer OpenAI for careful medical phrasing
                if openai_client and backend_choice.startswith("OpenAI"):
                    try:
                        msgs = [
                            {"role": "system", "content": "You are a clinical translator. Translate the message preserving meaning and medical terms."},
                            {"role": "user", "content": f"Translate this to {target_lang_name}: {doctor_input_text}"}
                        ]
                        resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=msgs, max_tokens=200, temperature=0.0)
                        translated = resp.choices[0].message.content.strip()
                    except Exception:
                        translated = ""

                if not translated:
                    if GoogleTranslator is not None:
                        try:
                            translated = translate_with_google(doctor_input_text, src="en", dest=LANG_CODES.get(target_lang_name, "ur"))
                        except Exception:
                            translated = doctor_input_text
                    else:
                        translated = doctor_input_text

                # TTS playback (in-memory)
                if gTTS is not None:
                    try:
                        tts_bytesio = tts_to_bytes(translated, lang_code=LANG_CODES.get(target_lang_name, "en"))
                        st.audio(tts_bytesio, format="audio/mp3")
                    except Exception:
                        pass

                st.session_state.conv.append({
                    "role": "Doctor",
                    "original": doctor_input_text,
                    "cleaned": doctor_input_text,
                    "translated": translated,
                    "tts": None,
                    "ts": time.time()
                })
                st.success("Added to conversation.")

    else:
        # Doctor record speech
        doc_record = mic_recorder(key="doctor_rec")
        if doc_record:
            st.success("Recorded — processing...")
            # extract bytes same as above (re-use logic)
            wav_bytes = None
            try:
                if isinstance(doc_record, dict):
                    for k in ("wav", "data", "bytes", "wavBytes"):
                        if k in doc_record and doc_record[k]:
                            val = doc_record[k]
                            if isinstance(val, str) and val.startswith("data:"):
                                wav_bytes = base64.b64decode(val.split(",", 1)[1])
                                break
                            if isinstance(val, str):
                                try:
                                    wav_bytes = base64.b64decode(val)
                                    break
                                except Exception:
                                    pass
                            if isinstance(val, (bytes, bytearray)):
                                wav_bytes = bytes(val)
                                break
                elif isinstance(doc_record, (bytes, bytearray)):
                    wav_bytes = bytes(doc_record)
            except Exception as e:
                st.warning(f"Could not parse recorder output: {e}")

            if not wav_bytes:
                st.warning("No audio bytes found in recorder output.")
            else:
                try:
                    pcm_path = convert_to_pcm_wav(wav_bytes)
                except Exception:
                    pcm_path = save_bytes_to_temp_wav(wav_bytes)

                raw_transcript = ""
                if backend_choice.startswith("OpenAI") and openai_client:
                    try:
                        raw_transcript = transcribe_with_openai(pcm_path, language=STT_LANG_CODES.get("English", "en-US"))
                    except Exception:
                        raw_transcript = ""

                if not raw_transcript and sr is not None:
                    try:
                        raw_transcript = transcribe_with_google_speech(pcm_path, language=STT_LANG_CODES.get("English", "en-US"))
                    except Exception:
                        raw_transcript = ""

                # Translate to patient language
                translated = ""
                if openai_client and backend_choice.startswith("OpenAI"):
                    try:
                        msgs = [
                            {"role": "system", "content": "You are a clinical translator."},
                            {"role": "user", "content": f"Translate this to {target_lang_name}: {raw_transcript}"}
                        ]
                        resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=msgs, max_tokens=200, temperature=0.0)
                        translated = resp.choices[0].message.content.strip()
                    except Exception:
                        translated = ""

                if not translated:
                    if GoogleTranslator is not None:
                        try:
                            translated = translate_with_google(raw_transcript, src="en", dest=LANG_CODES.get(target_lang_name, "ur"))
                        except Exception:
                            translated = raw_transcript
                    else:
                        translated = raw_transcript

                # TTS playback
                if gTTS is not None and translated:
                    try:
                        tts_bytesio = tts_to_bytes(translated, lang_code=LANG_CODES.get(target_lang_name, "en"))
                        st.audio(tts_bytesio, format="audio/mp3")
                    except Exception:
                        pass

                st.session_state.conv.append({
                    "role": "Doctor",
                    "original": raw_transcript,
                    "cleaned": raw_transcript,
                    "translated": translated,
                    "tts": None,
                    "ts": time.time()
                })

                # cleanup pcm temp
                try:
                    if Path(pcm_path).exists():
                        os.remove(pcm_path)
                except Exception:
                    pass

# Conversation history
st.markdown("---")
st.header("Conversation History")
if not st.session_state.conv:
    st.info("No conversation turns yet. Record a short phrase to start.")
else:
    for idx, msg in enumerate(reversed(st.session_state.conv)):
        role = msg.get("role", "User")
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.get("ts", time.time())))
        st.markdown(f"**{role}** — _{ts}_")
        st.write("Original:", msg.get("original", ""))
        st.write("Cleaned:", msg.get("cleaned", ""))
        st.write("Translated:", msg.get("translated", ""))
        st.markdown("---")

st.caption("Tip: Add OPENAI_API_KEY in Streamlit Secrets for Whisper/GPT. On Cloud, ensure ffmpeg-static and pydub are in requirements.")
