# app_full_refactored.py
import streamlit as st
import tempfile, os, time, base64, shutil, subprocess
from pathlib import Path
from pydub import AudioSegment

# Optional third-party
try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None
try:
    import speech_recognition as sr
except ImportError:
    sr = None
try:
    from gtts import gTTS
except ImportError:
    gTTS = None
try:
    from googletrans import Translator as GoogleTranslator
except ImportError:
    GoogleTranslator = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -------------------- FFmpeg Check --------------------
def check_ffmpeg():
    ff = shutil.which("ffmpeg")
    if ff:
        try:
            out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
            return True, out.stdout.splitlines()[0]
        except Exception as e:
            return True, f"ffmpeg present but error: {e}"
    return False, "ffmpeg not found"

ff_ok, ff_status = check_ffmpeg()
st.sidebar.markdown("**ffmpeg status:** " + ("✅ " + ff_status if ff_ok else "❌ " + ff_status))

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="Healthcare Translation Assistant", layout="centered")
st.title("🩺 Healthcare Translation Assistant (Patient ↔ Clinician)")

# -------------------- OpenAI Client --------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY and OpenAI:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.warning(f"OpenAI init failed: {e}")

# -------------------- Google Translator --------------------
google_translator = GoogleTranslator() if GoogleTranslator else None

# -------------------- Utility Functions --------------------
def log_error(msg: str):
    """Log error in session state and display to user"""
    if "errors" not in st.session_state:
        st.session_state.errors = []
    st.session_state.errors.append(f"{time.strftime('%H:%M:%S')}: {msg}")
    st.error(msg)

def save_bytes_to_file(b: bytes, suffix=".wav") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(b)
    tmp.flush()
    tmp.close()
    if suffix == ".wav":
        try:
            sound = AudioSegment.from_file(tmp.name)
            pcm_path = tmp.name.replace(".wav", "_pcm.wav")
            sound.export(pcm_path, format="wav", codec="pcm_s16le")
            return pcm_path
        except Exception as e:
            log_error(f"Audio conversion failed: {e}")
            return tmp.name
    return tmp.name

def extract_wav_bytes(record_value):
    try:
        if record_value is None:
            return None
        if isinstance(record_value, dict):
            for k in ("wav", "data", "bytes", "wavBytes"):
                if k in record_value and record_value[k]:
                    val = record_value[k]
                    if isinstance(val, str) and val.startswith("data:"):
                        return base64.b64decode(val.split(",",1)[1])
                    if isinstance(val, str):
                        return base64.b64decode(val)
                    if isinstance(val, (bytes, bytearray)):
                        return bytes(val)
        if isinstance(record_value, (bytes, bytearray)):
            return bytes(record_value)
    except Exception as e:
        log_error(f"Failed to extract audio bytes: {e}")
    return None

def read_file_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        log_error(f"Failed to read file bytes: {e}")
        return b""

# -------------------- Transcription --------------------
def transcribe_openai(file_path: str, lang=None) -> str:
    if not openai_client:
        log_error("OpenAI client not configured")
        return ""
    try:
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=open(file_path,"rb"), language=lang)
        return getattr(resp, "text","") or ""
    except Exception as e:
        log_error(f"OpenAI transcription error: {e}")
        return ""

def transcribe_google(file_path: str, lang="en-US") -> str:
    if not sr:
        log_error("speech_recognition not installed")
        return ""
    try:
        r = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = r.record(source)
        return r.recognize_google(audio, language=lang)
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        log_error(f"Google STT error: {e}")
        return ""
# After transcription
if not raw_transcript:
    log_error(f"Transcription failed for {input_lang_name} input. Check API key, mic, or dependencies.")

# -------------------- Translation / AI --------------------
def ai_translate(raw_text, src_lang_name, target_lang_name="English"):
    if not raw_text:
        return "", ""
    if openai_client:
        system_prompt = (
            "You are a clinical translation assistant. Correct transcript errors while preserving medical terms, "
            "produce clinician-ready English summary, and cleaned patient version."
        )
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"Source:{src_lang_name}, Target:{target_lang_name}\nText:{raw_text}"}
        ]
        try:
            resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=300, temperature=0.0)
            out = resp.choices[0].message.content.strip()
            cleaned, translation = "", ""
            if "CLEANED:" in out and "TRANSLATION:" in out:
                cleaned = out.split("CLEANED:")[1].split("TRANSLATION:")[0].strip()
                translation = out.split("TRANSLATION:")[1].strip()
            else:
                translation = out
            return cleaned or raw_text, translation or raw_text
        except Exception as e:
            log_error(f"OpenAI translation failed: {e}")
            return raw_text, ""
    else:
        cleaned = raw_text.replace(" um "," ").replace(" uh "," ").strip()
        return cleaned, ""
# After translation
if not translated:
    log_error(f"Translation failed from {input_lang_name} to {target_lang_name}. Check OpenAI / Google API or internet connection.")

def google_translate(text, src="auto", dest="en"):
    if not google_translator:
        return ""
    try:
        return google_translator.translate(text, src=src, dest=dest).text
    except Exception as e:
        log_error(f"Google translation failed: {e}")
        return text

# -------------------- TTS --------------------
def generate_tts(text, lang="en"):
    if not gTTS or not text.strip():
        return None
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(tmp_file.name)
        return tmp_file.name
    except Exception as e:
        log_error(f"TTS generation failed: {e}")
        return None
# After TTS
if not tts_path:
    log_error(f"TTS generation failed for {target_lang_name}. Check gTTS installation or text content.")

# -------------------- Session State --------------------
if "conv" not in st.session_state:
    st.session_state.conv = []
if "errors" not in st.session_state:
    st.session_state.errors = []

# -------------------- Languages --------------------
LANG_CODES = {"Urdu":"ur","English":"en","Hindi":"hi","Arabic":"ar"}
STT_LANG_CODES = {"Urdu":"ur-PK","English":"en-US","Hindi":"hi-IN","Arabic":"ar-SA"}

# -------------------- Sidebar --------------------
st.sidebar.header("Settings & Privacy")
backend_choice = st.sidebar.selectbox("Backend", ["OpenAI (if available)","Google","Vosk (offline)"])
store_locally = st.sidebar.checkbox("Store transcripts locally")
agree_privacy = st.sidebar.checkbox("Audio contains no real patient data", value=True)
if not agree_privacy:
    st.stop()
st.sidebar.header("Errors / Warnings")
if st.session_state.errors:
    for err in st.session_state.errors:
        st.sidebar.error(err)
st.sidebar.markdown("### Session Info")
st.sidebar.write(f"OpenAI configured: {bool(openai_client)}")
st.sidebar.write(f"speech_recognition installed: {bool(sr)}")
st.sidebar.write(f"gTTS installed: {bool(gTTS)}")
st.sidebar.write(f"Google Translator installed: {bool(google_translator)}")
st.sidebar.write(f"Mic recorder available: {bool(mic_recorder)}")


# -------------------- Language Selectors --------------------
col_a,col_b = st.columns(2)
with col_a:
    input_lang_name = st.selectbox("Input (patient)", list(LANG_CODES.keys()), index=0)
with col_b:
    target_lang_name = st.selectbox("Clinician language", list(LANG_CODES.keys()), index=1)

st.markdown("## Live Translation Console")
left, right = st.columns([1,1])

# -------------------- Patient → Clinician --------------------
with left:
    st.subheader("🧑 Patient → Clinician")
    if mic_recorder:
        patient_record = mic_recorder(key="patient_rec")
        if patient_record:
            st.success("Recorded — processing...")
            wav_bytes = extract_wav_bytes(patient_record)
            if wav_bytes:
                tmp_wav = save_bytes_to_file(wav_bytes)
                raw_transcript = ""
                # Transcription
                if backend_choice.startswith("OpenAI") and openai_client:
                    raw_transcript = transcribe_openai(tmp_wav, LANG_CODES.get(input_lang_name))
                if not raw_transcript and sr:
                    raw_transcript = transcribe_google(tmp_wav, STT_LANG_CODES.get(input_lang_name))
                if store_locally:
                    Path("transcripts").mkdir(exist_ok=True)
                    Path(f"transcripts/{int(time.time())}_raw.txt").write_text(raw_transcript or "")
                # Translation
                cleaned, translated = ai_translate(raw_transcript, input_lang_name, target_lang_name)
                if not translated:
                    translated = google_translate(cleaned or raw_transcript, src=LANG_CODES.get(input_lang_name), dest=LANG_CODES.get(target_lang_name))
                # TTS
                tts_path = generate_tts(translated, LANG_CODES.get(target_lang_name))
                st.session_state.conv.append({"role":"Patient","original":raw_transcript,"cleaned":cleaned,"translated":translated,"tts":tts_path,"ts":time.time()})
                st.info(raw_transcript)
                st.success(translated)
                try: os.remove(tmp_wav)
                except: pass

# -------------------- Clinician → Patient --------------------
with right:
    st.subheader("👩‍⚕️ Clinician → Patient")
    doctor_mode = st.radio("Mode", ["Type","Record"], index=0)
    doctor_input_text = ""
    if doctor_mode=="Type":
        doctor_input_text = st.text_area("Clinician input", height=120)
        if st.button("➡ Translate & Add"):
            if doctor_input_text.strip():
                translated = ai_translate(doctor_input_text, "English", target_lang_name)[1] if openai_client else google_translate(doctor_input_text, src="en", dest=LANG_CODES.get(target_lang_name))
                tts_path = generate_tts(translated, LANG_CODES.get(target_lang_name))
                st.session_state.conv.append({"role":"Doctor","original":doctor_input_text,"cleaned":doctor_input_text,"translated":translated,"tts":tts_path,"ts":time.time()})
                st.success("Added to conversation")
    else:
        if mic_recorder:
            doc_record = mic_recorder(key="doctor_rec")
            if doc_record:
                st.success("Recorded — processing...")
                wav_bytes = extract_wav_bytes(doc_record)
                if wav_bytes:
                    tmp_wav = save_bytes_to_file(wav_bytes)
                    raw_transcript = transcribe_openai(tmp_wav,"en-US") if backend_choice.startswith("OpenAI") and openai_client else transcribe_google(tmp_wav,"en-US")
                    translated = ai_translate(raw_transcript,"English",target_lang_name)[1] if openai_client else google_translate(raw_transcript,src="en",dest=LANG_CODES.get(target_lang_name))
                    tts_path = generate_tts(translated, LANG_CODES.get(target_lang_name))
                    st.session_state.conv.append({"role":"Doctor","original":raw_transcript,"cleaned":raw_transcript,"translated":translated,"tts":tts_path,"ts":time.time()})
                    st.info(raw_transcript)
                    st.success(translated)
                    try: os.remove(tmp_wav)
                    except: pass

# -------------------- Conversation History --------------------
st.markdown("---")
st.header("Conversation History")
if st.session_state.conv:
    for idx,msg in enumerate(reversed(st.session_state.conv)):
        ts = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(msg.get("ts",time.time())))
        st.markdown(f"**{msg.get('role','User')}** — _{ts}_")
        st.write("Original:", msg.get("original",""))
        st.write("Cleaned:", msg.get("cleaned",""))
        st.write("Translated:", msg.get("translated",""))
        tts_path = msg.get("tts")
        if tts_path and Path(tts_path).exists():
            if st.button("🔊 Speak", key=f"play_{idx}"):
                st.audio(read_file_bytes(tts_path), format="audio/mp3")
            b64 = base64.b64encode(read_file_bytes(tts_path)).decode()
            st.markdown(f'<a download="tts_{idx}.mp3" href="data:audio/mp3;base64,{b64}">Download</a>', unsafe_allow_html=True)
        st.markdown("---")

# -------------------- Errors Panel --------------------
if st.session_state.errors:
    with st.expander("Errors / Warnings"):
        for e in st.session_state.errors:
            st.write(e)
