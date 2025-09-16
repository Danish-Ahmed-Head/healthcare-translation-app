# User Guide — Healthcare Translation Assistant

**Purpose**  
This prototype helps clinicians and patients communicate across languages by converting short spoken phrases into text, translating them, and playing translated audio. It is a demo for a pre-interview assignment — do not use with real patient data without proper agreements.

---

## Quick highlights
- Record patient speech in the browser → get a transcript + clinician-facing translation (text + playable audio).
- Clinician can type or record a reply → patient-facing translation + audio.
- Conversation history stores original, cleaned, and translated text for each turn.
- Two operation modes: **AI-enhanced** (OpenAI Whisper + GPT, optional) or **fallback** (Google Web Speech + googletrans).

---

## Files in the repo
- `app.py` — main Streamlit application
- `requirements.txt` — Python dependencies
- `README.md` — project overview + setup
- `UserGuide.md` — (this file) instructions for use

---

## Before you start (requirements)
1. Python 3.8+ and pip.
2. Install dependencies:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate

   pip install -r requirements.txt
