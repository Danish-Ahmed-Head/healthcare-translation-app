# 🩺 Healthcare Translation Assistant (Patient ↔ Doctor)

A web-based prototype for real-time multilingual translation between patients and clinicians.
Built with **Streamlit**. Uses Google Web Speech + googletrans + gTTS by default, with optional
OpenAI Whisper + GPT integration for AI-enhanced transcription and clinical translation.

---

## Features
- Voice → text transcription (patient & clinician)
- AI-enhanced post-processing / clinical translation (optional via OpenAI)
- Per-turn translation + text-to-speech playback (Speak button + download)
- Conversation history with original / cleaned / translated text
- Language selector for input and output
- Privacy note and local storage toggle

---

## Quick start (local)

1. Create & activate a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
