# वाक्यAI (VākyaAI)

**VākyaAI** is a high-performance Sanskrit Speech-to-Text (ASR) system designed to transcribe sacred and classical Sanskrit audio with high precision. Built using state-of-the-art transformer models, it provides a seamless bridge between oral tradition and digital text.

<img width="1722" height="950" alt="image" src="https://github.com/user-attachments/assets/4f4ecb7f-c920-4c72-8c18-dbf27570aa90" />



##  Features
-  **Audio Upload:** Support for WAV, MP3, M4A, and FLAC formats.
-  **Dual-Model Intelligence:**
  - **Whisper (Precision):** Leveraging `large-v3-turbo` with Hindi phonetic alignment for high-accuracy Devanagari output.
  - **HuBERT (Phonetic):** Utilizing `Wav2Vec2` architecture for robust phonetic capturing.
-  **Real-time Processing:** Fast inference pipeline with automated audio preprocessing (16kHz Mono).
-  **Elegant UI:** A warm, minimal, and user-friendly web interface designed for a premium experience.

## 🛠️ Tech Stack
- **Backend:** Python, Flask, Gunicorn
- **AI/ML:** OpenAI Whisper, Facebook HuBERT (Transformers, PyTorch)
- **Audio Processing:** Librosa, SoundFile
- **Frontend:** Vanilla HTML5, Modern CSS3 (Outfit Typography), JavaScript

##  Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vedikagupta02/-AI.git
   cd -AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.
