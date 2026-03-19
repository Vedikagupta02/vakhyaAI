# वाक्यAI (VākyaAI)

**VākyaAI** is a high-performance Sanskrit Speech-to-Text (ASR) system designed to transcribe sacred and classical Sanskrit audio with high precision. Built using state-of-the-art transformer architectures and custom-trained models, it bridges the gap between oral tradition and digital text.

<img width="1722" height="950" alt="image" src="https://github.com/user-attachments/assets/4f4ecb7f-c920-4c72-8c18-dbf27570aa90" />

---

## Features

- **Audio Upload:** Support for WAV, MP3, M4A, and FLAC formats  
- **Dual-Model Intelligence:**
  - **Whisper (Custom-Trained):** Built and fine-tuned for improved Sanskrit phonetic alignment and more accurate Devanagari transcription  
  - **HuBERT (Custom-Trained):** Developed using Wav2Vec2 architecture for robust phonetic representation of Sanskrit speech  
- **Real-time Processing:** Fast inference pipeline with automated audio preprocessing (16kHz mono)  
- **Elegant UI:** A warm, minimal, and user-friendly web interface designed for a premium experience  

---

## Model Development

- Built and experimented with custom Whisper and HuBERT-based models  
- Evaluated performance using WER (Word Error Rate) and CER (Character Error Rate) on Sanskrit datasets  
- Developed modular Python pipelines for preprocessing, inference, and evaluation  
- Included custom Python implementation files for both models in this repository  

---

## Tech Stack

- **Backend:** Python, Flask 
- **AI/ML:** Custom Whisper and HuBERT models (Transformers, PyTorch)    
- **Frontend:** HTML, CSS, JavaScript

