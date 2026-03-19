import torch
import librosa
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# -----------------------------
# 1. Initialization (Run once)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None

def load_whisper_model():
    global processor, model
    if processor is None:
        # Upgrading to large-v3-turbo for much higher accuracy
        MODEL_PATH = "openai/whisper-large-v3-turbo"
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
        
        # Force Hindi (hi) for better phonetic matching with Sanskrit on base models
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="hindi",
            task="transcribe"
        )
        model.config.suppress_tokens = []
        model.to(device)
        model.eval()

# -----------------------------
# 2. Helper Functions
# -----------------------------
def normalize_sanskrit_text(text):
    """
    Relaxed normalization to allow Devanagari results to show up.
    """
    text = text.strip()
    # Keep Devanagari, spaces, and basic punctuation
    text = re.sub(r"[^\u0900-\u097F\s।॥]", "", text) 
    text = re.sub(r"\s+", " ", text)
    return text

# -----------------------------
# 3. Core Inference Function
# -----------------------------
def transcribe_whisper(audio_path):
    """
    Transcribes a specific audio file using the configured Whisper model.
    """
    load_whisper_model()
    # 1. Load audio and resample to 16kHz (Whiper requirement)
    audio_array, sr = librosa.load(audio_path, sr=16000)
    
    # 2. Audio -> log-Mel features
    inputs = processor(
        audio_array,
        sampling_rate=sr,
        return_tensors="pt"
    )
    
    # Move features to same device as model
    input_features = inputs.input_features.to(device)
    
    # 3. Generate prediction
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=model.config.forced_decoder_ids,
            max_length=225
        )
        
    # 4. Decode prediction to text
    pred_text = processor.tokenizer.decode(
        generated_ids[0], 
        skip_special_tokens=True
    )
    
    # Debug print so you can see what the model actually thought it heard!
    print("RAW OUTPUT BEFORE NORMALIZATION:", pred_text)
    
    # 5. Apply your Sanskrit normalization
    text = normalize_sanskrit_text(pred_text)
    print("FINAL NORMALIZED TEXT:", text)
    
    return text

# ==========================================
# Example Usage:
# text_output = transcribe_whisper("path/to/my_audio_file.wav")
# print(text_output)
# ==========================================
