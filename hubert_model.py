import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, HubertForCTC

# -----------------------------
# 1. Initialization (Run once)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None

def load_hubert_model():
    global processor, model
    if processor is None:
        # Using a reliable large model for better base accuracy
        CHECKPOINT_PATH = "facebook/wav2vec2-large-960h" 
        
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        processor = Wav2Vec2Processor.from_pretrained(CHECKPOINT_PATH)
        model = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT_PATH)
        model.to(device)
        model.eval()

# -----------------------------
# 2. Core Inference Function
# -----------------------------
def transcribe_hubert(audio_path):
    """
    Transcribes a single audio file using the fine-tuned HuBERT-CTC model.
    """
    load_hubert_model()
    # 1. Load audio file using soundfile
    waveform, sr = sf.read(audio_path, dtype="float32")
    
    # 2. Convert stereo to mono if needed
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
        
    # 3. Resample to 16kHz (HuBERT requirement)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        
    # 4. Extract features
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 5. Model inference (CTC)
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # 6. Decode predictions using the correct pipeline
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Properly decode to text
    pred_text = processor.batch_decode(predicted_ids)[0]
    
    return pred_text.lower()


# ==========================================
# Example Usage:
# text_output = transcribe_hubert("path/to/my_audio_file.wav")
# print(text_output)
# ==========================================
