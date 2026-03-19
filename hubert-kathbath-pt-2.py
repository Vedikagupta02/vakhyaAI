# Generated from: hubert-kathbath-pt-2.ipynb
# Converted at: 2026-03-19T14:20:21.758Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!pip install -q transformers datasets torchaudio jiwer accelerate


from huggingface_hub import login

login("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")  # paste your token


from datasets import load_dataset

dataset = load_dataset(
    "ai4bharat/Kathbath",
    "sanskrit",
    token=True
)


print(dataset)


from datasets import Audio

dataset = dataset.cast_column(
    "audio_filepath",
    Audio(decode=False)
)


sample = dataset["train"][0]

print("Text:", sample["text"][:80])
print("Duration:", sample["duration"])
print("Audio ref:", sample["audio_filepath"])


import unicodedata
import re

def normalize_sanskrit(text):
    # 1. Unicode normalization (important)
    text = unicodedata.normalize("NFC", text)

    # 2. Remove unwanted punctuation (keep danda if you want)
    text = re.sub(r"[^\u0900-\u097F\s।]", "", text)

    # 3. Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


for i in range(3):
    raw = dataset["train"][i]["text"]
    norm = normalize_sanskrit(raw)

    print("RAW :", raw)
    print("NORM:", norm)
    print("-" * 50)


def normalize_batch(batch):
    batch["normalized_text"] = normalize_sanskrit(batch["text"])
    return batch

dataset = dataset.map(
    normalize_batch,
    remove_columns=[],
    desc="Normalizing Sanskrit text"
)


def extract_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    return {"chars": list(set(all_text))}

chars = dataset["train"].map(
    extract_chars,
    batched=True,
    batch_size=1000,
    remove_columns=dataset["train"].column_names
)

vocab_chars = set()
for item in chars["chars"]:
    vocab_chars.update(item)

vocab_chars = sorted(vocab_chars)
print("Total characters:", len(vocab_chars))
print(vocab_chars)


vocab_dict = {v: i for i, v in enumerate(vocab_chars)}

# Add special tokens
vocab_dict["<pad>"] = len(vocab_dict)
vocab_dict["<unk>"] = len(vocab_dict)
vocab_dict["|"] = len(vocab_dict)  # word boundary (space replacement)

# Replace space with |
vocab_dict[" "] = vocab_dict["|"]
del vocab_dict[" "]

print("Final vocab size:", len(vocab_dict))


import json

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)


from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|"
)


from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)


from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)


sample["audio_filepath"]["bytes"]


import unicodedata
import re

def normalize_sanskrit(text):
    # 1. Unicode normalization (important)
    text = unicodedata.normalize("NFC", text)

    # 2. Remove unwanted punctuation (keep danda if you want)
    text = re.sub(r"[^\u0900-\u097F\s।]", "", text)

    # 3. Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


for i in range(3):
    raw = dataset["train"][i]["text"]
    norm = normalize_sanskrit(raw)

    print("RAW :", raw)
    print("NORM:", norm)
    print("-" * 50)


def normalize_batch(batch):
    batch["normalized_text"] = normalize_sanskrit(batch["text"])
    return batch

dataset = dataset.map(
    normalize_batch,
    remove_columns=[],
    desc="Normalizing Sanskrit text"
)


def extract_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    return {"chars": list(set(all_text))}

chars = dataset["train"].map(
    extract_chars,
    batched=True,
    batch_size=1000,
    remove_columns=dataset["train"].column_names
)

vocab_chars = set()
for item in chars["chars"]:
    vocab_chars.update(item)

vocab_chars = sorted(vocab_chars)
print("Total characters:", len(vocab_chars))
print(vocab_chars)


vocab_dict = {v: i for i, v in enumerate(vocab_chars)}

# Add special tokens
vocab_dict["<pad>"] = len(vocab_dict)
vocab_dict["<unk>"] = len(vocab_dict)
vocab_dict["|"] = len(vocab_dict)  # word boundary (space replacement)

# Replace space with |
vocab_dict[" "] = vocab_dict["|"]
del vocab_dict[" "]

print("Final vocab size:", len(vocab_dict))


import json

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)


from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|"
)


from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)


from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)


import soundfile as sf
import io
import numpy as np

def load_audio_from_bytes(audio_dict):
    audio_bytes = audio_dict["bytes"]

    with io.BytesIO(audio_bytes) as f:
        waveform, sr = sf.read(f)

    # Convert stereo → mono
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    # Ensure 16kHz
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    return waveform, sr


sample = dataset["train"][0]

waveform, sr = load_audio_from_bytes(sample["audio_filepath"])

print("Waveform shape:", waveform.shape)
print("Sampling rate:", sr)
print("Duration (sec):", len(waveform) / sr)


def prepare_example(batch):
    waveform, sr = load_audio_from_bytes(batch["audio_filepath"])

    batch["input_values"] = processor(
        waveform,
        sampling_rate=sr
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["normalized_text"]).input_ids

    return batch


test_sample = dataset["train"].select(range(5)).map(
    prepare_example,
    remove_columns=dataset["train"].column_names
)


from dataclasses import dataclass
from typing import Dict, List, Union
import torch

@dataclass
class DataCollatorCTCWithPadding:
    processor: any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # separate inputs and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )

        # replace padding with -100 (ignored by CTC loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)


from transformers import HubertForCTC

model = HubertForCTC.from_pretrained(
    "facebook/hubert-base-ls960",
    vocab_size=len(processor.tokenizer),
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)


model.freeze_feature_encoder()


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


test_batch = data_collator(
    dataset["train"].select(range(2))
)

test_batch = {k: v.to(device) for k, v in test_batch.items()}

with torch.no_grad():
    outputs = model(**test_batch)

print("Loss:", outputs.loss.item())


def prepare_example(example):
    audio_bytes = example["audio_filepath"]["bytes"]

    waveform, sr = sf.read(io.BytesIO(audio_bytes))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != 16000:
        waveform = librosa.resample(waveform, sr, 16000)

    input_values = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values[0]

    labels = processor.tokenizer(
        example["text"],
        return_tensors="pt"
    ).input_ids[0]

    return {
        "input_values": input_values,
        "labels": labels
    }


import io
import soundfile as sf
import numpy as np
import torch

class DataCollatorCTCWithPadding:
    def __init__(self, processor, sampling_rate=16000):
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __call__(self, features):
        audio_arrays = []
        texts = []

        for f in features:
            audio_bytes = f["audio_filepath"]["bytes"]

            # Decode FLAC bytes
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sr = sf.read(audio_buffer, dtype="float32")

            if sr != self.sampling_rate:
                raise ValueError(f"Expected {self.sampling_rate}, got {sr}")

            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            audio_arrays.append(waveform)   # NumPy array
            texts.append(f["text"])

        # Audio → padded tensors
        inputs = self.processor(
            audio_arrays,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt"
        )

        # ✅ FIX: tokenizer ONLY for labels
        labels = self.processor.tokenizer(
            texts,
            padding=True,
            return_tensors="pt"
        )

        inputs["labels"] = labels["input_ids"]
        inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return inputs


from torch.utils.data import DataLoader

data_collator = DataCollatorCTCWithPadding(processor)

train_loader = DataLoader(
    dataset["train"],
    batch_size=2,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    dataset["valid"],
    batch_size=2,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=2,
    pin_memory=True
)


device = "cuda"
model.to(device)
model.train()

batch = next(iter(train_loader))
batch = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    outputs = model(**batch)

print("Loss:", outputs.loss.item())


model.freeze_feature_encoder()
print("Feature encoder frozen")


from torch.optim import AdamW
from transformers import get_scheduler

optimizer = AdamW(
    model.parameters(),
    lr=1e-4
)

num_epochs = 10
gradient_accumulation_steps = 8
num_training_steps = (
    len(train_loader) // gradient_accumulation_steps
) * num_epochs

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps,
)


from tqdm.auto import tqdm
import math

device = "cuda"
model.train()

global_step = 0

for epoch in range(num_epochs):
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(progress):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Skipping NaN/Inf loss")
            optimizer.zero_grad()
            continue

        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        progress.set_postfix(loss=loss.item() * gradient_accumulation_steps)

    # 🔹 Save checkpoint each epoch
    model.save_pretrained(f"./hubert-sanskrit-epoch{epoch+1}")
    processor.save_pretrained(f"./hubert-sanskrit-epoch{epoch+1}")


import os
import torch

ckpt_dir = "./hubert-sanskrit-checkpoint"
os.makedirs(ckpt_dir, exist_ok=True)

# 1️⃣ Model + processor
model.save_pretrained(ckpt_dir)
processor.save_pretrained(ckpt_dir)

# 2️⃣ Optimizer + scheduler
torch.save({
    "optimizer": optimizer.state_dict(),
    "scheduler": lr_scheduler.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
}, os.path.join(ckpt_dir, "trainer_state.pt"))

print("✅ Checkpoint saved")


import jiwer
import torch

model.eval()

predictions = []
references = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward
        logits = model(batch["input_values"]).logits
        pred_ids = torch.argmax(logits, dim=-1)

        # 🔧 FIX 1: remove special tokens
        preds = processor.batch_decode(
            pred_ids,
            skip_special_tokens=True
        )

        # 🔧 FIX 2: clean labels
        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id

        refs = processor.batch_decode(
            labels,
            skip_special_tokens=True,
            group_tokens=False
        )

        predictions.extend(preds)
        references.extend(refs)

# Metrics
wer = jiwer.wer(references, predictions)
cer = jiwer.cer(references, predictions)

print(f"Validation WER: {wer:.4f}")
print(f"Validation CER: {cer:.4f}")


import torch
import io
import soundfile as sf

def show_predictions(dataset, model, processor, num_samples=5):
    model.eval()
    device = next(model.parameters()).device

    print("\n🔍 Showing predictions:\n")

    for i in range(num_samples):
        sample = dataset[i]

        # ---- Decode audio from bytes ----
        audio_bytes = sample["audio_filepath"]["bytes"]
        waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        assert sr == 16000, f"Expected 16kHz, got {sr}"

        # ---- Processor ----
        inputs = processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ---- Model inference ----
        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)

        # ---- 🔧 FIXED decoding ----
        pred_text = processor.batch_decode(
            pred_ids,
            skip_special_tokens=True
        )[0]

        ref_text = sample["text"]

        print(f"🔹 Sample {i+1}")
        print("REF :", ref_text)
        print("PRED:", pred_text)
        print("-" * 70)

    model.train()


show_predictions(
    dataset["valid"],
    model,
    processor,
    num_samples=10
)