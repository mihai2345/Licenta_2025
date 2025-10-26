import pandas as pd
import json
import re
from tqdm import tqdm

# === CONFIG ===
base_path = "/content"  
vocab_path = f"{base_path}/vocab.json"

splits = {
    "train": f"{base_path}/train_split.csv",
    "val": "/content/drive/MyDrive/MedicalCaptioning/splits/val_split.csv",
    "test": "/content/drive/MyDrive/MedicalCaptioning/splits/test_split.csv",
}

# === ÎNCARCĂ VOCABULAR ===
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)
stoi = vocab["stoi"]

# === FUNCȚIE DE TOKENIZARE ===
def tokenize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\-\/]+", " ", text)
    tokens = text.split()
    token_ids = [stoi.get(tok, stoi["<unk>"]) for tok in tokens]
    return [stoi["<bos>"]] + token_ids + [stoi["<eos>"]]

# === TOKENIZARE PENTRU FIECARE SPLIT ===
for split_name, csv_path in splits.items():
    print(f"\n Procesăm splitul: {split_name.upper()}")
    df = pd.read_csv(csv_path)
    tokenized_data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = str(row["ID"]).strip()
        caption = str(row["Caption"])
        tokens = tokenize_text(caption)
        tokenized_data.append({"image_id": image_id, "tokens": tokens})

    # === SALVARE ===
    save_path = f"{base_path}/tokenized_{split_name}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(tokenized_data, f, indent=2, ensure_ascii=False)

    print(f"Tokenizare completă pentru {split_name} — {len(tokenized_data)} exemple salvate în {save_path}")