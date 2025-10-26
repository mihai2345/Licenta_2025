#preprocesare text plus creeare vocabular
import pandas as pd
import json
import re
from collections import Counter


csv_path = "/content/train_split.csv" 
output_path = "/content/vocab.json"

# === CITIRE CSV ===
df = pd.read_csv(csv_path)
print(f"[INFO] Fișier încărcat cu {len(df)} rânduri")
print("📊 Coloane disponibile:", list(df.columns))
print("\nExemplu rând:")
print(df.iloc[0])

# === FUNCȚIE DE TOKENIZARE ===
def tokenize(text):
    # lowercase + eliminare spații inutile
    text = text.lower().strip()
    # păstrăm litere, cifre, "-" și "/" (ex: x-ray, t1/t2)
    text = re.sub(r"[^a-z0-9\-\/]+", " ", text)
    tokens = text.split()
    return tokens

# === CONSTRUIRE VOCABULAR ===
counter = Counter()
for caption in df["Caption"]:
    tokens = tokenize(str(caption))
    counter.update(tokens)

# === Pastrare TOATE CUVINTELE 
words = [w for w, c in counter.items() if c >= 1]
print(f"\n Număr total de cuvinte unice: {len(words)}")

# === TOKENURI SPECIALE ===
special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
itos = special_tokens + sorted(words)
stoi = {w: i for i, w in enumerate(itos)}

# === SALVARE VOCABULAR ===
vocab = {"stoi": stoi, "itos": itos}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2, ensure_ascii=False)

print(f"\n Vocabular salvat în: {output_path}")
print(f" Dimensiune vocabular total: {len(itos)}")
print(f"Primele 30 tokenuri: {itos[:30]}")