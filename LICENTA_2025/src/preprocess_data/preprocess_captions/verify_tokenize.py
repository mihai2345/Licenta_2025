#Verificăm structura fișierului cu captionuri tokenizate
import json

# Calea către fișierul tokenizat
tokens_path = "/content/tokenized_train.json"  

# Încarcă JSON-ul
with open(tokens_path, "r", encoding="utf-8") as f:
    token_data = json.load(f)

print(f" Număr total de captionuri: {len(token_data)}")
print(" Chei disponibile în fiecare intrare:", token_data[0].keys())
print("\n Primul exemplu:")
print(json.dumps(token_data[0], indent=2))