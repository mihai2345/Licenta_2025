
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import torch.nn.utils.rnn as rnn_utils


#   Clasa Dataset

class MedicalCaptionDataset(Dataset):
    def __init__(self, features_path, captions_path):
        # Încarcă vectorii de feature (.npz)
        self.features = np.load(features_path, allow_pickle=True)

        # Încarcă captionurile tokenizate (.json)
        with open(captions_path, "r", encoding="utf-8") as f:
            self.captions = json.load(f)

        # Construim mapare rapidă image_id → tokens
        self.caption_map = {item["image_id"]: item["tokens"] for item in self.captions}

        # Listează doar ID-urile care există în ambele seturi
        self.ids = [img_id for img_id in self.features.files if img_id in self.caption_map]

        print(f"Dataset încărcat cu {len(self.ids)} exemple comune.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        feature = torch.tensor(self.features[img_id], dtype=torch.float32)
        caption = torch.tensor(self.caption_map[img_id], dtype=torch.long)
        return feature, caption


#  Funcție pentru collate_fn (padding dinamic)

def collate_fn(batch):
    features, captions = zip(*batch)
    features = torch.stack(features)
    captions_padded = rnn_utils.pad_sequence(captions, batch_first=True, padding_value=0)  # <pad> = 0
    return features, captions_padded


#  Creare DataLoader


features_train = "/content/drive/MyDrive/MedicalCaptioning/features/features_train.npz"
captions_train = "/content/drive/MyDrive/MedicalCaptioning/tokenized_train.json"

train_dataset = MedicalCaptionDataset(features_train, captions_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,         
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)


# Verificare rapidă a unui batch

for features, captions in train_loader:
    print("🔸 features:", features.shape)   # (batch, 2048)
    print("🔸 captions:", captions.shape)   # (batch, seq_len_max)
    print("Exemplu primii 10 indici caption:", captions[0][:10])
    break