import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# Importă modulele locale
from utils.data_utils import MedicalCaptionDataset, collate_fn 
from models.model_2_arch import ImageCaptioningModel_v2 # Arhitectura Model 2.0

# Setează variabila de mediu pentru memorie
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# Configurare Căi și Constante

BASE_DIR = "/content/drive/MyDrive/MedicalCaptioning"
MODEL_DIR = os.path.join(BASE_DIR, "model_2")
os.makedirs(MODEL_DIR, exist_ok=True) 

# Căi de date
FEATURES_TRAIN_PATH = os.path.join(BASE_DIR, "features/features_train.npz")
CAPTIONS_TRAIN_PATH = os.path.join(BASE_DIR, "tokenized_train.json")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab/vocab.json")

# Căi de checkpointing izolate în model_2
CSV_PATH = os.path.join(MODEL_DIR, "date_epoci_2.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model_2.pth")

# Hiperparametri și DataLoader

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)
vocab_size = len(vocab["itos"])

embed_size = 512
hidden_size = 768 # H=768 pentru Model 2.0
initial_learning_rate = 5e-4
num_epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# DataLoader (Batch Size 8, adesea necesar pentru Attention pe GPU)
train_dataset = MedicalCaptionDataset(FEATURES_TRAIN_PATH, CAPTIONS_TRAIN_PATH)
train_loader = DataLoader(
    train_dataset,
    batch_size=8, 
    shuffle=True,
    num_workers=0, 
    collate_fn=collate_fn
)


#  Instanțiere Model, Loss, Optimizer și Scheduler

model = ImageCaptioningModel_v2(embed_size, hidden_size, vocab_size).to(device) 

# CrossEntropyLoss este suficient
criterion = nn.CrossEntropyLoss(ignore_index=0) 
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

# Scheduler-ul ReducerLRonPlateau este standard pentru a reduce LR după o perioadă fără îmbunătățiri
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

print(f"🚀 Modelul 2.0 (Attention) se antrenează pe {device}")


#  Funcții de Antrenare/Validare (Adaptate pentru Atenție)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    # Atenție: Acum primim outputs și alphas din modelul V2
    for features, captions in tqdm(dataloader, desc="Training", leave=False):
        features, captions = features.to(device), captions.to(device)
        optimizer.zero_grad()
        
        outputs, alphas = model(features, captions[:, :-1])
        
        # Loss total = Loss de Generare + Loss de Regularizare (ponderea alpha)
        # Loss Regularizare (Doubly Stochastic Attention): alpha_sum - 1
        # Această regulă ajută atenția să acopere uniform imaginea
        attention_regularization_loss = ((1. - alphas.sum(dim=1)) ** 2).mean() 
        
        # Loss de Generare (Cross Entropy)
        generation_loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))
        
        # Total Loss (lambda=1.0 este un hyperparametru comun)
        loss = generation_loss + 1.0 * attention_regularization_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, captions in tqdm(dataloader, desc="Validation", leave=False):
            features, captions = features.to(device), captions.to(device)
            outputs, alphas = model(features, captions[:, :-1])
            
            attention_regularization_loss = ((1. - alphas.sum(dim=1)) ** 2).mean() 
            generation_loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))
            loss = generation_loss + 1.0 * attention_regularization_loss
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)



#  LOGICA DE CONTINUARE (Checkpointing Robust)

best_val_loss = float("inf")
start_epoch = 1
df_logs = pd.DataFrame(columns=["ID", "Train_Loss", "Val_Loss", "LR"]) 

# Logica de încărcare checkpoint (model, optimizer, scheduler)
if os.path.exists(CSV_PATH):
    
    # Simplificată pentru a nu repeta codul:

    if os.path.exists(MODEL_SAVE_PATH):
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['loss']
            
            # Reîncarcă și log-ul pentru continuitate
            df_logs = pd.read_csv(CSV_PATH)

            print(f" Continuăm antrenamentul de la Epoca {start_epoch}, încărcând modelul și starea OPTIMIZATORULUI.")
        except Exception as e:
            print(f" Eroare la încărcarea checkpoint-ului complet ({e}). Începem de la zero.")
            
else:
    print(f" Log-ul și antrenamentul încep de la Epoca 1 (Arhitectură nouă).")



# Bucla Principală de Antrenament

for epoch in range(start_epoch, num_epochs + 1):
    print(f"\n====================== 🔁 Epoch {epoch}/{num_epochs} ======================")

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    val_loss = validate_one_epoch(model, train_loader, criterion, device)

    current_lr = optimizer.param_groups[0]['lr']
    print(f" Train Loss: {train_loss:.4f} |  Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # SALVEAZĂ CHECKPOINT-UL COMPLET (Format Nou: Model + Optimizer + Epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'loss': best_val_loss,
        }, MODEL_SAVE_PATH)
        print(f" Model îmbunătățit (greutăți + optimizator) salvat în Drive la {MODEL_SAVE_PATH}!")

    # Aplică Scheduler-ul
    scheduler.step(val_loss)

    # Loghează și Salvează
    df_logs.loc[len(df_logs)] = [epoch, train_loss, val_loss, current_lr]
    df_logs.to_csv(CSV_PATH, index=False)
    torch.cuda.empty_cache()

print(f"\n Toate valorile salvate în: {CSV_PATH}")