#impartire set in train validation si test 80% / 10% / 10%
import pandas as pd
from sklearn.model_selection import train_test_split
import os

#  Căi către fișiere
csv_path = '/content/drive/MyDrive/MedicalCaptioning/train_captions.csv'
save_dir = '/content/drive/MyDrive/MedicalCaptioning/splits'

#  Creăm folderul /splits 
os.makedirs(save_dir, exist_ok=True)

#  Citim fișierul complet cu captionuri
df = pd.read_csv(csv_path)
print(f"📄 Total intrări în dataset: {len(df)}")

#  Împărțim datasetul în train (80%) și temp (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

#   Din temp, împărțim 50/50 => validation (10%) și test (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

#  Afișăm proporțiile
print("\n Distribuția seturilor:")
print(f" Train: {len(train_df)}")
print(f" Validation: {len(val_df)}")
print(f" Test: {len(test_df)}")

#  Salvăm fișierele CSV în /splits
train_path = f"{save_dir}/train_split.csv"
val_path = f"{save_dir}/val_split.csv"
test_path = f"{save_dir}/test_split.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n Fișiere salvate cu succes în:", save_dir)
print(f" - {train_path}")
print(f" - {val_path}")
print(f" - {test_path}")