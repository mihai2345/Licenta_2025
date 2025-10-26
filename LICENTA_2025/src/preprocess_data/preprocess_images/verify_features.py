#Verificăm structura fișierului cu feature-uri (features_train.npz)
import numpy as np

# Calea către features_train.npz 
features_path = "/content/drive/MyDrive/MedicalCaptioning/features/features_train.npz"

# Încarcare fișierul .npz
data = np.load(features_path, allow_pickle=True)

# Listează toate cheile =
print(f"🔹 Număr total imagini în features_train: {len(data.files)}")
print("🔹 Primele 5 chei (image IDs):", data.files[:5])

# Afișează forma unui vector de feature
sample_key = data.files[0]
print(f"\n Exemplu imagine: {sample_key}")
print(" Formă vector feature:", data[sample_key].shape)
print(" Primii 10 parametri:", data[sample_key][:10])