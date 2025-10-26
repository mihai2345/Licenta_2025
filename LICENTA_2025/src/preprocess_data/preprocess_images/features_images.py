#preprocesare imagini
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


# Căi fisiere

base_dir = '/content/drive/MyDrive/MedicalCaptioning'
image_folder = f'{base_dir}/images'
splits_folder = f'{base_dir}/splits'
features_folder = f'{base_dir}/features'
os.makedirs(features_folder, exist_ok=True)


#  Încarcă modelul pre-antrenat

base_model = InceptionV3(weights='imagenet')
model = tf.keras.Model(base_model.input, base_model.layers[-2].output)


# Funcție pentru extragerea feature-urilor

def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x, verbose=0)
        return np.squeeze(feat)
    except Exception as e:
        print(f" Eroare la {img_path}: {e}")
        return None


#  Funcție generală pentru procesare + salvare

def process_split(split_name):
    print(f"\n Procesăm setul: {split_name.upper()}")

    csv_path = f"{splits_folder}/{split_name}_split.csv"
    df = pd.read_csv(csv_path)
    features_dict = {}

    #  Verificăm dacă fișierul există deja
    save_path = f"{features_folder}/features_{split_name}.npz"
    if os.path.exists(save_path):
        print(f" Fișierul {save_path} există deja — se trece peste acest subset.")
        return

    for img_id in tqdm(df['ID']):
        img_path = os.path.join(image_folder, f"{img_id}.jpg")
        if os.path.exists(img_path):
            feat = extract_features(img_path)
            if feat is not None:
                features_dict[img_id] = feat
        else:
            print(f" Imagine lipsă: {img_id}.jpg")

    np.savez_compressed(save_path, **features_dict)
    print(f" Salvate feature-urile pentru {len(features_dict)} imagini în {save_path}")


#  Rulează pentru fiecare subset

for split in ["train", "val", "test"]:
    process_split(split)