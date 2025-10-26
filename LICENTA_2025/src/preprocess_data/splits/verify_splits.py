#verificare impartire corecta prin proportionalitate
import pandas as pd

#  Căi către fișiere
base_path = '/content/drive/MyDrive/MedicalCaptioning'
csv_full = f'{base_path}/train_captions.csv'
train_split = f'{base_path}/splits/train_split.csv'
val_split = f'{base_path}/splits/val_split.csv'
test_split = f'{base_path}/splits/test_split.csv'

#  Citim fișierul complet și pe cele împărțite
df_full = pd.read_csv(csv_full)
df_train = pd.read_csv(train_split)
df_val = pd.read_csv(val_split)
df_test = pd.read_csv(test_split)

#  Număr total rânduri
total = len(df_full)
train_n = len(df_train)
val_n = len(df_val)
test_n = len(df_test)

print(f" Total în fișierul complet: {total}")
print("\nDistribuție fișiere split:")
print(f"Train: {train_n} ({train_n/total*100:.2f}%)")
print(f" Validation: {val_n} ({val_n/total*100:.2f}%)")
print(f" Test: {test_n} ({test_n/total*100:.2f}%)")

#  Verificăm începutul și sfârșitul fiecărui fișier
def show_edges(name, df):
    print(f"\n{name} — Primul rând:")
    print(df.iloc[0])
    print(f"\n{name} — Ultimul rând:")
    print(df.iloc[-1])

show_edges(" Train", df_train)
show_edges(" Validation", df_val)
show_edges(" Test", df_test)