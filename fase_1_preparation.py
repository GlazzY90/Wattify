# fase_1_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

# === 1. BACA DATASET ===
df = pd.read_csv("dataset/household.csv")

# === 2. CEK DAN HAPUS MISSING VALUES ===
print("Missing values per kolom:")
print(df.isnull().sum())
df.dropna(inplace=True)
print()

# === 3. BUAT LABEL KATEGORI PENGGUNA ===
def categorize_user(amount):
    if amount < 500:
        return 'hemat'
    elif amount < 700:
        return 'netral'
    else:
        return 'boros'

df['user_category'] = df['amount_paid'].apply(categorize_user)

print(df['user_category'].value_counts())


# === 4. PILIH FITUR FINAL ===
fitur_final = [
    'num_rooms', 'num_people', 'housearea', 'is_ac', 'is_tv',
    'is_flat', 'ave_monthly_income', 'num_children', 'is_urban'
]

X = df[fitur_final]
y_reg = df['amount_paid']
y_clf = df['user_category']

# === 5. SPLIT DATA: TRAIN, VAL, TEST ===
# Train (70%) + Temp (30%)
X_train, X_temp, y_reg_train, y_reg_temp, y_clf_train, y_clf_temp = train_test_split(
    X, y_reg, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

# Val (15%) + Test (15%) dari Temp
X_val, X_test, y_reg_val, y_reg_test, y_clf_val, y_clf_test = train_test_split(
    X_temp, y_reg_temp, y_clf_temp, test_size=0.5, random_state=42, stratify=y_clf_temp
)

# === 6. SIMPAN FILE ===
os.makedirs("processed", exist_ok=True)
joblib.dump({
    "X_train": X_train,
    "X_val": X_val,
    "X_test": X_test,
    "y_reg_train": y_reg_train,
    "y_reg_val": y_reg_val,
    "y_reg_test": y_reg_test,
    "y_clf_train": y_clf_train,
    "y_clf_val": y_clf_val,
    "y_clf_test": y_clf_test,
    "feature_order": fitur_final
}, "processed/dataset_split.pkl")

print("✅ Data preparation selesai. File tersimpan di 'processed/dataset_split.pkl'")
