# main.py

import customtkinter as ctk
import joblib
import numpy as np
import json
import pandas as pd

# === Load Model, Scaler & Feature Order ===
reg_model = joblib.load("model/regressor.pkl")
clf_model = joblib.load("model/classifier.pkl")
scaler = joblib.load("model/scaler.pkl")

with open("model/feature_order.json", "r") as f:
    fitur_order = json.load(f)

numeric_features = ['num_rooms', 'num_people', 'housearea', 'ave_monthly_income', 'num_children']

# === Setup GUI ===
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("700x800")
app.title("🔌 Prediksi Tagihan Listrik & Kategori Pengguna")

# === FRAME CONTAINER ===
frame_main = ctk.CTkFrame(app)
frame_main.pack(pady=20, padx=20, fill="both", expand=True)

# === Header ===
ctk.CTkLabel(frame_main, text="📋 Input Data Rumah Tangga", font=("Arial", 24, "bold")).pack(pady=15)

# === Input Fields ===
form_frame = ctk.CTkFrame(frame_main)
form_frame.pack(pady=10, padx=20, fill="x")

def add_entry(label_text, entry_var):
    ctk.CTkLabel(form_frame, text=label_text, anchor="w").pack(fill="x", padx=10, pady=(5, 0))
    entry = ctk.CTkEntry(form_frame, textvariable=entry_var)
    entry.pack(fill="x", padx=10, pady=5)
    return entry

var_rooms = ctk.StringVar()
var_people = ctk.StringVar()
var_area = ctk.StringVar()
var_income = ctk.StringVar()
var_children = ctk.StringVar()

entry_rooms = add_entry("Jumlah Ruangan", var_rooms)
entry_people = add_entry("Jumlah Orang", var_people)
entry_area = add_entry("Luas Rumah (m2)", var_area)
entry_income = add_entry("Pendapatan Bulanan ($)", var_income)
entry_children = add_entry("Jumlah Anak", var_children)

# === Switches ===
switch_frame = ctk.CTkFrame(frame_main)
switch_frame.pack(pady=15, padx=20, fill="x")

switch_ac = ctk.CTkSwitch(switch_frame, text="Menggunakan AC")
switch_ac.pack(anchor="w", pady=5, padx=10)

switch_tv = ctk.CTkSwitch(switch_frame, text="Memiliki TV")
switch_tv.pack(anchor="w", pady=5, padx=10)

switch_flat = ctk.CTkSwitch(switch_frame, text="Tipe Flat")
switch_flat.pack(anchor="w", pady=5, padx=10)

switch_urban = ctk.CTkSwitch(switch_frame, text="Tinggal di Area Urban")
switch_urban.pack(anchor="w", pady=5, padx=10)

# === Fungsi Prediksi ===
def prediksi():
    try:
        input_dict = {
            'num_rooms': int(var_rooms.get()),
            'num_people': int(var_people.get()),
            'housearea': float(var_area.get()),
            'is_ac': int(switch_ac.get()),
            'is_tv': int(switch_tv.get()),
            'is_flat': int(switch_flat.get()),
            'ave_monthly_income': float(var_income.get()),
            'num_children': int(var_children.get()),
            'is_urban': int(switch_urban.get())
        }

        input_df = pd.DataFrame([input_dict])
        input_df_ordered = input_df[fitur_order]

        # === Prediksi Regresi ===
        amount_pred = reg_model.predict(input_df_ordered)[0]

        # === Prediksi Klasifikasi ===
        input_df_scaled = input_df_ordered.copy()
        input_df_scaled[numeric_features] = scaler.transform(input_df[numeric_features])
        kategori_pred = clf_model.predict(input_df_scaled)[0]

        label_result.configure(
            text=(f"📊 Tagihan Diprediksi: ${amount_pred:.2f}\n⚡️ Kategori: {kategori_pred}"),
            text_color="green"
        )

    except Exception as e:
        label_result.configure(text=f"❌ Error: {str(e)}", text_color="red")

# === Tombol Prediksi ===
ctk.CTkButton(frame_main, text="🔍 Prediksi Sekarang", command=prediksi, height=40, font=("Arial", 16)).pack(pady=20)

# === Label Hasil ===
label_result = ctk.CTkLabel(frame_main, text="", font=("Arial", 16), text_color="black", wraplength=600, justify="center")
label_result.pack(pady=10)

app.mainloop()
