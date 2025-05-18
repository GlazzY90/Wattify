from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime
import os
import subprocess
import sys

app = Flask(__name__)

# Function to run a Python script and handle errors
def run_script(script_name):
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(f"Successfully ran {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:\n{e.stderr}")
        raise Exception(f"Failed to run {script_name}: {e.stderr}")
    except FileNotFoundError:
        print(f"Script {script_name} not found")
        raise Exception(f"Script {script_name} not found")

# Run preparation and training scripts
try:
    print("Running fase_1_preparation.py...")
    run_script("fase_1_preparation.py")
    print("Running model_training.py...")
    run_script("model_training.py")
except Exception as e:
    print(f"Initialization error: {e}")
    exit(1)

# Load models, scaler, feature order, and metrics
try:
    regressor = joblib.load("model/regressor.pkl")
    classifier = joblib.load("model/classifier.pkl")
    scaler = joblib.load("model/scaler.pkl")
    with open("model/feature_order.json", "r") as f:
        feature_order = json.load(f)
    with open("model/metrics.json", "r") as f:
        metrics = json.load(f)
    regressor_accuracy = metrics["regressor_r2"]
    classifier_accuracy = metrics["classifier_accuracy"]
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    regressor, classifier, scaler, feature_order = None, None, None, []
    regressor_accuracy, classifier_accuracy = 0.0, 0.0

# Path to the CSV file for logging predictions
LOG_FILE = 'predictions_log.csv'

# Initialize CSV file with headers
def init_log_file():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=['timestamp', 'input', 'amount_paid', 'user_category']).to_csv(LOG_FILE, index=False)

# Log prediction to CSV
def log_prediction(inputs, amount_paid, user_category):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = pd.DataFrame([[timestamp, str(inputs), amount_paid, user_category]], 
                            columns=['timestamp', 'input', 'amount_paid', 'user_category'])
    log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)

# Generate suggestion based on user_category
def get_suggestion(user_category):
    if user_category == 'boros':
        return 'Berhemat Lagi Yah!'
    elif user_category == 'netral':
        return 'Penggunaan Anda Cukup Seimbang!'
    else:  # hemat
        return 'Hebat, Anda Sangat Hemat!'

# Route for the landing page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the first form (Data Anda)
@app.route('/form', methods=['POST', 'GET'])
def form():
    if request.method == 'POST':
        try:
            # Store form data in session
            session['form1_data'] = {
                'jumlah_ruangan': float(request.form['jumlah_ruangan']),
                'jumlah_orang': float(request.form['jumlah_orang']),
                'luas_rumah': float(request.form['luas_rumah']),
                'pendapatan_bulanan': float(request.form['pendapatan_bulanan']),
                'jumlah_anak': float(request.form['jumlah_anak'])
            }
            return redirect(url_for('keterangan'))
        except ValueError as e:
            return render_template('form.html', error=f"Invalid input: {str(e)}")
    return render_template('form.html')

# Route for the second form (Keterangan)
@app.route('/keterangan', methods=['POST', 'GET'])
def keterangan():
    if request.method == 'POST':
        if 'form1_data' not in session:
            return redirect(url_for('form'))  # Redirect if first form data is missing
        
        try:
            # Collect second form data and convert "Ya"/"Tidak" to 1/0
            form2_data = {
                'is_ac': 1 if request.form['ac'] == 'Ya' else 0,
                'is_tv': 1 if request.form['tv'] == 'Ya' else 0,
                'is_flat': 1 if request.form['flat'] == 'Ya' else 0,
                'is_urban': 1 if request.form['area_urban'] == 'Ya' else 0
            }
            
            # Combine form data in the correct feature order
            input_data = [
                session['form1_data']['jumlah_ruangan'],
                session['form1_data']['jumlah_orang'],
                session['form1_data']['luas_rumah'],
                form2_data['is_ac'],
                form2_data['is_tv'],
                form2_data['is_flat'],
                session['form1_data']['pendapatan_bulanan'],
                session['form1_data']['jumlah_anak'],
                form2_data['is_urban']
            ]
            
            # Make prediction
            if regressor is None or classifier is None or scaler is None:
                return render_template('result.html', result={'error': "Model atau scaler tidak ditemukan"})
            
            # Convert input to numpy array
            input_array = np.array(input_data).reshape(1, -1)
            
            # Scale numeric features for classifier
            numeric_features = ['num_rooms', 'num_people', 'housearea', 'ave_monthly_income', 'num_children']
            numeric_indices = [feature_order.index(f) for f in numeric_features]
            input_array_scaled = input_array.copy()
            input_array_scaled[:, numeric_indices] = scaler.transform(input_array[:, numeric_indices])
            
            # Predict
            amount_paid = regressor.predict(input_array)[0]
            user_category = classifier.predict(input_array_scaled)[0]
            
            # Log prediction
            init_log_file()
            log_prediction(input_data, amount_paid, user_category)
            
            # Prepare result
            result = {
                'amount_paid': round(amount_paid, 2),
                'user_category': user_category.capitalize(),
                'regressor_accuracy': round(regressor_accuracy, 2),
                'classifier_accuracy': round(classifier_accuracy, 2),
                'suggestion': get_suggestion(user_category)
            }
            
            return render_template('result.html', result=result)
        except Exception as e:
            return render_template('result.html', result={'error': f"Error prediksi: {str(e)}"})
    
    return render_template('keterangan.html')

if __name__ == '__main__':
    app.run(debug=True)