from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Route for the landing page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the first form (Data Anda)
@app.route('/form', methods=['POST', 'GET'])
def form():
    if request.method == 'POST':
        jumlah_ruangan = request.form['jumlah_ruangan']
        jumlah_orang = request.form['jumlah_orang']
        luas_rumah = request.form['luas_rumah']
        pendapatan_bulanan = request.form['pendapatan_bulanan']
        jumlah_anak = request.form['jumlah_anak']
        
        # Logic to handle the form submission
        # For example, passing data to the next page
        return redirect(url_for('keterangan'))  # Redirect to the Keterangan page
    
    return render_template('form.html')

# Route for the second form (Keterangan)
@app.route('/keterangan', methods=['POST', 'GET'])
def keterangan():
    if request.method == 'POST':
        ac = request.form['ac']
        tv = request.form['tv']
        flat = request.form['flat']
        area_urban = request.form['area_urban']
        
        # Logic for handling the submitted data (e.g., making predictions)
        result = make_prediction(ac, tv, flat, area_urban)
        return render_template('result.html', result=result)
    
    return render_template('keterangan.html')

# Logic for prediction (example logic)
def make_prediction(ac, tv, flat, area_urban):
    # Example logic for prediction
    prediction = "Predicted Result: Example Category"
    return prediction

if __name__ == '__main__':
    app.run(debug=True)
