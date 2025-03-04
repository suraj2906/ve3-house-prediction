from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('rf_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from form
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        house_age = float(request.form['house_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bedrooms = float(request.form['total_bedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        median_income = float(request.form['median_income'])

        input_data = {
                'longitude': float(request.form['longitude']),
                'latitude': float(request.form['latitude']),
                'house_age': float(request.form['house_age']),
                'total_rooms': float(request.form['total_rooms']),
                'total_bedrooms': float(request.form['total_bedrooms']),
                'population': float(request.form['population']),
                'households': float(request.form['households']),
                'median_income': float(request.form['median_income'])
            }
            
        # Convert the input into a numpy array
        features_array = np.array([[longitude, latitude, house_age, total_rooms, total_bedrooms, population, households, median_income/10000]])

        # Predicting using the loaded model
        prediction = model.predict(features_array)

        
        return render_template('results.html', 
                               prediction=prediction[0],
                               **input_data)




@app.route('/')
def hello():
    return render_template("index.html")
    
if __name__=="__main__":
    app.run(debug=True)