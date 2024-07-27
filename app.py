# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
with open('rf.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the model and encoders
Car_Name_le = pickle.load(open('Car_Name_le.pkl', 'rb'))
Fuel_Type_le = pickle.load(open('Fuel_Type_le.pkl', 'rb'))
Seller_Type_le = pickle.load(open('Seller_Type_le.pkl', 'rb'))
Transmission_le = pickle.load(open('Transmission_le.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.form.to_dict()
    data = {
        'Year': [int(data['Year'])],
        'Present_Price': [float(data['Present_Price'])],
        'Kms_Driven': [int(data['Kms_Driven'])],
        'Car_Name': [data['Car_Name']],
        'Fuel_Type': [data['Fuel_Type']],
        'Seller_Type': [data['Seller_Type']],
        'Transmission': [data['Transmission']]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Apply label encoding
    df['Car_Name'] = Car_Name_le.transform(df['Car_Name'])
    df['Fuel_Type'] = Fuel_Type_le.transform(df['Fuel_Type'])
    df['Seller_Type'] = Seller_Type_le.transform(df['Seller_Type'])
    df['Transmission'] = Transmission_le.transform(df['Transmission'])

    # Scale the features
    scaled_data = scaler.transform(df[['Car_Name','Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission']])
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    # Return the result
    # return jsonify({'Estimated Selling Price': float(prediction[0])})
    return render_template('index.html', prediction_text=f'Estimated Selling Price: {prediction[0]:.2f} lakhs')


if __name__ == "__main__":
    app.run(debug=True)
