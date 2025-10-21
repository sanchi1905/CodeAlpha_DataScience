
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open(r"C:\Users\Lenovo\OneDrive\Desktop\DataScience tasks\DataScience tasks\Unemployment Analysis\unemployment_model_v2.pkl", 'rb'))

# Define the feature columns (must be in the same order as the model was trained on)
feature_columns = ['Estimated Employed', 'Estimated Labour Participation Rate', 'longitude', 'latitude', 'Month', 'Year', 'Region_Assam', 'Region_Bihar', 'Region_Chhattisgarh', 'Region_Delhi', 'Region_Goa', 'Region_Gujarat', 'Region_Haryana', 'Region_Himachal Pradesh', 'Region_Jammu & Kashmir', 'Region_Jharkhand', 'Region_Karnataka', 'Region_Kerala', 'Region_Madhya Pradesh', 'Region_Maharashtra', 'Region_Meghalaya', 'Region_Odisha', 'Region_Puducherry', 'Region_Punjab', 'Region_Rajasthan', 'Region_Sikkim', 'Region_Tamil Nadu', 'Region_Telangana', 'Region_Tripura', 'Region_Uttar Pradesh', 'Region_Uttarakhand', 'Region_West Bengal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Create a dictionary with all feature columns initialized to 0
    features = {col: 0 for col in feature_columns}
    
    # Update the dictionary with the user's input
    features.update(data)
    
    # One-hot encode the region
    if 'Region' in data:
        region_feature = f"Region_{data['Region']}"
        if region_feature in features:
            features[region_feature] = 1

    # Create the feature array in the correct order
    feature_array = [features[col] for col in feature_columns]
    
    # Make prediction
    prediction = model.predict(np.array([feature_array]))
    
    return jsonify(prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
