
import pickle

# Load the trained model
with open(r"C:\Users\Lenovo\OneDrive\Desktop\DataScience tasks\DataScience tasks\Sales Prediction\sales_prediction_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Get the coefficients and intercept
coeffs = model.coef_
intercept = model.intercept_

print(f"Coefficients: {coeffs}")
print(f"Intercept: {intercept}")
