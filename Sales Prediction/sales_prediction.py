import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset
df = pd.read_csv("C:\\Users\\Lenovo\\OneDrive\\Desktop\\DataScience tasks\\DataScience tasks\\Sales Prediction\\Advertising.csv")

# Split the data into features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the model
with open("C:\\Users\\Lenovo\\OneDrive\\Desktop\\DataScience tasks\\DataScience tasks\\Sales Prediction\\sales_prediction_model.pkl", 'wb') as f:
    pickle.dump(model, f)

print("Model training complete and saved to sales_prediction_model.pkl")
