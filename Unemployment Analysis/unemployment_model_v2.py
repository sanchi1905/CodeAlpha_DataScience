import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset
df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\DataScience tasks\DataScience tasks\Unemployment Analysis\Unemployment_Rate_upto_11_2020.csv")

# Clean column names
df.columns = ['Region', 'Date', 'Frequency', 'Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate', 'Region.Name', 'longitude', 'latitude']

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# One-hot encode the 'Region' column
df = pd.get_dummies(df, columns=['Region'], drop_first=True)

# Print the columns
print(df.columns)

# Select features and target
features = ['Estimated Employed', 'Estimated Labour Participation Rate', 'longitude', 'latitude', 'Month', 'Year'] + [col for col in df.columns if 'Region_' in col]
target = 'Estimated Unemployment Rate'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"New Model - Mean Squared Error: {mse}")
print(f"New Model - R-squared: {r2}")

# Save the new model
with open(r"C:\Users\Lenovo\OneDrive\Desktop\DataScience tasks\DataScience tasks\Unemployment Analysis\unemployment_model_v2.pkl", 'wb') as f:
    pickle.dump(model, f)

print("New, improved model training complete and saved to unemployment_model_v2.pkl")