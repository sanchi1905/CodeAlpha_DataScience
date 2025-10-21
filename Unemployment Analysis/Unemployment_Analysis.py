import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\Lenovo\\OneDrive\\Desktop\\DataScience tasks\\DataScience tasks\\Unemployment Analysis\\Unemployment_Rate_upto_11_2020.csv')

print(df.head())
print(df.info())
print(df.describe())

# Clean column names
df.columns = ['Region', 'Date', 'Frequency', 'Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate', 'Region.Name', 'longitude', 'latitude']

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Clean Frequency column
df['Frequency'] = df['Frequency'].str.strip()

print(df.head())
print(df.info())

# Visualize unemployment rate over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate', data=df)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate')
plt.savefig('unemployment_rate_over_time.png')

# Visualize unemployment rate by state
plt.figure(figsize=(12, 8))
sns.boxplot(x='Region', y='Estimated Unemployment Rate', data=df)
plt.xticks(rotation=90)
plt.title('Unemployment Rate by State')
plt.xlabel('State')
plt.ylabel('Estimated Unemployment Rate')
plt.savefig('unemployment_rate_by_state.png')
