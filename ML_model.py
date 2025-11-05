# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the data
df = pd.read_csv('city_day.csv')

# Step 3: Explore your data
print(df.head())
print(df.columns)
print(df.isnull().sum())
print(df.dtypes)

# Step 4: Handle missing values
df = df.dropna(subset=['PM2.5', 'PM10', 'NO', 'NO2', 'CO', 'AQI_Bucket'])

# Step 5: Define features (X) and target (y)
X = df[['PM2.5', 'PM10', 'NO', 'NO2', 'CO']]
y = df['AQI_Bucket']

# Convert target to binary classification: Good (0) vs Poor (1)
y = y.apply(lambda x: 1 if x in ['Poor', 'Very Poor', 'Severe'] else 0)

# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Step 7: Initialize the model
model = LogisticRegression(max_iter=1000)

# Step 8: Train the model
model.fit(X_train, y_train)
print("Model trained successfully!")

# Step 9: Make predictions
y_pred = model.predict(X_test)
print("Predictions made!")

# Step 10: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 11: Predict on new data (optional)
new_data = pd.DataFrame({
    'PM2.5': [120],
    'PM10': [180],
    'NO': [30],
    'NO2': [45],
    'CO': [1.2]
})
prediction = model.predict(new_data)
print(f"Prediction for new data (1=Poor, 0=Good): {prediction[0]}")
