# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

# Step 2: Load your data
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
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

# =============================
# Step 12: Precision–Recall Curve
# =============================
y_prob = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, color='green', linewidth=2, label=f'AP = {avg_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

print(f"Average Precision (AP): {avg_precision:.3f}")

# =============================
# Step 13: Classification Report as DataFrame
# =============================
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nClassification Report Table:")
print(report_df)

report_df[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar', figsize=(8,5))
plt.title('Precision, Recall, and F1-Score per Class')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# =============================
# Step 14: Learning Curve
# =============================
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy',
    n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-Validation Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
