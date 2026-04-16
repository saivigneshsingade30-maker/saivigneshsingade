import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Dataset (Standard Pima Indians Dataset)
# URL to the raw dataset on GitHub
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# 2. Data Preprocessing
# Handling zeros: In this dataset, 0 in 'Glucose' or 'BMI' usually means missing data.
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    data[col] = data[col].replace(0, np.nan)
    data[col] = data[col].fillna(data[col].median())

# 3. Split Data into Features (X) and Target (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Standardize the features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Prediction and Evaluation
y_pred = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Sample Prediction for a New Patient
# [Pregnancies, Glucose, BP, Skin, Insulin, BMI, Pedigree, Age]
new_patient = np.array([[2, 120, 70, 20, 80, 25.5, 0.6, 30]])
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)

print("--- Result ---")
print("Diabetic" if prediction[0] == 1 else "Non-Diabetic")