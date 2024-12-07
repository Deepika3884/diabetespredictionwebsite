# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pickle

# Step 1: Load the Dataset
dataset = pd.read_csv('diabetes.csv')

# Step 2: Data Preprocessing
# Selecting relevant features
dataset_X = dataset.iloc[:, [1, 4, 5, 7]].values  # Glucose, Skin Thickness, Insulin, Age
dataset_Y = dataset.iloc[:, 8].values  # Outcome

# Scaling features
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset_X)

# Splitting dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    dataset_scaled, dataset_Y, test_size=0.20, random_state=42, stratify=dataset_Y
)

# Step 3: Model Training
# Train an SVM model
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, Y_train)

# Step 4: Evaluate the Model
accuracy = svc.score(X_test, Y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 5: Save the Model and Scaler
pickle.dump(svc, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("Model and Scaler saved successfully!")

# Step 6: Test the Model with Sample Input
# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Test Prediction (example input: Glucose=86, Skin Thickness=66, BMI=26.6, Age=31)
test_input = np.array([[86, 66, 26.6, 31]])
scaled_input = scaler.transform(test_input)
prediction = model.predict(scaled_input)
print("Test Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
