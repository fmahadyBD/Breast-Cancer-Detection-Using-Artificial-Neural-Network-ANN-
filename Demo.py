# breast_cancer_ann.py

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC, Precision, Recall

# 1. Load the Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. Build ANN Model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(30,)))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer

# 5. Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()])

# 6. Train the Model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# 7. Evaluate on Test Data
loss, accuracy, auc, precision, recall = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
