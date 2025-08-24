# Simple Breast Cancer Detection with ANN
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. Load Data
print("Loading data...")
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels (0=malignant, 1=benign)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Malignant: {np.sum(y==0)}, Benign: {np.sum(y==1)}")

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create Simple Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(30,)),  # Hidden layer
    Dense(8, activation='relu'),                      # Hidden layer
    Dense(1, activation='sigmoid')                    # Output layer
])

# 5. Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train Model
print("Training model...")
history = model.fit(X_train_scaled, y_train, 
                   epochs=50, 
                   batch_size=32, 
                   validation_split=0.2, 
                   verbose=1)

# 7. Test Model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# 8. Make Prediction on New Sample
def predict_cancer(features):
    """Predict if cancer is benign (1) or malignant (0)"""
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0][0]
    
    if prediction > 0.5:
        result = "Benign"
        confidence = prediction
    else:
        result = "Malignant" 
        confidence = 1 - prediction
        
    return result, confidence

# 9. Test with Sample
print("\n--- Testing Prediction ---")
# Use first test sample
sample = X_test[0]
actual = "Benign" if y_test[0] == 1 else "Malignant"

prediction, confidence = predict_cancer(sample)
print(f"Actual: {actual}")
print(f"Predicted: {prediction}")
print(f"Confidence: {confidence:.4f}")

# 10. Save Model
model.save('simple_cancer_model.h5')
print("\nModel saved as 'simple_cancer_model.h5'")

# 11. Plot Training History
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("\nDone! ✅")