# Predict New Cancer Case - Simple Version
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load trained model and scaler
try:
    model = load_model('simple_cancer_model.h5')
    scaler = joblib.load('scaler.pkl')
    print("Model loaded successfully!")
except:
    print("Please train the model first!")
    exit()

def predict_new_patient(features):
    """
    Predict cancer for new patient
    features: list of 30 numbers (patient measurements)
    """
    # Convert to numpy array
    features = np.array(features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0][0]
    
    # Interpret result
    if prediction > 0.5:
        result = "BENIGN (Not Cancer)"
        confidence = prediction * 100
    else:
        result = "MALIGNANT (Cancer)"
        confidence = (1 - prediction) * 100
    
    return result, confidence

# Example: New patient data (30 features)
new_patient = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 
    0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 
    0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 
    0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
]

# Make prediction
result, confidence = predict_new_patient(new_patient)

print("=== PREDICTION RESULT ===")
print(f"Diagnosis: {result}")
print(f"Confidence: {confidence:.1f}%")

# You can change the values above for different patients