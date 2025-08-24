# Predict New Cancer Case - Simple Version with CSV Support
import numpy as np
import pandas as pd  # ADD THIS LINE
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



# ADD THIS NEW FUNCTION - CSV LOADER
def predict_from_csv(csv_filename):
    """Load patients from CSV and predict all of them"""
    try:
        # Load CSV file
        print(f"Loading data from {csv_filename}...")
        df = pd.read_csv(csv_filename)
        print(f"Found {len(df)} patients in CSV file")
        
        # Check if we have enough columns (need 30 features)
        if df.shape[1] < 30:
            print(f"Error: Need 30 feature columns, but CSV has only {df.shape[1]} columns")
            return
        
        results = []
        
        # Process each patient (each row)
        for index, row in df.iterrows():
            # Get patient name if available, otherwise use row number
            if 'patient_name' in df.columns:
                patient_name = row['patient_name']
                # Get the 30 features (skip the name column)
                features = row.drop('patient_name').values[:30]
            else:
                patient_name = f"Patient_{index+1}"
                # Get first 30 columns as features
                features = row.values[:30]
            
            # Make prediction
            result, confidence = predict_new_patient(features)
            
            # Store result
            results.append({
                'Patient': patient_name,
                'Diagnosis': result,
                'Confidence': f"{confidence:.1f}%"
            })
            
            # Print result
            print(f"{patient_name}: {result} (Confidence: {confidence:.1f}%)")
        
        return results
        
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found!")
        print("Please make sure the CSV file is in the same folder as this script.")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None



# MAIN EXECUTION - CHECK FOR CSV FIRST
print("Breast Cancer Prediction")

csv_file = "data.csv"


# Check if CSV file exists
import os
if os.path.exists(csv_file):
    print(f"CSV file '{csv_file}' found! Loading patients...")
    predict_from_csv(csv_file)
else:
    print(f"CSV file '{csv_file}' not found.")
  




