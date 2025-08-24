# 🏥 Breast Cancer Prediction with Neural Networks

A machine learning project that uses Artificial Neural Networks (ANN) to predict whether breast cancer tumors are benign or malignant. The system includes both model training and batch prediction capabilities with CSV file support.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Architecture](#model-architecture)
- [Data Format](#data-format)
- [Results](#results)
- [Contributing](#contributing)

## 🔍 Overview

This project implements a binary classification system using deep learning to distinguish between malignant and benign breast cancer cases. The model is trained on the Wisconsin Breast Cancer Dataset and can make predictions on new patient data either individually or in batch mode through CSV files.

## ✨ Features

- **🧠 Neural Network Model**: Simple yet effective ANN with 2 hidden layers
- **📊 High Accuracy**: Achieves reliable prediction accuracy on test data
- **📁 CSV Support**: Batch prediction for multiple patients from CSV files
- **⚡ Real-time Predictions**: Individual patient prediction capability
- **📈 Model Persistence**: Save and load trained models
- **🔧 Data Preprocessing**: Automatic feature scaling and normalization

## 🛠️ Requirements

```
numpy
pandas
tensorflow
scikit-learn
matplotlib
joblib
```

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/fmahadyBD/Breast-Cancer-Detection-Using-Artificial-Neural-Network-ANN-.git
cd breast-cancer-prediction
```

2. Install required packages:
```bash
pip install numpy pandas tensorflow scikit-learn matplotlib joblib
```

3. Run the training script:
```bash
python cancer_model_training.py
```

4. Use the prediction script:
```bash
python cancer_prediction.py
```

## 🚀 Usage

### Training the Model

The training script (`cancer_model_training.py`) will:
- Load the Wisconsin Breast Cancer Dataset
- Split data into training and testing sets
- Train the neural network
- Save the trained model as `simple_cancer_model.h5`
- Save the scaler as `scaler.pkl`

### Making Predictions

#### Option 1: CSV File Prediction (Recommended)
1. Prepare your CSV file with patient data (see format below)
2. Name it `data.csv` or modify the filename in the script
3. Run the prediction script - it will automatically detect and process the CSV

#### Option 2: Individual Prediction
Modify the script to use the `predict_new_patient()` function with a list of 30 feature values.

## 📁 File Structure

```
breast-cancer-prediction/
├── cancer_model_training.py    # Model training script
├── cancer_prediction.py       # Prediction script with CSV support
├── data.csv                   # Sample patient data (optional)
├── simple_cancer_model.h5     # Trained model (generated)
├── scaler.pkl                 # Feature scaler (generated)
└── README.md                  # This file
```

## 🏗️ Model Architecture

```
Input Layer (30 features) 
    ↓
Hidden Layer 1 (16 neurons, ReLU activation)
    ↓
Hidden Layer 2 (8 neurons, ReLU activation)
    ↓
Output Layer (1 neuron, Sigmoid activation)
```

**Training Parameters:**
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 50
- Batch Size: 32
- Validation Split: 20%

## 📊 Data Format

### Required CSV Format

Your CSV file should contain 31 columns (30 features + patient name):

```csv
patient_name,mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension
Patient_John,13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259
```

### Feature Categories

The 30 features are divided into three groups:
1. **Mean values** (10 features): Basic measurements
2. **Standard error** (10 features): Variability measurements  
3. **Worst values** (10 features): Largest/most extreme measurements

Each group contains measurements for:
- Radius, Texture, Perimeter, Area, Smoothness
- Compactness, Concavity, Concave points, Symmetry, Fractal dimension

## 📈 Results

### Model Performance
- **Test Accuracy**: ~95%+ (varies by run due to random initialization)
- **Prediction Speed**: Near-instantaneous for individual predictions
- **Batch Processing**: Efficient handling of multiple patients

### Sample Output
```
Patient_John: BENIGN (Not Cancer) (Confidence: 87.3%)
Patient_Mary: MALIGNANT (Cancer) (Confidence: 92.1%)
Patient_Ahmed: MALIGNANT (Cancer) (Confidence: 78.9%)
```

## ⚠️ Important Notes

- **Medical Disclaimer**: This is an educational project and should NOT be used for actual medical diagnosis
- **Feature Scaling**: All input features are automatically scaled using the saved scaler
- **Model Persistence**: The trained model and scaler are saved for reuse
- **Error Handling**: Built-in error checking for missing files and invalid data


## 📝 License

This project is intended for educational purposes. Please ensure compliance with relevant regulations when working with medical data.

## 🔗 References

- [Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

**Made with  for advancing healthcare through AI**