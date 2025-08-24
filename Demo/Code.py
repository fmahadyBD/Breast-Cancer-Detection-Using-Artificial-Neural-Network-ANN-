# Comprehensive Breast Cancer Detection System using ANN
# This system includes both tabular data analysis and image processing capabilities

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import cv2
from PIL import Image
import os
import joblib

class BreastCancerDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.image_model = None
        
    def load_and_explore_data(self):
        """Step 1: Load and explore the breast cancer dataset"""
        print("=== STEP 1: DATA LOADING AND EXPLORATION ===")
        
        # Load the dataset
        data = load_breast_cancer()
        self.feature_names = data.feature_names
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {len(data.feature_names)}")
        print(f"Classes: {data.target_names}")
        print(f"\nClass distribution:")
        print(df['target'].value_counts())
        
        # Basic statistics
        print(f"\nDataset Info:")
        print(df.info())
        
        return df, data
    
    def visualize_data(self, df):
        """Step 2: Data visualization"""
        print("\n=== STEP 2: DATA VISUALIZATION ===")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution
        df['target'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Class Distribution')
        axes[0,0].set_xlabel('Class (0=Malignant, 1=Benign)')
        
        # Correlation heatmap (first 10 features)
        corr_matrix = df.iloc[:, :10].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[0,1])
        axes[0,1].set_title('Feature Correlation (First 10 features)')
        
        # Feature distribution
        df['mean radius'].hist(bins=30, ax=axes[1,0])
        axes[1,0].set_title('Mean Radius Distribution')
        
        # Box plot for a key feature
        df.boxplot(column='mean radius', by='target', ax=axes[1,1])
        axes[1,1].set_title('Mean Radius by Class')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, data):
        """Step 3: Data preprocessing"""
        print("\n=== STEP 3: DATA PREPROCESSING ===")
        
        X = data.data
        y = data.target
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Training set class distribution: {np.bincount(y_train)}")
        print(f"Test set class distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Step 4: Build the ANN model"""
        print("\n=== STEP 4: MODEL BUILDING ===")
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Model Architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Step 5: Train the model"""
        print("\n=== STEP 5: MODEL TRAINING ===")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_breast_cancer_model.h5', save_best_only=True, monitor='val_accuracy'
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(history.history['accuracy'], label='Training')
        axes[0,0].plot(history.history['val_accuracy'], label='Validation')
        axes[0,0].set_title('Model Accuracy')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        
        # Loss
        axes[0,1].plot(history.history['loss'], label='Training')
        axes[0,1].plot(history.history['val_loss'], label='Validation')
        axes[0,1].set_title('Model Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        
        # Precision
        axes[1,0].plot(history.history['precision'], label='Training')
        axes[1,0].plot(history.history['val_precision'], label='Validation')
        axes[1,0].set_title('Model Precision')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].legend()
        
        # Recall
        axes[1,1].plot(history.history['recall'], label='Training')
        axes[1,1].plot(history.history['val_recall'], label='Validation')
        axes[1,1].set_title('Model Recall')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """Step 6: Evaluate the model"""
        print("\n=== STEP 6: MODEL EVALUATION ===")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Evaluation metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Malignant', 'Benign'],
                   yticklabels=['Malignant', 'Benign'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
        return y_pred, y_pred_proba
    
    def save_model(self):
        """Save the trained model and scaler"""
        print("\n=== SAVING MODEL ===")
        self.model.save('breast_cancer_ann_model.h5')
        joblib.dump(self.scaler, 'breast_cancer_scaler.pkl')
        print("Model and scaler saved successfully!")
    
    def load_trained_model(self):
        """Load a pre-trained model"""
        try:
            self.model = load_model('breast_cancer_ann_model.h5')
            self.scaler = joblib.load('breast_cancer_scaler.pkl')
            print("Model loaded successfully!")
            return True
        except:
            print("No pre-trained model found!")
            return False
    
    def predict_new_sample(self, features):
        """Make prediction on new sample"""
        if self.model is None or self.scaler is None:
            print("Model not trained or loaded!")
            return None
        
        # Ensure features is 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale the features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction_proba = self.model.predict(features_scaled)
        prediction = (prediction_proba > 0.5).astype(int)
        
        result = {
            'prediction': 'Benign' if prediction[0][0] == 1 else 'Malignant',
            'confidence': float(prediction_proba[0][0]) if prediction[0][0] == 1 else float(1 - prediction_proba[0][0]),
            'probability': float(prediction_proba[0][0])
        }
        
        return result
    
    def build_image_model(self, input_shape=(224, 224, 3)):
        """Build CNN model for image classification"""
        print("\n=== BUILDING IMAGE MODEL ===")
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.image_model = model
        return model
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for prediction"""
        try:
            # Load and resize image
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize
            
            return img_array
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def predict_from_image(self, image_path):
        """Predict breast cancer from medical image"""
        if self.image_model is None:
            print("Image model not built or trained!")
            return None
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return None
        
        # Make prediction
        prediction_proba = self.image_model.predict(processed_image)
        prediction = (prediction_proba > 0.5).astype(int)
        
        result = {
            'prediction': 'Benign' if prediction[0][0] == 1 else 'Malignant',
            'confidence': float(prediction_proba[0][0]) if prediction[0][0] == 1 else float(1 - prediction_proba[0][0]),
            'probability': float(prediction_proba[0][0])
        }
        
        return result

def main():
    """Main function to run the complete pipeline"""
    print("🎯 Breast Cancer Detection System using Artificial Neural Networks")
    print("=" * 70)
    
    # Initialize detector
    detector = BreastCancerDetector()
    
    # Step 1: Load and explore data
    df, data = detector.load_and_explore_data()
    
    # Step 2: Visualize data
    detector.visualize_data(df)
    
    # Step 3: Preprocess data
    X_train, X_test, y_train, y_test = detector.preprocess_data(data)
    
    # Step 4: Build model
    detector.build_model(X_train.shape[1])
    
    # Step 5: Train model
    history = detector.train_model(X_train, X_test, y_train, y_test)
    
    # Step 6: Evaluate model
    y_pred, y_pred_proba = detector.evaluate_model(X_test, y_test)
    
    # Step 7: Save model
    detector.save_model()
    
    # Step 8: Test prediction on new sample
    print("\n=== STEP 7: TESTING NEW SAMPLE PREDICTION ===")
    
    # Create a sample from test data
    sample_idx = 0
    sample_features = X_test[sample_idx:sample_idx+1]
    actual_label = 'Benign' if y_test[sample_idx] == 1 else 'Malignant'
    
    # Convert back to original scale for demonstration
    sample_original = detector.scaler.inverse_transform(sample_features)
    
    # Make prediction
    result = detector.predict_new_sample(sample_original)
    
    print(f"Sample features (first 5): {sample_original[0][:5]}")
    print(f"Actual label: {actual_label}")
    print(f"Predicted: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probability: {result['probability']:.4f}")
    
    print("\n🎉 Training completed successfully!")
    print("You can now use the trained model to make predictions on new samples.")
    
    return detector

# Example usage for new predictions
def predict_new_case(detector, features_list):
    """
    Function to predict on new cases
    features_list: list of 30 feature values in the same order as the training data
    """
    features_array = np.array(features_list).reshape(1, -1)
    result = detector.predict_new_sample(features_array)
    return result

# Feature names for reference
FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

if __name__ == "__main__":
    # Run the complete pipeline
    trained_detector = main()
    
    # Example of how to use the trained model for new predictions
    print("\n" + "="*50)
    print("EXAMPLE: Using trained model for new predictions")
    print("="*50)
    
    # Example feature values (you would replace these with actual measurements)
    example_features = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
    
    prediction_result = predict_new_case(trained_detector, example_features)
    print(f"New case prediction: {prediction_result}")