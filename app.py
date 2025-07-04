import os
import pickle
import numpy as np
import cv2
import librosa
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sounddevice as sd
import threading
import time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class FaceVoiceRecognitionSystem:
    def __init__(self):
        self.face_model = None
        self.voice_model = None
        self.face_encoder = LabelEncoder()
        self.voice_encoder = LabelEncoder()
        self.face_data = []
        self.voice_data = []
        self.face_labels = []
        self.voice_labels = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.voice_people_count = 0
        self.face_people_count = 0
        
    def load_face_data(self, data_dir="face_data"):
        """Load and preprocess face images using CNN with better preprocessing"""
        print("Loading face data...")
        face_images = []
        labels = []
        
        for person_name in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_name)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Detect faces and resize
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Apply histogram equalization to handle lighting variations
                        gray = cv2.equalizeHist(gray)
                        
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        for (x, y, w, h) in faces:
                            face = gray[y:y+h, x:x+w]
                            face = cv2.resize(face, (128, 128))
                            
                            # Apply Gaussian blur to reduce noise
                            face = cv2.GaussianBlur(face, (3, 3), 0)
                            
                            face = face.reshape(128, 128, 1)
                            face_images.append(face)
                            labels.append(person_name)
        
        if face_images:
            self.face_data = np.array(face_images, dtype='float32') / 255.0
            self.face_labels = self.face_encoder.fit_transform(labels)
            print(f"Loaded {len(self.face_data)} face images for {len(set(labels))} people")
            print(f"Face data shape: {self.face_data.shape}")
            print(f"Unique labels: {set(labels)}")
        
    def load_voice_data(self, data_dir="voice_data"):
        """Load and preprocess voice data with improved features"""
        print("Loading voice data...")
        voice_features = []
        labels = []
        
        for person_name in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_name)
            if os.path.isdir(person_path):
                print(f"Processing voice data for: {person_name}")
                person_samples = 0
                for wav_file in os.listdir(person_path):
                    wav_path = os.path.join(person_path, wav_file)
                    try:
                        # Extract robust audio features that work across librosa versions
                        y, sr = librosa.load(wav_path, sr=22050, duration=3.0)
                        
                        # Always start with MFCC (most reliable)
                        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
                        mfccs = mfccs[:, :130]
                        
                        # Try additional features with fallbacks
                        features_list = [mfccs]
                        
                        # Try chroma features
                        try:
                            if hasattr(librosa.feature, 'chroma_stft'):
                                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                            else:
                                # Fallback using spectral features
                                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                                chroma = np.repeat(spectral_centroid, 12, axis=0)  # Create 12-dim substitute
                        except:
                            # Last resort: use spectral centroid repeated
                            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                            chroma = np.repeat(spectral_centroid, 12, axis=0)
                        
                        chroma = np.pad(chroma, ((0, 0), (0, max(0, 130 - chroma.shape[1]))), mode='constant')
                        chroma = chroma[:, :130]
                        features_list.append(chroma)
                        
                        # Try spectral contrast
                        try:
                            if hasattr(librosa.feature, 'spectral_contrast'):
                                contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                            else:
                                # Fallback using spectral bandwidth
                                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                                contrast = np.repeat(spectral_bandwidth, 7, axis=0)  # Create 7-dim substitute
                        except:
                            # Last resort: use spectral bandwidth repeated
                            try:
                                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                                contrast = np.repeat(spectral_bandwidth, 7, axis=0)
                            except:
                                # Ultimate fallback: use zero crossing rate
                                zcr = librosa.feature.zero_crossing_rate(y)
                                contrast = np.repeat(zcr, 7, axis=0)
                        
                        contrast = np.pad(contrast, ((0, 0), (0, max(0, 130 - contrast.shape[1]))), mode='constant')
                        contrast = contrast[:, :130]
                        features_list.append(contrast)
                        
                        # Combine all features
                        combined_features = np.vstack(features_list)  # Should be (32, 130)
                        
                        voice_features.append(combined_features)
                        labels.append(person_name)
                        person_samples += 1
                        print(f"  Processed: {wav_file} -> shape: {combined_features.shape}")
                        
                    except Exception as e:
                        print(f"Error processing {wav_path}: {e}")
                
                print(f"  Total samples for {person_name}: {person_samples}")
        
        if voice_features:
            self.voice_data = np.array(voice_features)
            
            # Better normalization: standardize each feature type separately
            print("Applying feature-wise normalization...")
            for i in range(len(self.voice_data)):
                sample = self.voice_data[i]
                # Normalize MFCC (first 13 features)
                mfcc_part = sample[:13, :]
                sample[:13, :] = (mfcc_part - np.mean(mfcc_part)) / (np.std(mfcc_part) + 1e-8)
                
                # Normalize Chroma (next 12 features)
                chroma_part = sample[13:25, :]
                sample[13:25, :] = (chroma_part - np.mean(chroma_part)) / (np.std(chroma_part) + 1e-8)
                
                # Normalize Contrast (last 7 features)
                contrast_part = sample[25:32, :]
                sample[25:32, :] = (contrast_part - np.mean(contrast_part)) / (np.std(contrast_part) + 1e-8)
                
                self.voice_data[i] = sample
            
            self.voice_labels = self.voice_encoder.fit_transform(labels)
            
            print(f"Loaded {len(self.voice_data)} voice samples for {len(set(labels))} people")
            print(f"Voice data shape: {self.voice_data.shape}")
            print(f"Unique labels: {set(labels)}")
            print(f"Label encoder classes: {self.voice_encoder.classes_}")
            print(f"Label mapping: {dict(zip(self.voice_encoder.classes_, range(len(self.voice_encoder.classes_))))}")
            
            # Debug: Show distribution of labels
            unique_labels, counts = np.unique(self.voice_labels, return_counts=True)
            for label_idx, count in zip(unique_labels, counts):
                person_name = self.voice_encoder.inverse_transform([label_idx])[0]
                print(f"  {person_name} (label {label_idx}): {count} samples")
    
    def create_face_cnn_model(self, num_classes):
        """Create CNN model for face recognition"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def create_voice_cnn_model(self, num_classes):
        """Create model for voice recognition with combined features"""
        model = Sequential([
            Flatten(input_shape=(32, 130, 1)),  # Updated for combined features (MFCC + chroma + contrast)
            Dense(64, activation='relu'),  # Increased size for richer features
            Dropout(0.4),
            Dense(32, activation='relu'),
            Dropout(0.4),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def train_models(self):
        """Train both face and voice recognition models"""
        # Train face model
        if len(self.face_data) > 0:
            print("Training face recognition model...")
            num_face_classes = len(np.unique(self.face_labels))
            self.face_model = self.create_face_cnn_model(num_face_classes)
            
            y_face_categorical = to_categorical(self.face_labels, num_face_classes)
            X_train, X_test, y_train, y_test = train_test_split(
                self.face_data, y_face_categorical, test_size=0.2, random_state=42
            )
            
            self.face_model.fit(X_train, y_train, epochs=20, batch_size=32, 
                              validation_data=(X_test, y_test), verbose=1)
            
            # Save model
            self.face_model.save('face_model.h5')
            with open('face_encoder.pkl', 'wb') as f:
                pickle.dump(self.face_encoder, f)
          # Train voice model
        if len(self.voice_data) > 0:
            num_voice_classes = len(np.unique(self.voice_labels))
            
            if num_voice_classes > 1:
                print("Training voice recognition model...")
                print(f"Number of voice classes: {num_voice_classes}")
                print(f"Voice encoder classes: {self.voice_encoder.classes_}")
                
                self.voice_model = self.create_voice_cnn_model(num_voice_classes)
                
                voice_data_reshaped = self.voice_data.reshape(-1, 32, 130, 1)  # Updated for combined features
                y_voice_categorical = to_categorical(self.voice_labels, num_voice_classes)
                
                print(f"Voice data reshaped: {voice_data_reshaped.shape}")
                print(f"Voice labels categorical: {y_voice_categorical.shape}")
                
                # Show some examples of the mapping
                for i in range(min(5, len(self.voice_labels))):
                    original_label = self.voice_labels[i]
                    person_name = self.voice_encoder.inverse_transform([original_label])[0]
                    categorical_label = y_voice_categorical[i]
                    print(f"  Sample {i}: {person_name} -> label {original_label} -> categorical {categorical_label}")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    voice_data_reshaped, y_voice_categorical, test_size=0.2, random_state=42, stratify=self.voice_labels
                )
                
                # Early stopping to prevent overfitting
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                print(f"Training data shape: {X_train.shape}")
                print(f"Training labels shape: {y_train.shape}")
                print(f"Validation data shape: {X_test.shape}")
                
                history = self.voice_model.fit(X_train, y_train, epochs=20, batch_size=4,  # Increased epochs, smaller batch
                                   validation_data=(X_test, y_test), verbose=1,
                                   callbacks=[early_stopping])
                
                print(f"Training completed. Final training accuracy: {history.history['accuracy'][-1]:.4f}")
                print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
                
                # Test the model on a few training samples to verify it's working
                print("\nTesting model on training samples:")
                test_indices = [0, len(X_train)//2, len(X_train)-1]  # First, middle, last samples
                for idx in test_indices:
                    sample = X_train[idx:idx+1]
                    true_label = np.argmax(y_train[idx])
                    prediction = self.voice_model.predict(sample, verbose=0)
                    predicted_label = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    true_person = self.voice_encoder.inverse_transform([true_label])[0]
                    predicted_person = self.voice_encoder.inverse_transform([predicted_label])[0]
                    
                    print(f"  Sample {idx}: True={true_person} (label {true_label}), Predicted={predicted_person} (label {predicted_label}), Confidence={confidence:.4f}")
                
                # Save model
                self.voice_model.save('voice_model.h5')
                with open('voice_encoder.pkl', 'wb') as f:
                    pickle.dump(self.voice_encoder, f)
            else:
                print(f"✅ Voice data loaded for {num_voice_classes} person: {self.voice_encoder.classes_[0]}")
                print("   Voice recognition will work when you add more people.")
                print("   Creating simple voice template for future use...")
                
                # Create a simple template model that can be expanded later
                self.voice_model = self.create_voice_cnn_model(max(2, num_voice_classes))
                
                # Train with dummy data to prepare the model architecture
                voice_data_reshaped = self.voice_data.reshape(-1, 32, 130, 1)  # Updated for combined features
                  # Create temporary expanded dataset for training
                if num_voice_classes == 1:
                    # Add dummy "unknown" samples for training (same number as real samples)
                    num_samples = voice_data_reshaped.shape[0]
                    dummy_voice = np.random.normal(0, 0.1, voice_data_reshaped.shape)
                    expanded_voice_data = np.concatenate([voice_data_reshaped, dummy_voice])
                    expanded_labels = np.concatenate([self.voice_labels, np.ones(num_samples)])  # 1 for "unknown"
                    
                    y_voice_categorical = to_categorical(expanded_labels, 2)
                    
                    # Train with minimal epochs just to establish the model
                    self.voice_model.fit(expanded_voice_data, y_voice_categorical, 
                                       epochs=5, batch_size=16, verbose=1)
                
                # Save model and encoder for future expansion
                self.voice_model.save('voice_model.h5')
                with open('voice_encoder.pkl', 'wb') as f:
                    pickle.dump(self.voice_encoder, f)
                    
                print("   Voice model template saved. Add more people to enable full recognition.")
        else:
            print("⚠️  No voice data found for training.")
    
    def retrain_voice_model(self):
        """Retrain voice model when new voice data is added"""
        print("Retraining voice recognition model...")
        self.load_voice_data()
        
        if len(self.voice_data) > 0:
            num_voice_classes = len(np.unique(self.voice_labels))
            
            if num_voice_classes > 1:
                print(f"Training voice model with {num_voice_classes} people...")
                self.voice_model = self.create_voice_cnn_model(num_voice_classes)
                
                voice_data_reshaped = self.voice_data.reshape(-1, 32, 130, 1)  # Updated for combined features
                y_voice_categorical = to_categorical(self.voice_labels, num_voice_classes)
                
                if len(voice_data_reshaped) > 2:  # Need at least some data for train/test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        voice_data_reshaped, y_voice_categorical, test_size=0.2, random_state=42, stratify=self.voice_labels
                    )
                    
                    # Early stopping to prevent overfitting
                    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                    
                    self.voice_model.fit(X_train, y_train, epochs=10, batch_size=8,  # Reduced epochs and batch size
                                       validation_data=(X_test, y_test), verbose=1,
                                       callbacks=[early_stopping])
                else:
                    # Train on all data if we don't have enough for splitting
                    self.voice_model.fit(voice_data_reshaped, y_voice_categorical, 
                                       epochs=10, batch_size=16, verbose=1)  # Reduced epochs
                  # Save updated model
                self.voice_model.save('voice_model.h5')
                with open('voice_encoder.pkl', 'wb') as f:
                    pickle.dump(self.voice_encoder, f)
                    
                print("✅ Voice model retrained successfully!")
                return True
            else:
                print("Still only one person - voice recognition limited")
                return False
        else:
            print("No voice data found")
            return False
    
    def load_trained_models(self):
        """Load pre-trained models"""
        try:
            if os.path.exists('face_model.h5'):
                self.face_model = load_model('face_model.h5')
                with open('face_encoder.pkl', 'rb') as f:
                    self.face_encoder = pickle.load(f)
                print("Face model loaded successfully")
            
            if os.path.exists('voice_model.h5'):
                self.voice_model = load_model('voice_model.h5')
                with open('voice_encoder.pkl', 'rb') as f:
                    self.voice_encoder = pickle.load(f)
                print("Voice model loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def recognize_face(self, image):
        """Recognize face from image with better preprocessing for lighting variations"""
        if self.face_model is None:
            return "No face model loaded"
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to handle lighting variations
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        recognized_names = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128))
            
            # Apply Gaussian blur to reduce noise
            face = cv2.GaussianBlur(face, (3, 3), 0)
            
            face = face.reshape(1, 128, 128, 1)
            face = face.astype('float32') / 255.0
            
            prediction = self.face_model.predict(face, verbose=0)
            predicted_class = int(np.argmax(prediction))  # Convert to Python int
            confidence = float(np.max(prediction))  # Convert to Python float for consistency
            
            print(f"Face prediction: {prediction[0]}")
            print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
            
            # Lower confidence threshold since we improved preprocessing
            if confidence > 0.5:  # Lowered from 0.7
                name = self.face_encoder.inverse_transform([predicted_class])[0]
                recognized_names.append(f"{name} ({confidence:.2f})")
            else:
                recognized_names.append("Unknown")
        
        return recognized_names if recognized_names else ["No faces detected"]
    
    def recognize_voice(self, audio_data):
        """Enhanced dual-model voice recognition for maximum accuracy across all users"""
        if self.voice_model is None:
            return {
                "name": "Error",
                "confidence": 0.0,
                "mode": "error",
                "message": "Voice model not available (add more people to enable)",
                "status": "error"
            }
        
        try:
            # Use dual-model approach for better coverage
            predictions = []
            confidences = []
            
            # Model 1: Standard CNN model (good for most users)
            try:
                standard_result = self._predict_with_standard_model(audio_data)
                if standard_result:
                    predictions.append(standard_result)
                    print(f"Standard model: {standard_result['name']} ({standard_result['confidence']:.3f})")
            except Exception as e:
                print(f"Standard model failed: {e}")
            
            # Model 2: Try ensemble/specialized models if available
            try:
                ensemble_result = self._predict_with_ensemble_model(audio_data)
                if ensemble_result:
                    predictions.append(ensemble_result)
                    print(f"Ensemble model: {ensemble_result['name']} ({ensemble_result['confidence']:.3f})")
            except Exception as e:
                print(f"Ensemble model failed: {e}")
            
            # Model 3: Try Athul-focused model for Athul specifically
            try:
                athul_result = self._predict_with_athul_model(audio_data)
                if athul_result:
                    predictions.append(athul_result)
                    print(f"Athul-focused model: {athul_result['name']} ({athul_result['confidence']:.3f})")
            except Exception as e:
                print(f"Athul-focused model failed: {e}")
            
            # Combine predictions intelligently
            if not predictions:
                return {
                    "name": "Error", 
                    "confidence": 0.0,
                    "mode": "error",
                    "message": "All voice models failed",
                    "status": "error"
                }
            
            # Smart ensemble: give priority to high-confidence predictions
            best_prediction = max(predictions, key=lambda x: x['confidence'])
            
            # Check if multiple models agree
            names = [p['name'] for p in predictions]
            name_counts = {name: names.count(name) for name in set(names)}
            most_common_name = max(name_counts.keys(), key=lambda x: name_counts[x])
            
            # If multiple models agree and confidence is reasonable, boost confidence
            if name_counts[most_common_name] > 1 and best_prediction['confidence'] > 0.25:
                final_confidence = min(best_prediction['confidence'] * 1.2, 0.95)
                final_name = most_common_name
                status_msg = f"✅ {final_name} (Dual-model consensus - {final_confidence:.2%} confidence)"
                status = "success"
            elif best_prediction['confidence'] > 0.3:
                final_confidence = best_prediction['confidence']
                final_name = best_prediction['name']
                status_msg = f"✅ {final_name} (Single model - {final_confidence:.2%} confidence)"
                status = "success"
            elif best_prediction['confidence'] > 0.2:
                final_confidence = best_prediction['confidence']
                final_name = best_prediction['name']
                status_msg = f"⚠️ Possibly {final_name} (Low confidence: {final_confidence:.2%})"
                status = "uncertain"
            else:
                final_confidence = best_prediction['confidence']
                final_name = "Unknown"
                status_msg = f"❌ Voice not recognized (Very low confidence: {final_confidence:.2%})"
                status = "unknown"
            
            return {
                "name": final_name,
                "confidence": float(final_confidence),
                "mode": "dual_model",
                "message": status_msg,
                "status": status,
                "model_details": {
                    "predictions": predictions,
                    "consensus": most_common_name if len(predictions) > 1 else None
                }
            }
                    
        except Exception as e:
            return {
                "name": "Error",
                "confidence": 0.0,
                "mode": "error",
                "message": f"Voice recognition error: {str(e)}",
                "status": "error"
            }
    
    def _predict_with_standard_model(self, audio_data):
        """Predict using the standard CNN model"""
        try:
            # Extract standard features
            y = audio_data
            sr = 22050
            
            # Standard feature extraction
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
            mfccs = mfccs[:, :130]
            
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma = np.pad(chroma, ((0, 0), (0, max(0, 130 - chroma.shape[1]))), mode='constant')
            chroma = chroma[:, :130]
            
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_contrast = np.pad(spectral_contrast, ((0, 0), (0, max(0, 130 - spectral_contrast.shape[1]))), mode='constant')
            spectral_contrast = spectral_contrast[:, :130]
            
            # Combine features
            combined_features = np.vstack([mfccs, chroma, spectral_contrast])
            
            # Normalize
            mfcc_part = combined_features[:13, :]
            combined_features[:13, :] = (mfcc_part - np.mean(mfcc_part)) / (np.std(mfcc_part) + 1e-8)
            
            chroma_part = combined_features[13:25, :]
            combined_features[13:25, :] = (chroma_part - np.mean(chroma_part)) / (np.std(chroma_part) + 1e-8)
            
            contrast_part = combined_features[25:32, :]
            combined_features[25:32, :] = (contrast_part - np.mean(contrast_part)) / (np.std(contrast_part) + 1e-8)
            
            # Predict
            combined_features = combined_features.reshape(1, 32, 130, 1)
            prediction = self.voice_model.predict(combined_features, verbose=0)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            
            name = self.voice_encoder.inverse_transform([predicted_class])[0]
            
            return {
                "name": name,
                "confidence": confidence,
                "model": "standard_cnn"
            }
            
        except Exception as e:
            print(f"Standard model prediction failed: {e}")
            return None
    
    def _predict_with_ensemble_model(self, audio_data):
        """Predict using ensemble model if available"""
        try:
            if not os.path.exists('voice_ensemble_models.pkl'):
                return None
                
            with open('voice_ensemble_models.pkl', 'rb') as f:
                models = pickle.load(f)
            
            # Extract comprehensive features for ensemble
            y = librosa.util.normalize(audio_data)
            sr = 22050
            
            # Comprehensive feature extraction (same as ensemble training)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            pitches, _ = librosa.piptrack(y=y, sr=sr, hop_length=512)
            pitch_values = pitches[pitches > 0]
            pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
            pitch_median = np.median(pitch_values) if len(pitch_values) > 0 else 0
            
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)
            rms = librosa.feature.rms(y=y, hop_length=512)
            
            # Combine features
            features = np.vstack([
                mfcc, mfcc_delta, mfcc_delta2, chroma,
                spectral_centroids, spectral_contrast, spectral_rolloff,
                spectral_bandwidth, zcr, rms
            ])
            
            # Pad or truncate to 64 time steps
            if features.shape[1] < 64:
                features = np.pad(features, ((0, 0), (0, 64 - features.shape[1])), mode='constant')
            else:
                features = features[:, :64]
            
            # Statistical features
            stats = []
            for i in range(features.shape[0]):
                feat_row = features[i, :]
                stats.extend([
                    np.mean(feat_row), np.std(feat_row), np.min(feat_row),
                    np.max(feat_row), np.median(feat_row)
                ])
            
            stats.extend([pitch_mean, pitch_std, pitch_median])
            features_array = np.array(stats).reshape(1, -1)
            
            # Scale features
            features_scaled = models['scaler'].transform(features_array)
            
            # Get predictions from both models
            rf_proba = models['rf_model'].predict_proba(features_scaled)[0]
            svm_proba = models['svm_model'].predict_proba(features_scaled)[0]
            
            # Ensemble prediction
            ensemble_proba = 0.6 * rf_proba + 0.4 * svm_proba
            predicted_class_idx = np.argmax(ensemble_proba)
            confidence = float(ensemble_proba[predicted_class_idx])
            
            name = models['label_encoder'].inverse_transform([predicted_class_idx])[0]
            
            return {
                "name": name,
                "confidence": confidence,
                "model": "ensemble_rf_svm"
            }
            
        except Exception as e:
            print(f"Ensemble model prediction failed: {e}")
            return None
    
    def _predict_with_athul_model(self, audio_data):
        """Predict using Athul-focused model if available"""
        try:
            if not os.path.exists('voice_model_athul_focused.h5'):
                return None
                
            from tensorflow.keras.models import load_model
            athul_model = load_model('voice_model_athul_focused.h5')
            
            with open('voice_encoder_athul_focused.pkl', 'rb') as f:
                athul_encoder = pickle.load(f)
                
            with open('voice_scalers_athul_focused.pkl', 'rb') as f:
                scalers = pickle.load(f)
            
            # Extract features for Athul model
            y = librosa.util.normalize(audio_data)
            sr = 22050
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            pitches, _ = librosa.piptrack(y=y, sr=sr, hop_length=512)
            pitch_mean = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
            
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)
            
            # Combine features
            features = np.vstack([
                mfcc, chroma, spectral_contrast, zcr, spectral_centroids, spectral_rolloff
            ])
            
            # Pad or truncate to 32 time steps
            if features.shape[1] < 32:
                features = np.pad(features, ((0, 0), (0, 32 - features.shape[1])), mode='constant')
            else:
                features = features[:, :32]
            
            # Add pitch feature
            pitch_features = np.full((1, 32), pitch_mean)
            features = np.vstack([features, pitch_features])
            
            features = features.T  # (time_steps, features)
            
            # Apply normalization
            features_normalized = features.copy()
            features_normalized[:, :13] = scalers['mfcc_scaler'].transform(features_normalized[:, :13])
            features_normalized[:, 13:25] = scalers['chroma_scaler'].transform(features_normalized[:, 13:25])
            features_normalized[:, 25:] = scalers['other_scaler'].transform(features_normalized[:, 25:])
            
            # Reshape for CNN
            features_reshaped = features_normalized.reshape(1, 32, 36, 1)
            
            # Predict
            prediction = athul_model.predict(features_reshaped, verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            
            name = athul_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Give Athul predictions a slight boost since this model is specialized for him
            if name == "Athul":
                confidence = min(confidence * 1.1, 0.95)
            
            return {
                "name": name,
                "confidence": confidence,
                "model": "athul_focused_cnn"
            }
            
        except Exception as e:
            print(f"Athul-focused model prediction failed: {e}")
            return None
    
    def get_system_status(self):
        """Get current system status for both face and voice recognition"""
        # Count people in face_data
        face_people = []
        if os.path.exists("face_data"):
            face_people = [name for name in os.listdir("face_data") 
                          if os.path.isdir(os.path.join("face_data", name))]
        
        # Count people in voice_data  
        voice_people = []
        if os.path.exists("voice_data"):
            voice_people = [name for name in os.listdir("voice_data") 
                           if os.path.isdir(os.path.join("voice_data", name))]
        
        self.face_people_count = len(face_people)
        self.voice_people_count = len(voice_people)
        
        status = {
            "face_recognition": {
                "enabled": self.face_model is not None,
                "people_count": self.face_people_count,
                "people_names": face_people,
                "status": "ready" if self.face_model is not None else "needs_training"
            },
            "voice_recognition": {
                "enabled": self.voice_model is not None,
                "people_count": self.voice_people_count,
                "people_names": voice_people,
                "status": "full_recognition" if self.voice_people_count > 1 else "limited_recognition" if self.voice_people_count == 1 else "needs_training"
            }
        }
        
        return status
    
    def get_voice_data_info(self):
        """Get detailed information about voice data"""
        voice_info = {}
        if os.path.exists("voice_data"):
            for person_name in os.listdir("voice_data"):
                person_path = os.path.join("voice_data", person_name)
                if os.path.isdir(person_path):
                    files = [f for f in os.listdir(person_path) if f.endswith(('.wav', '.mp3', '.m4a'))]
                    voice_info[person_name] = {
                        "sample_count": len(files),
                        "files": files
                    }
        return voice_info

# Initialize the recognition system
recognition_system = FaceVoiceRecognitionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_models():
    """Train the models with current data"""
    try:
        recognition_system.load_face_data()
        recognition_system.load_voice_data()
        recognition_system.train_models()
        return jsonify({"status": "success", "message": "Models trained successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    """Recognize face from uploaded image"""
    try:
        data = request.json
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = recognition_system.recognize_face(image)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/recognize_voice', methods=['POST'])
def recognize_voice():
    """Recognize voice from uploaded audio"""
    try:
        audio_file = request.files['audio']
        audio_data, sr = librosa.load(audio_file, sr=16000)
        
        result = recognition_system.recognize_voice(audio_data)
        
        # Handle both old string format and new dict format
        if isinstance(result, dict):
            return jsonify({"status": "success", "result": result})
        else:
            # Backward compatibility for old string format
            return jsonify({"status": "success", "result": {"message": result}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/recognize_face_realtime', methods=['POST'])
def recognize_face_realtime():
    """Recognize face from camera frame in real-time"""
    try:
        data = request.json
        print("Received face recognition request...")  # Debug log
        
        # Handle base64 image data
        if 'image' not in data:
            return jsonify({"status": "error", "message": "No image data received"})
        
        img_data_str = data['image']
        if img_data_str.startswith('data:image'):
            img_data_str = img_data_str.split(',')[1]
        
        img_data = base64.b64decode(img_data_str)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"status": "error", "message": "Failed to decode image"})
        
        print(f"Image shape: {image.shape}")  # Debug log
        
        result = recognition_system.recognize_face(image)
        print(f"Recognition result: {result}")  # Debug log
        
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        print(f"Error in face recognition: {str(e)}")  # Debug log
        return jsonify({"status": "error", "message": str(e)})

@app.route('/retrain_voice', methods=['POST'])
def retrain_voice():
    """Retrain voice recognition model when new voice data is added"""
    try:
        success = recognition_system.retrain_voice_model()
        if success:
            return jsonify({"status": "success", "message": "Voice model retrained successfully"})
        else:
            return jsonify({"status": "info", "message": "Voice model updated but still needs more people"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_system_status', methods=['GET'])
def get_system_status():
    """Get current system status"""
    try:
        status = recognition_system.get_system_status()
        return jsonify({"status": "success", "data": status})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_voice_info', methods=['GET'])
def get_voice_info():
    """Get voice data information"""
    try:
        voice_info = recognition_system.get_voice_data_info()
        return jsonify({"status": "success", "data": voice_info})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Load existing models or train new ones
    recognition_system.load_trained_models()
    
    # If no models exist, load data and train
    if recognition_system.face_model is None or recognition_system.voice_model is None:
        print("No trained models found. Training new models...")
        recognition_system.load_face_data()
        recognition_system.load_voice_data()
        recognition_system.train_models()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
