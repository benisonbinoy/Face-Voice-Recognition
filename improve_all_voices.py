#!/usr/bin/env python3
"""
Voice Recognition Improvement System
Implements specific improvements for each person based on analysis
"""

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from app import FaceVoiceRecognitionSystem
import pickle

def improve_voice_recognition():
    """Implement comprehensive improvements for voice recognition"""
    
    print("=" * 80)
    print("VOICE RECOGNITION IMPROVEMENT SYSTEM")
    print("=" * 80)
    
    # 1. Enhanced Feature Extraction with Person-Specific Optimization
    print("\n1. Implementing enhanced feature extraction...")
    
    # 2. Data Augmentation for Better Training
    print("2. Creating augmented training data...")
    
    # 3. Improved Model Architecture
    print("3. Optimizing model architecture...")
    
    # 4. Advanced Confidence Tuning
    print("4. Implementing adaptive confidence thresholds...")
    
    # 5. Cross-validation and Testing
    print("5. Running comprehensive testing...")
    
    # Initialize improved system
    system = ImprovedFaceVoiceRecognitionSystem()
    
    # Load and process data with improvements
    system.load_voice_data_with_augmentation()
    system.train_improved_models()
    
    # Test the improved system
    test_results = system.comprehensive_test()
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT RESULTS")
    print("=" * 80)
    
    for person, results in test_results.items():
        print(f"\n{person}:")
        print(f"  Before: {results['before_accuracy']:.1%}")
        print(f"  After:  {results['after_accuracy']:.1%}")
        print(f"  Improvement: {results['improvement']:.1%}")

class ImprovedFaceVoiceRecognitionSystem(FaceVoiceRecognitionSystem):
    """Enhanced version of the voice recognition system"""
    
    def __init__(self):
        super().__init__()
        self.person_specific_thresholds = {}
        self.augmented_data = []
        self.augmented_labels = []
    
    def extract_robust_features(self, audio_data, sr=22050):
        """Extract robust features with multiple techniques"""
        
        # Standard features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
        mfccs = mfccs[:, :130]
        
        # Enhanced chroma with harmonic analysis
        try:
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        except:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            chroma = np.repeat(spectral_centroid, 12, axis=0)
        
        chroma = np.pad(chroma, ((0, 0), (0, max(0, 130 - chroma.shape[1]))), mode='constant')
        chroma = chroma[:, :130]
        
        # Enhanced spectral features
        try:
            contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        except:
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
            contrast = np.repeat(spectral_bandwidth, 7, axis=0)
        
        contrast = np.pad(contrast, ((0, 0), (0, max(0, 130 - contrast.shape[1]))), mode='constant')
        contrast = contrast[:, :130]
        
        # Additional robust features
        try:
            # Spectral rolloff for voice timber
            rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            rolloff = np.pad(rolloff, ((0, 0), (0, max(0, 130 - rolloff.shape[1]))), mode='constant')
            rolloff = rolloff[:, :130]
            
            # Zero crossing rate for voice activity
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            zcr = np.pad(zcr, ((0, 0), (0, max(0, 130 - zcr.shape[1]))), mode='constant')
            zcr = zcr[:, :130]
            
            # Combine all features (37 features total)
            combined_features = np.vstack([mfccs, chroma, contrast, rolloff, zcr])
            
        except:
            # Fallback to original 32 features
            combined_features = np.vstack([mfccs, chroma, contrast])
        
        return combined_features
    
    def augment_voice_data(self, audio_data, sr=22050):
        """Create augmented versions of voice data"""
        augmented_samples = []
        
        # Original sample
        augmented_samples.append(audio_data)
        
        # Time stretching (slightly slower/faster)
        try:
            stretched_slow = librosa.effects.time_stretch(audio_data, rate=0.9)
            stretched_fast = librosa.effects.time_stretch(audio_data, rate=1.1)
            augmented_samples.extend([stretched_slow, stretched_fast])
        except:
            pass
        
        # Pitch shifting (slightly higher/lower)
        try:
            pitched_up = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=1)
            pitched_down = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=-1)
            augmented_samples.extend([pitched_up, pitched_down])
        except:
            pass
        
        # Add subtle noise for robustness
        try:
            noise_factor = 0.005
            noisy = audio_data + noise_factor * np.random.normal(0, 1, len(audio_data))
            augmented_samples.append(noisy)
        except:
            pass
        
        return augmented_samples
    
    def load_voice_data_with_augmentation(self, data_dir="voice_data"):
        """Load voice data with augmentation for better training"""
        print("Loading voice data with augmentation...")
        
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
                        # Load original audio
                        y, sr = librosa.load(wav_path, sr=22050, duration=3.0)
                        
                        # Create augmented versions
                        augmented_samples = self.augment_voice_data(y, sr)
                        
                        for i, aug_audio in enumerate(augmented_samples):
                            # Extract features
                            features = self.extract_robust_features(aug_audio, sr)
                            voice_features.append(features)
                            labels.append(person_name)
                            person_samples += 1
                            
                            if i == 0:  # Original sample
                                print(f"  Processed: {wav_file} -> shape: {features.shape}")
                            else:
                                print(f"    Augmented version {i} -> shape: {features.shape}")
                    
                    except Exception as e:
                        print(f"Error processing {wav_path}: {e}")
                
                print(f"  Total samples for {person_name}: {person_samples}")
        
        if voice_features:
            self.voice_data = np.array(voice_features)
            
            # Enhanced normalization
            print("Applying enhanced normalization...")
            for i in range(len(self.voice_data)):
                sample = self.voice_data[i]
                
                # Normalize different feature types
                if sample.shape[0] >= 32:  # Standard features
                    # MFCC normalization
                    mfcc_part = sample[:13, :]
                    sample[:13, :] = (mfcc_part - np.mean(mfcc_part)) / (np.std(mfcc_part) + 1e-8)
                    
                    # Chroma normalization
                    chroma_part = sample[13:25, :]
                    sample[13:25, :] = (chroma_part - np.mean(chroma_part)) / (np.std(chroma_part) + 1e-8)
                    
                    # Contrast normalization
                    contrast_part = sample[25:32, :]
                    sample[25:32, :] = (contrast_part - np.mean(contrast_part)) / (np.std(contrast_part) + 1e-8)
                    
                    # Additional features if present
                    if sample.shape[0] > 32:
                        for feat_idx in range(32, sample.shape[0]):
                            feat_part = sample[feat_idx:feat_idx+1, :]
                            sample[feat_idx:feat_idx+1, :] = (feat_part - np.mean(feat_part)) / (np.std(feat_part) + 1e-8)
                
                self.voice_data[i] = sample
            
            self.voice_labels = self.voice_encoder.fit_transform(labels)
            
            print(f"Loaded {len(self.voice_data)} voice samples (including augmented) for {len(set(labels))} people")
            print(f"Voice data shape: {self.voice_data.shape}")
    
    def create_enhanced_voice_model(self, num_classes):
        """Create enhanced model architecture"""
        
        input_shape = self.voice_data.shape[1:]  # Dynamic input shape
        
        model = Sequential([
            Flatten(input_shape=input_shape),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(0.4),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_improved_models(self):
        """Train models with improvements"""
        if len(self.voice_data) > 0:
            num_voice_classes = len(np.unique(self.voice_labels))
            
            if num_voice_classes > 1:
                print("Training enhanced voice recognition model...")
                
                self.voice_model = self.create_enhanced_voice_model(num_voice_classes)
                
                # Reshape for the model
                voice_data_reshaped = self.voice_data.reshape(len(self.voice_data), *self.voice_data.shape[1:], 1)
                y_voice_categorical = to_categorical(self.voice_labels, num_voice_classes)
                
                # Enhanced training with callbacks
                
                X_train, X_test, y_train, y_test = train_test_split(
                    voice_data_reshaped, y_voice_categorical, 
                    test_size=0.2, random_state=42, stratify=self.voice_labels
                )
                
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
                ]
                
                self.voice_model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=8,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Calculate person-specific thresholds
                self.calculate_person_thresholds(X_test, y_test)
                
                # Save enhanced model
                self.voice_model.save('enhanced_voice_model.h5')
                with open('enhanced_voice_encoder.pkl', 'wb') as f:
                    pickle.dump(self.voice_encoder, f)
                
                print("✅ Enhanced voice model trained successfully!")
    
    def calculate_person_thresholds(self, X_test, y_test):
        """Calculate optimal confidence thresholds for each person"""
        print("Calculating person-specific confidence thresholds...")
        
        predictions = self.voice_model.predict(X_test, verbose=0)
        
        for person_idx in range(len(self.voice_encoder.classes_)):
            person_name = self.voice_encoder.classes_[person_idx]
            
            # Find samples belonging to this person in test set
            person_mask = np.argmax(y_test, axis=1) == person_idx
            if np.sum(person_mask) > 0:
                person_predictions = predictions[person_mask]
                person_confidences = np.max(person_predictions, axis=1)
                
                # Set threshold as mean - 1 std for this person
                threshold = np.mean(person_confidences) - np.std(person_confidences)
                threshold = max(0.2, min(0.8, threshold))  # Clamp between 0.2 and 0.8
                
                self.person_specific_thresholds[person_name] = threshold
                print(f"  {person_name}: threshold = {threshold:.3f}")
    
    def recognize_voice_enhanced(self, audio_data):
        """Enhanced voice recognition with person-specific thresholds"""
        if self.voice_model is None:
            return {"name": "Error", "confidence": 0.0, "message": "No model loaded"}
        
        try:
            # Extract features using enhanced method
            features = self.extract_robust_features(audio_data)
            features = features.reshape(1, *features.shape, 1)
            
            # Get prediction
            prediction = self.voice_model.predict(features, verbose=0)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            
            person_name = self.voice_encoder.inverse_transform([predicted_class])[0]
            
            # Use person-specific threshold
            threshold = self.person_specific_thresholds.get(person_name, 0.4)
            
            if confidence > threshold:
                status = "success"
                message = f"✅ {person_name} (Enhanced: {confidence:.2%})"
            elif confidence > threshold * 0.7:
                status = "uncertain"
                message = f"⚠️ Possibly {person_name} (Low confidence: {confidence:.2%})"
            else:
                status = "unknown"
                message = f"❓ Unknown voice (Confidence: {confidence:.2%})"
            
            return {
                "name": person_name if confidence > threshold * 0.7 else "Unknown",
                "confidence": confidence,
                "status": status,
                "message": message
            }
            
        except Exception as e:
            return {"name": "Error", "confidence": 0.0, "message": f"Recognition error: {str(e)}"}
    
    def comprehensive_test(self):
        """Run comprehensive testing on all voice samples"""
        print("Running comprehensive testing...")
        
        voice_dir = "voice_data"
        results = {}
        
        for person_name in os.listdir(voice_dir):
            person_path = os.path.join(voice_dir, person_name)
            if os.path.isdir(person_path):
                person_correct = 0
                person_total = 0
                
                for wav_file in os.listdir(person_path):
                    if wav_file.endswith('.wav'):
                        wav_path = os.path.join(person_path, wav_file)
                        
                        try:
                            audio_data, sr = librosa.load(wav_path, sr=22050, duration=3.0)
                            result = self.recognize_voice_enhanced(audio_data)
                            
                            is_correct = result['name'].lower() == person_name.lower()
                            if is_correct:
                                person_correct += 1
                            person_total += 1
                            
                        except Exception as e:
                            print(f"Error testing {wav_file}: {e}")
                
                accuracy = person_correct / person_total if person_total > 0 else 0
                results[person_name] = {
                    'after_accuracy': accuracy,
                    'before_accuracy': 0.65,  # From previous test
                    'improvement': accuracy - 0.65
                }
        
        return results

if __name__ == "__main__":
    improve_voice_recognition()
