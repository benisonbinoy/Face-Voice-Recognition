#!/usr/bin/env python3
"""
Balanced Voice Recognition Improvement
Creating a properly balanced and robust voice recognition system
"""

import os
import sys
import numpy as np
import librosa
from app import FaceVoiceRecognitionSystem
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

def create_balanced_voice_system():
    """Create a balanced voice recognition system"""
    
    print("=" * 80)
    print("BALANCED VOICE RECOGNITION IMPROVEMENT")
    print("Creating properly balanced training with moderate augmentation")
    print("=" * 80)
    
    # Initialize balanced system
    system = BalancedVoiceSystem()
    
    # Load data with balanced augmentation
    system.load_balanced_voice_data()
    
    # Train with proper regularization
    system.train_balanced_model()
    
    # Test the balanced system
    results = system.test_balanced_system()
    
    print("\n" + "=" * 80)
    print("BALANCED SYSTEM RESULTS")
    print("=" * 80)
    
    overall_accuracy = np.mean([r['accuracy'] for r in results.values()])
    print(f"Overall Accuracy: {overall_accuracy:.1%}")
    
    for person, result in results.items():
        print(f"{person}: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        if result['accuracy'] >= 0.8:
            print(f"  ✅ Excellent performance")
        elif result['accuracy'] >= 0.6:
            print(f"  ✅ Good performance")
        else:
            print(f"  ⚠️ Needs improvement")

class BalancedVoiceSystem(FaceVoiceRecognitionSystem):
    """Balanced voice recognition system"""
    
    def __init__(self):
        super().__init__()
        self.balanced_features = True
    
    def extract_stable_features(self, audio_data, sr=22050):
        """Extract stable, reliable features for voice recognition"""
        
        # Core MFCC features (most reliable)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
        mfccs = mfccs[:, :130]
        
        # Chroma features for pitch characteristics
        try:
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        except:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            chroma = np.repeat(spectral_centroid, 12, axis=0)
        
        chroma = np.pad(chroma, ((0, 0), (0, max(0, 130 - chroma.shape[1]))), mode='constant')
        chroma = chroma[:, :130]
        
        # Spectral contrast for timbre
        try:
            contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        except:
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
            contrast = np.repeat(spectral_bandwidth, 7, axis=0)
        
        contrast = np.pad(contrast, ((0, 0), (0, max(0, 130 - contrast.shape[1]))), mode='constant')
        contrast = contrast[:, :130]
        
        # Just use these core 32 features for stability
        combined_features = np.vstack([mfccs, chroma, contrast])
        
        return combined_features
    
    def create_modest_augmentation(self, audio_data, sr=22050):
        """Create modest augmentation - only 1-2 extra samples per original"""
        augmented_samples = []
        
        # Always include original
        augmented_samples.append(audio_data)
        
        # Add one slightly modified version only
        try:
            # Very subtle time stretch
            stretched = librosa.effects.time_stretch(audio_data, rate=0.98)
            augmented_samples.append(stretched)
        except:
            # Fallback: add small amount of noise
            noise_factor = 0.003
            noisy = audio_data + noise_factor * np.random.normal(0, 1, len(audio_data))
            augmented_samples.append(noisy)
        
        return augmented_samples
    
    def load_balanced_voice_data(self):
        """Load voice data with balanced, modest augmentation"""
        print("Loading balanced voice data...")
        
        voice_features = []
        labels = []
        
        voice_dir = "voice_data"
        people_data = {}
        
        # First pass: collect all original samples
        for person_name in os.listdir(voice_dir):
            person_path = os.path.join(voice_dir, person_name)
            if os.path.isdir(person_path):
                people_data[person_name] = []
                
                for wav_file in os.listdir(person_path):
                    if wav_file.endswith('.wav'):
                        wav_path = os.path.join(person_path, wav_file)
                        try:
                            y, sr = librosa.load(wav_path, sr=22050, duration=3.0)
                            people_data[person_name].append(y)
                        except Exception as e:
                            print(f"Error loading {wav_path}: {e}")
        
        # Second pass: create balanced augmentation
        for person_name, audio_samples in people_data.items():
            print(f"Processing {person_name}...")
            person_sample_count = 0
            
            for audio_data in audio_samples:
                # Create modest augmentation (max 2 samples per original)
                augmented_samples = self.create_modest_augmentation(audio_data)
                
                for aug_audio in augmented_samples:
                    # Extract features
                    features = self.extract_stable_features(aug_audio)
                    voice_features.append(features)
                    labels.append(person_name)
                    person_sample_count += 1
            
            print(f"  Total samples for {person_name}: {person_sample_count}")
        
        if voice_features:
            self.voice_data = np.array(voice_features)
            
            # Apply consistent normalization
            print("Applying balanced normalization...")
            for i in range(len(self.voice_data)):
                sample = self.voice_data[i]
                
                # MFCC normalization
                mfcc_part = sample[:13, :]
                sample[:13, :] = (mfcc_part - np.mean(mfcc_part)) / (np.std(mfcc_part) + 1e-8)
                
                # Chroma normalization  
                chroma_part = sample[13:25, :]
                sample[13:25, :] = (chroma_part - np.mean(chroma_part)) / (np.std(chroma_part) + 1e-8)
                
                # Contrast normalization
                contrast_part = sample[25:32, :]
                sample[25:32, :] = (contrast_part - np.mean(contrast_part)) / (np.std(contrast_part) + 1e-8)
                
                self.voice_data[i] = sample
            
            self.voice_labels = self.voice_encoder.fit_transform(labels)
            
            print(f"Loaded {len(self.voice_data)} balanced voice samples for {len(set(labels))} people")
            print(f"Balanced voice data shape: {self.voice_data.shape}")
            
            # Show balanced distribution
            unique_labels, counts = np.unique(self.voice_labels, return_counts=True)
            for label_idx, count in zip(unique_labels, counts):
                person_name = self.voice_encoder.inverse_transform([label_idx])[0]
                print(f"  {person_name}: {count} samples")
    
    def create_balanced_model(self, num_classes):
        """Create a balanced model with proper regularization"""
        
        model = Sequential([
            Flatten(input_shape=(32, 130, 1)),
            
            # More conservative architecture
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),  # Higher dropout for regularization
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(16, activation='relu'),
            Dropout(0.4),
            
            Dense(num_classes, activation='softmax')
        ])
        
        # Conservative learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0003),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_balanced_model(self):
        """Train model with balanced approach"""
        
        if len(self.voice_data) > 0:
            num_voice_classes = len(np.unique(self.voice_labels))
            
            if num_voice_classes > 1:
                print(f"Training balanced model for {num_voice_classes} people...")
                
                self.voice_model = self.create_balanced_model(num_voice_classes)
                
                # Prepare data
                voice_data_reshaped = self.voice_data.reshape(len(self.voice_data), 32, 130, 1)
                y_voice_categorical = to_categorical(self.voice_labels, num_voice_classes)
                
                # Stratified split to ensure balanced representation
                X_train, X_test, y_train, y_test = train_test_split(
                    voice_data_reshaped, y_voice_categorical,
                    test_size=0.25, random_state=42, stratify=self.voice_labels
                )
                
                print(f"Training with {len(X_train)} samples, validating with {len(X_test)} samples")
                
                # Conservative training with multiple callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_accuracy',  # Monitor accuracy instead of loss
                        patience=20, 
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=10,
                        min_lr=0.00001,
                        verbose=1
                    )
                ]
                
                history = self.voice_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=8,   # Small batch size for stability
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Save balanced model
                self.voice_model.save('balanced_voice_model.h5')
                with open('balanced_voice_encoder.pkl', 'wb') as f:
                    pickle.dump(self.voice_encoder, f)
                
                print("✅ Balanced voice model trained!")
                
                # Final validation accuracy
                final_val_acc = max(history.history['val_accuracy'])
                print(f"Best validation accuracy: {final_val_acc:.3f}")
    
    def test_balanced_system(self):
        """Test the balanced voice recognition system"""
        print("\nTesting balanced voice recognition...")
        
        voice_dir = "voice_data"
        results = {}
        
        for person_name in os.listdir(voice_dir):
            person_path = os.path.join(voice_dir, person_name)
            if os.path.isdir(person_path):
                person_correct = 0
                person_total = 0
                
                print(f"\nTesting {person_name}:")
                
                for wav_file in os.listdir(person_path):
                    if wav_file.endswith('.wav'):
                        wav_path = os.path.join(person_path, wav_file)
                        
                        try:
                            # Load and test
                            audio_data, sr = librosa.load(wav_path, sr=22050, duration=3.0)
                            
                            # Extract features using same method as training
                            features = self.extract_stable_features(audio_data, sr)
                            features_reshaped = features.reshape(1, 32, 130, 1)
                            
                            # Predict
                            prediction = self.voice_model.predict(features_reshaped, verbose=0)
                            predicted_class = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            predicted_name = self.voice_encoder.inverse_transform([predicted_class])[0]
                            is_correct = predicted_name.lower() == person_name.lower()
                            
                            if is_correct:
                                person_correct += 1
                            person_total += 1
                            
                            status = "✅" if is_correct else "❌"
                            print(f"  {status} {wav_file}: {predicted_name} ({confidence:.3f})")
                            
                            # Show all class probabilities for debugging
                            print(f"    Class probabilities:")
                            for i, prob in enumerate(prediction[0]):
                                class_name = self.voice_encoder.inverse_transform([i])[0]
                                print(f"      {class_name}: {prob:.3f}")
                            
                        except Exception as e:
                            print(f"  ❌ Error testing {wav_file}: {e}")
                
                accuracy = person_correct / person_total if person_total > 0 else 0
                
                results[person_name] = {
                    'accuracy': accuracy,
                    'correct': person_correct,
                    'total': person_total
                }
        
        return results

if __name__ == "__main__":
    try:
        create_balanced_voice_system()
    except Exception as e:
        print(f"❌ System creation failed: {e}")
        import traceback
        traceback.print_exc()
