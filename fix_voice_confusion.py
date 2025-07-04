#!/usr/bin/env python3
"""
Targeted Voice Recognition Improvements
Focused on fixing the Jai Singh vs Nandalal confusion
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

def fix_voice_confusion():
    """Fix the Jai Singh vs Nandalal voice confusion"""
    
    print("=" * 80)
    print("TARGETED VOICE RECOGNITION IMPROVEMENT")
    print("Addressing Jai Singh vs Nandalal confusion")
    print("=" * 80)
    
    # 1. Enhanced feature extraction to better distinguish voices
    print("\n1. Implementing enhanced feature extraction...")
    
    # 2. Data balancing and augmentation
    print("2. Creating balanced training data...")
    
    # 3. Focused model training
    print("3. Training focused discrimination model...")
    
    # Initialize improved system
    system = ImprovedVoiceSystem()
    
    # Load data with enhanced features
    system.load_enhanced_voice_data()
    
    # Train with focus on problematic pair
    system.train_focused_model()
    
    # Test improvements
    results = system.test_improvements()
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT RESULTS")
    print("=" * 80)
    
    for person, result in results.items():
        print(f"{person}: {result['accuracy']:.1%} (was {result['before']:.1%})")
        if result['accuracy'] > result['before']:
            print(f"  ✅ Improved by {result['accuracy'] - result['before']:.1%}")
        else:
            print(f"  ⚠️ Still needs work")

class ImprovedVoiceSystem(FaceVoiceRecognitionSystem):
    """Enhanced voice system with focus on discrimination"""
    
    def __init__(self):
        super().__init__()
        self.enhanced_features = True
    
    def extract_discriminative_features(self, audio_data, sr=22050):
        """Extract features specifically designed to distinguish similar voices"""
        
        # Standard MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
        mfccs = mfccs[:, :130]
        
        # Enhanced chroma for pitch discrimination
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
        
        # Additional discriminative features
        additional_features = []
        
        # Spectral rolloff for voice character
        try:
            rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            rolloff = np.pad(rolloff, ((0, 0), (0, max(0, 130 - rolloff.shape[1]))), mode='constant')
            rolloff = rolloff[:, :130]
            additional_features.append(rolloff)
        except:
            pass
        
        # Zero crossing rate for voice activity patterns
        try:
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            zcr = np.pad(zcr, ((0, 0), (0, max(0, 130 - zcr.shape[1]))), mode='constant')
            zcr = zcr[:, :130]
            additional_features.append(zcr)
        except:
            pass
        
        # Spectral flatness for voice texture
        try:
            flatness = librosa.feature.spectral_flatness(y=audio_data)
            flatness = np.pad(flatness, ((0, 0), (0, max(0, 130 - flatness.shape[1]))), mode='constant')
            flatness = flatness[:, :130]
            additional_features.append(flatness)
        except:
            pass
        
        # Combine all features
        all_features = [mfccs, chroma, contrast] + additional_features
        combined_features = np.vstack(all_features)
        
        return combined_features
    
    def create_balanced_dataset(self, voice_dir="voice_data"):
        """Create balanced dataset with data augmentation for problematic voices"""
        
        voice_features = []
        labels = []
        
        for person_name in os.listdir(voice_dir):
            person_path = os.path.join(voice_dir, person_name)
            if os.path.isdir(person_path):
                print(f"Processing {person_name}...")
                person_samples = []
                
                for wav_file in os.listdir(person_path):
                    if wav_file.endswith('.wav'):
                        wav_path = os.path.join(person_path, wav_file)
                        try:
                            # Load audio
                            y, sr = librosa.load(wav_path, sr=22050, duration=3.0)
                            
                            # Extract enhanced features
                            features = self.extract_discriminative_features(y, sr)
                            person_samples.append((features, person_name))
                            
                            # For Jai Singh, add augmented samples to balance with Nandalal
                            if person_name.lower() == "jai singh":
                                # Add time-stretched versions
                                try:
                                    y_slow = librosa.effects.time_stretch(y, rate=0.95)
                                    features_slow = self.extract_discriminative_features(y_slow, sr)
                                    person_samples.append((features_slow, person_name))
                                    
                                    y_fast = librosa.effects.time_stretch(y, rate=1.05)
                                    features_fast = self.extract_discriminative_features(y_fast, sr)
                                    person_samples.append((features_fast, person_name))
                                except:
                                    pass
                                
                                # Add pitch-shifted versions
                                try:
                                    y_high = librosa.effects.pitch_shift(y, sr=sr, n_steps=0.5)
                                    features_high = self.extract_discriminative_features(y_high, sr)
                                    person_samples.append((features_high, person_name))
                                    
                                    y_low = librosa.effects.pitch_shift(y, sr=sr, n_steps=-0.5)
                                    features_low = self.extract_discriminative_features(y_low, sr)
                                    person_samples.append((features_low, person_name))
                                except:
                                    pass
                        
                        except Exception as e:
                            print(f"  Error processing {wav_file}: {e}")
                
                # Add all samples for this person
                for features, label in person_samples:
                    voice_features.append(features)
                    labels.append(label)
                
                print(f"  Total samples for {person_name}: {len(person_samples)}")
        
        return np.array(voice_features), labels
    
    def load_enhanced_voice_data(self):
        """Load voice data with enhanced features and balancing"""
        print("Loading enhanced voice data...")
        
        voice_features, labels = self.create_balanced_dataset()
        
        if len(voice_features) > 0:
            self.voice_data = voice_features
            
            # Enhanced normalization per feature type
            print("Applying enhanced normalization...")
            for i in range(len(self.voice_data)):
                sample = self.voice_data[i]
                
                # Get the number of feature types
                n_features = sample.shape[0]
                
                if n_features >= 32:  # Standard + additional features
                    # MFCC (0-12)
                    mfcc_part = sample[:13, :]
                    sample[:13, :] = (mfcc_part - np.mean(mfcc_part)) / (np.std(mfcc_part) + 1e-8)
                    
                    # Chroma (13-24)
                    chroma_part = sample[13:25, :]
                    sample[13:25, :] = (chroma_part - np.mean(chroma_part)) / (np.std(chroma_part) + 1e-8)
                    
                    # Contrast (25-31)
                    contrast_part = sample[25:32, :]
                    sample[25:32, :] = (contrast_part - np.mean(contrast_part)) / (np.std(contrast_part) + 1e-8)
                    
                    # Additional features (32+)
                    for feat_idx in range(32, n_features):
                        feat_part = sample[feat_idx:feat_idx+1, :]
                        sample[feat_idx:feat_idx+1, :] = (feat_part - np.mean(feat_part)) / (np.std(feat_part) + 1e-8)
                
                self.voice_data[i] = sample
            
            self.voice_labels = self.voice_encoder.fit_transform(labels)
            
            print(f"Loaded {len(self.voice_data)} voice samples for {len(set(labels))} people")
            print(f"Enhanced voice data shape: {self.voice_data.shape}")
            
            # Show sample distribution
            unique_labels, counts = np.unique(self.voice_labels, return_counts=True)
            for label_idx, count in zip(unique_labels, counts):
                person_name = self.voice_encoder.inverse_transform([label_idx])[0]
                print(f"  {person_name}: {count} samples")
    
    def create_focused_model(self, num_classes):
        """Create model specifically tuned for voice discrimination"""
        
        input_shape = self.voice_data.shape[1:]
        
        model = Sequential([
            Flatten(input_shape=input_shape),
            
            # Larger first layer to capture more patterns
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Focus layers for discrimination
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])
        
        # Use a lower learning rate for more stable training
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_focused_model(self):
        """Train model with focus on discrimination"""
        
        if len(self.voice_data) > 0:
            num_voice_classes = len(np.unique(self.voice_labels))
            
            if num_voice_classes > 1:
                print(f"Training focused model for {num_voice_classes} people...")
                
                self.voice_model = self.create_focused_model(num_voice_classes)
                
                # Prepare data
                voice_data_reshaped = self.voice_data.reshape(len(self.voice_data), *self.voice_data.shape[1:], 1)
                y_voice_categorical = to_categorical(self.voice_labels, num_voice_classes)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    voice_data_reshaped, y_voice_categorical,
                    test_size=0.2, random_state=42, stratify=self.voice_labels
                )
                
                # Enhanced training
                early_stopping = EarlyStopping(
                    monitor='val_loss', 
                    patience=15, 
                    restore_best_weights=True,
                    verbose=1
                )
                
                print(f"Training with {len(X_train)} samples...")
                
                history = self.voice_model.fit(
                    X_train, y_train,
                    epochs=100,  # More epochs with early stopping
                    batch_size=4,   # Small batch for stability
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    verbose=1
                )
                
                # Save enhanced model
                self.voice_model.save('focused_voice_model.h5')
                with open('focused_voice_encoder.pkl', 'wb') as f:
                    pickle.dump(self.voice_encoder, f)
                
                print("✅ Focused voice model trained!")
                
                # Test on training samples to verify learning
                print("\nTesting model understanding...")
                for person_idx, person_name in enumerate(self.voice_encoder.classes_):
                    person_mask = np.argmax(y_train, axis=1) == person_idx
                    if np.sum(person_mask) > 0:
                        person_sample = X_train[person_mask][0:1]  # Take first sample
                        prediction = self.voice_model.predict(person_sample, verbose=0)
                        predicted_class = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        predicted_name = self.voice_encoder.inverse_transform([predicted_class])[0]
                        is_correct = predicted_name == person_name
                        
                        status = "✅" if is_correct else "❌"
                        print(f"  {status} {person_name}: predicted as {predicted_name} ({confidence:.3f})")
    
    def test_improvements(self):
        """Test the improved voice recognition"""
        print("\nTesting improved voice recognition...")
        
        voice_dir = "voice_data"
        results = {}
        
        # Previous results from analysis
        previous_results = {
            'Athul': 1.0,
            'Benison': 1.0, 
            'Jai Singh': 0.0,  # The problem case
            'Nandalal': 1.0
        }
        
        for person_name in os.listdir(voice_dir):
            person_path = os.path.join(voice_dir, person_name)
            if os.path.isdir(person_path):
                person_correct = 0
                person_total = 0
                
                for wav_file in os.listdir(person_path):
                    if wav_file.endswith('.wav'):
                        wav_path = os.path.join(person_path, wav_file)
                        
                        try:
                            # Load and test
                            audio_data, sr = librosa.load(wav_path, sr=22050, duration=3.0)
                            
                            # Extract enhanced features
                            features = self.extract_discriminative_features(audio_data, sr)
                            features_reshaped = features.reshape(1, *features.shape, 1)
                            
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
                            print(f"  {status} {person_name}/{wav_file}: {predicted_name} ({confidence:.3f})")
                            
                        except Exception as e:
                            print(f"  ❌ Error testing {wav_file}: {e}")
                
                accuracy = person_correct / person_total if person_total > 0 else 0
                before_accuracy = previous_results.get(person_name, 0.0)
                
                results[person_name] = {
                    'accuracy': accuracy,
                    'before': before_accuracy,
                    'correct': person_correct,
                    'total': person_total
                }
        
        return results

if __name__ == "__main__":
    try:
        fix_voice_confusion()
    except Exception as e:
        print(f"❌ Improvement failed: {e}")
        import traceback
        traceback.print_exc()
