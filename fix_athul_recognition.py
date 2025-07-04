#!/usr/bin/env python3
"""
Focused script to diagnose and fix Athul's voice recognition issues.
"""

import numpy as np
import librosa
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("🔍 DIAGNOSING ATHUL'S VOICE RECOGNITION ISSUES")
print("=" * 80)

def extract_enhanced_features(file_path, max_length=32):
    """Extract enhanced features with better error handling."""
    try:
        # Load audio with higher sample rate for better quality
        y, sr = librosa.load(file_path, sr=22050, duration=3.0)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Extract MFCC features (mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        
        # Extract additional features for better discrimination
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        
        # Extract pitch-related features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512)
        pitch_mean = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
        
        # Extract zero crossing rate (voice activity)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)
        
        # Combine all features
        features = np.vstack([
            mfcc,           # 13 features
            chroma,         # 12 features  
            spectral_contrast,  # 7 features
            zcr,            # 1 feature
            spectral_centroids,  # 1 feature
            spectral_rolloff     # 1 feature
        ])  # Total: 35 features
        
        # Pad or truncate to fixed length
        if features.shape[1] < max_length:
            features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant')
        else:
            features = features[:, :max_length]
        
        # Add pitch as a separate feature
        pitch_features = np.full((1, max_length), pitch_mean)
        features = np.vstack([features, pitch_features])  # Total: 36 features
        
        print(f"  ✅ Extracted features from {os.path.basename(file_path)}: {features.shape}")
        print(f"    Pitch mean: {pitch_mean:.2f}")
        print(f"    MFCC mean: {np.mean(mfcc):.4f}, std: {np.std(mfcc):.4f}")
        print(f"    Chroma mean: {np.mean(chroma):.4f}, std: {np.std(chroma):.4f}")
        print(f"    Spectral contrast mean: {np.mean(spectral_contrast):.4f}")
        
        return features.T  # Return as (time_steps, features)
        
    except Exception as e:
        print(f"  ❌ Error extracting features from {file_path}: {e}")
        return np.zeros((max_length, 36))

def analyze_voice_characteristics():
    """Analyze voice characteristics for each person, especially Athul."""
    print("\n📊 ANALYZING VOICE CHARACTERISTICS")
    print("-" * 50)
    
    voice_data_dir = "voice_data"
    characteristics = {}
    
    for person in ["Athul", "Benison", "Jai Singh", "Nandalal"]:
        person_dir = os.path.join(voice_data_dir, person)
        if not os.path.exists(person_dir):
            continue
            
        print(f"\nAnalyzing {person}:")
        person_features = []
        
        for i, audio_file in enumerate(os.listdir(person_dir), 1):
            if audio_file.endswith('.wav'):
                file_path = os.path.join(person_dir, audio_file)
                features = extract_enhanced_features(file_path)
                person_features.append(features)
                
        if person_features:
            person_features = np.array(person_features)
            characteristics[person] = {
                'features': person_features,
                'mean': np.mean(person_features, axis=(0, 1)),
                'std': np.std(person_features, axis=(0, 1)),
                'pitch_mean': np.mean(person_features[:, :, -1]),  # Last feature is pitch
                'pitch_std': np.std(person_features[:, :, -1]),
                'mfcc_mean': np.mean(person_features[:, :, :13]),  # First 13 are MFCC
                'mfcc_std': np.std(person_features[:, :, :13])
            }
            
            print(f"  Features shape: {person_features.shape}")
            print(f"  Pitch characteristics: {characteristics[person]['pitch_mean']:.2f} ± {characteristics[person]['pitch_std']:.2f}")
            print(f"  MFCC characteristics: {characteristics[person]['mfcc_mean']:.4f} ± {characteristics[person]['mfcc_std']:.4f}")
    
    return characteristics

def load_balanced_voice_data_with_focus():
    """Load voice data with special focus on Athul's samples."""
    print("\n🎵 LOADING VOICE DATA WITH ATHUL FOCUS")
    print("-" * 50)
    
    voice_data_dir = "voice_data"
    X, y = [], []
    people = ["Athul", "Benison", "Jai Singh", "Nandalal"]
    
    for person in people:
        person_dir = os.path.join(voice_data_dir, person)
        if not os.path.exists(person_dir):
            print(f"⚠️ Directory not found: {person_dir}")
            continue
            
        print(f"\nProcessing {person}...")
        person_samples = 0
        
        for audio_file in os.listdir(person_dir):
            if audio_file.endswith('.wav'):
                file_path = os.path.join(person_dir, audio_file)
                features = extract_enhanced_features(file_path)
                
                if features is not None and features.shape == (32, 36):
                    X.append(features)
                    y.append(person)
                    person_samples += 1
                    
                    # Add extra augmented samples for Athul to help the model learn
                    if person == "Athul":
                        # Add noise-augmented version
                        noise = np.random.normal(0, 0.01, features.shape)
                        X.append(features + noise)
                        y.append(person)
                        
                        # Add pitch-shifted version (simulate slight variation)
                        pitch_shifted = features.copy()
                        pitch_shifted[:, -1] *= 1.05  # Slightly shift pitch
                        X.append(pitch_shifted)
                        y.append(person)
                        
                        person_samples += 2
                        print(f"  Added 2 augmented samples for Athul")
        
        print(f"  Total samples for {person}: {person_samples}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nLoaded {len(X)} total voice samples for {len(set(y))} people")
    print(f"Voice data shape: {X.shape}")
    
    # Print distribution
    for person in people:
        count = np.sum(y == person)
        print(f"  {person}: {count} samples")
    
    return X, y

def train_athul_focused_model():
    """Train a model with special focus on Athul recognition."""
    print("\n🧠 TRAINING ATHUL-FOCUSED MODEL")
    print("-" * 50)
    
    # Load data
    X, y = load_balanced_voice_data_with_focus()
    
    if len(X) == 0:
        print("❌ No voice data loaded!")
        return None, None
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Normalize features per feature type for better learning
    X_normalized = X.copy()
    
    # Normalize MFCC features (0-12)
    mfcc_scaler = StandardScaler()
    X_normalized[:, :, :13] = mfcc_scaler.fit_transform(X_normalized[:, :, :13].reshape(-1, 13)).reshape(X_normalized.shape[0], X_normalized.shape[1], 13)
    
    # Normalize chroma features (13-24)
    chroma_scaler = StandardScaler()
    X_normalized[:, :, 13:25] = chroma_scaler.fit_transform(X_normalized[:, :, 13:25].reshape(-1, 12)).reshape(X_normalized.shape[0], X_normalized.shape[1], 12)
    
    # Normalize other features (25-35)
    other_scaler = StandardScaler()
    X_normalized[:, :, 25:] = other_scaler.fit_transform(X_normalized[:, :, 25:].reshape(-1, 11)).reshape(X_normalized.shape[0], X_normalized.shape[1], 11)
    
    print("Applied per-feature-type normalization")
    
    # Reshape for CNN input
    X_reshaped = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], X_normalized.shape[2], 1)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_categorical, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    
    # Build enhanced model architecture
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(32, 36, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Enhanced callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=15, 
        restore_best_weights=True,
        mode='max'
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=8, 
        min_lr=1e-7
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=8,  # Smaller batch size for better learning
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Get best validation accuracy
    best_val_acc = max(history.history['val_accuracy'])
    print(f"✅ Athul-focused model trained!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    
    # Save the model and encoders
    model.save('voice_model_athul_focused.h5')
    
    with open('voice_encoder_athul_focused.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save scalers
    with open('voice_scalers_athul_focused.pkl', 'wb') as f:
        pickle.dump({
            'mfcc_scaler': mfcc_scaler,
            'chroma_scaler': chroma_scaler,
            'other_scaler': other_scaler
        }, f)
    
    print("Model and encoders saved!")
    
    return model, label_encoder

def test_athul_focused_model():
    """Test the Athul-focused model on all users."""
    print("\n🧪 TESTING ATHUL-FOCUSED MODEL")
    print("-" * 50)
    
    try:
        # Load model and encoders
        from tensorflow.keras.models import load_model
        model = load_model('voice_model_athul_focused.h5')
        
        with open('voice_encoder_athul_focused.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        with open('voice_scalers_athul_focused.pkl', 'rb') as f:
            scalers = pickle.load(f)
        
        voice_data_dir = "voice_data"
        results = {}
        total_correct = 0
        total_samples = 0
        
        for person in ["Athul", "Benison", "Jai Singh", "Nandalal"]:
            person_dir = os.path.join(voice_data_dir, person)
            if not os.path.exists(person_dir):
                continue
                
            print(f"\nTesting {person}:")
            correct = 0
            total = 0
            
            for audio_file in os.listdir(person_dir):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(person_dir, audio_file)
                    
                    # Extract features
                    features = extract_enhanced_features(file_path)
                    if features is None:
                        continue
                    
                    # Apply the same normalization as training
                    features_normalized = features.copy()
                    
                    # Normalize MFCC features (0-12)
                    features_normalized[:, :13] = scalers['mfcc_scaler'].transform(features_normalized[:, :13])
                    
                    # Normalize chroma features (13-24)
                    features_normalized[:, 13:25] = scalers['chroma_scaler'].transform(features_normalized[:, 13:25])
                    
                    # Normalize other features (25-35)
                    features_normalized[:, 25:] = scalers['other_scaler'].transform(features_normalized[:, 25:])
                    
                    # Reshape for prediction
                    features_reshaped = features_normalized.reshape(1, 32, 36, 1)
                    
                    # Predict
                    prediction = model.predict(features_reshaped, verbose=0)
                    predicted_class_idx = np.argmax(prediction[0])
                    predicted_person = label_encoder.inverse_transform([predicted_class_idx])[0]
                    confidence = prediction[0][predicted_class_idx]
                    
                    is_correct = predicted_person == person
                    status = "✅" if is_correct else "❌"
                    
                    print(f"  {status} {audio_file}: {predicted_person} ({confidence:.3f})")
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    # Show class probabilities for wrong predictions
                    if not is_correct:
                        print(f"    Class probabilities:")
                        for i, class_name in enumerate(label_encoder.classes_):
                            print(f"      {class_name}: {prediction[0][i]:.3f}")
            
            accuracy = (correct / total * 100) if total > 0 else 0
            results[person] = accuracy
            total_correct += correct
            total_samples += total
            
            print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        
        print("\n" + "=" * 80)
        print("ATHUL-FOCUSED MODEL RESULTS")
        print("=" * 80)
        print(f"Overall Accuracy: {overall_accuracy:.1f}%")
        
        for person, accuracy in results.items():
            status = "✅ Excellent" if accuracy >= 80 else "⚠️ Needs improvement" if accuracy >= 60 else "❌ Poor"
            print(f"{person}: {accuracy:.1f}% {status}")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    # Step 1: Analyze voice characteristics
    characteristics = analyze_voice_characteristics()
    
    # Step 2: Train Athul-focused model
    model, encoder = train_athul_focused_model()
    
    # Step 3: Test the model
    if model is not None:
        test_athul_focused_model()
    
    print("\n🎯 ATHUL DIAGNOSIS COMPLETE!")
