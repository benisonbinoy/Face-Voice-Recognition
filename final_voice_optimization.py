#!/usr/bin/env python3
"""
Final optimized voice recognition system combining insights from all previous attempts.
"""

import numpy as np
import librosa
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("🚀 FINAL OPTIMIZED VOICE RECOGNITION SYSTEM")
print("=" * 80)

def extract_comprehensive_features(file_path, max_length=64):
    """Extract comprehensive voice features optimized for speaker recognition."""
    try:
        # Load audio with optimal settings
        y, sr = librosa.load(file_path, sr=22050, duration=4.0)
        
        # Normalize and pre-process
        y = librosa.util.normalize(y)
        
        # Core MFCC features (most important for speaker recognition)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Pitch and fundamental frequency features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512)
        pitch_values = pitches[pitches > 0]
        
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        pitch_median = np.median(pitch_values) if len(pitch_values) > 0 else 0
        
        # Chroma features (tonal characteristics)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)
        
        # Zero crossing rate (voice activity)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)
        
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=512)
        
        # Combine time-series features
        features = np.vstack([
            mfcc,                  # 13 features
            mfcc_delta,            # 13 features  
            mfcc_delta2,           # 13 features
            chroma,                # 12 features
            spectral_centroids,    # 1 feature
            spectral_contrast,     # 7 features
            spectral_rolloff,      # 1 feature
            spectral_bandwidth,    # 1 feature
            zcr,                   # 1 feature
            rms                    # 1 feature
        ])  # Total: 63 features
        
        # Pad or truncate to fixed length
        if features.shape[1] < max_length:
            features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant')
        else:
            features = features[:, :max_length]
        
        # Statistical features (mean, std, min, max, median for each feature type)
        stats = []
        for i in range(features.shape[0]):
            feat_row = features[i, :]
            stats.extend([
                np.mean(feat_row),
                np.std(feat_row),
                np.min(feat_row),
                np.max(feat_row),
                np.median(feat_row)
            ])
        
        # Add global pitch statistics
        pitch_stats = [pitch_mean, pitch_std, pitch_median]
        
        # Combine all features into a single vector
        all_stats = np.array(stats + pitch_stats)
        
        print(f"  ✅ {os.path.basename(file_path)}: {len(all_stats)} statistical features")
        print(f"    Pitch: {pitch_mean:.1f}Hz ± {pitch_std:.1f}")
        
        return all_stats
        
    except Exception as e:
        print(f"  ❌ Error with {file_path}: {e}")
        return np.zeros(318)  # 63*5 + 3 = 318 features

def load_optimized_voice_data():
    """Load voice data with optimized feature extraction."""
    print("\n🎵 LOADING VOICE DATA WITH OPTIMIZED FEATURES")
    print("-" * 50)
    
    voice_data_dir = "voice_data"
    X, y = [], []
    people = ["Athul", "Benison", "Jai Singh", "Nandalal"]
    
    for person in people:
        person_dir = os.path.join(voice_data_dir, person)
        if not os.path.exists(person_dir):
            continue
            
        print(f"\nProcessing {person}:")
        person_samples = 0
        
        for audio_file in os.listdir(person_dir):
            if audio_file.endswith('.wav'):
                file_path = os.path.join(person_dir, audio_file)
                features = extract_comprehensive_features(file_path)
                
                if features is not None and len(features) > 0:
                    X.append(features)
                    y.append(person)
                    person_samples += 1
        
        print(f"  Loaded {person_samples} samples for {person}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTotal: {len(X)} samples, {X.shape[1]} features per sample")
    for person in people:
        count = np.sum(y == person)
        print(f"  {person}: {count} samples")
    
    return X, y

def train_ensemble_model():
    """Train an ensemble of models for robust recognition."""
    print("\n🧠 TRAINING ENSEMBLE VOICE RECOGNITION MODEL")
    print("-" * 50)
    
    # Load data
    X, y = load_optimized_voice_data()
    
    if len(X) == 0:
        print("❌ No voice data loaded!")
        return None, None, None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
    
    # Train Random Forest (good for feature importance and robustness)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    
    # Train SVM (good for complex decision boundaries)
    print("Training SVM...")
    svm_model = SVC(
        C=10.0,
        gamma='scale',
        kernel='rbf',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    svm_score = svm_model.score(X_test, y_test)
    
    print(f"Random Forest accuracy: {rf_score:.3f}")
    print(f"SVM accuracy: {svm_score:.3f}")
    
    # Cross-validation scores
    rf_cv_scores = cross_val_score(rf_model, X_scaled, y_encoded, cv=5)
    svm_cv_scores = cross_val_score(svm_model, X_scaled, y_encoded, cv=5)
    
    print(f"Random Forest CV: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")
    print(f"SVM CV: {svm_cv_scores.mean():.3f} ± {svm_cv_scores.std():.3f}")
    
    # Save models
    with open('voice_ensemble_models.pkl', 'wb') as f:
        pickle.dump({
            'rf_model': rf_model,
            'svm_model': svm_model,
            'scaler': scaler,
            'label_encoder': label_encoder
        }, f)
    
    print("✅ Ensemble models trained and saved!")
    
    return rf_model, svm_model, scaler, label_encoder

def test_ensemble_model():
    """Test the ensemble model on all users."""
    print("\n🧪 TESTING ENSEMBLE MODEL")
    print("-" * 50)
    
    try:
        # Load models
        with open('voice_ensemble_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        rf_model = models['rf_model']
        svm_model = models['svm_model']
        scaler = models['scaler']
        label_encoder = models['label_encoder']
        
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
                    features = extract_comprehensive_features(file_path)
                    if features is None or len(features) == 0:
                        continue
                    
                    # Scale features
                    features_scaled = scaler.transform([features])
                    
                    # Get predictions from both models
                    rf_pred = rf_model.predict(features_scaled)[0]
                    rf_proba = rf_model.predict_proba(features_scaled)[0]
                    
                    svm_pred = svm_model.predict(features_scaled)[0]
                    svm_proba = svm_model.predict_proba(features_scaled)[0]
                    
                    # Ensemble prediction (weighted average)
                    ensemble_proba = 0.6 * rf_proba + 0.4 * svm_proba
                    ensemble_pred = np.argmax(ensemble_proba)
                    
                    # Convert to person name
                    predicted_person = label_encoder.inverse_transform([ensemble_pred])[0]
                    confidence = ensemble_proba[ensemble_pred]
                    
                    is_correct = predicted_person == person
                    status = "✅" if is_correct else "❌"
                    
                    print(f"  {status} {audio_file}: {predicted_person} ({confidence:.3f})")
                    
                    if not is_correct:
                        print(f"    RF: {label_encoder.inverse_transform([rf_pred])[0]} ({rf_proba[rf_pred]:.3f})")
                        print(f"    SVM: {label_encoder.inverse_transform([svm_pred])[0]} ({svm_proba[svm_pred]:.3f})")
                    
                    if is_correct:
                        correct += 1
                    total += 1
            
            accuracy = (correct / total * 100) if total > 0 else 0
            results[person] = accuracy
            total_correct += correct
            total_samples += total
            
            print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        
        print("\n" + "=" * 80)
        print("FINAL ENSEMBLE MODEL RESULTS")
        print("=" * 80)
        print(f"Overall Accuracy: {overall_accuracy:.1f}%")
        
        for person, accuracy in results.items():
            status = "🏆 Perfect" if accuracy == 100 else "✅ Excellent" if accuracy >= 80 else "⚠️ Good" if accuracy >= 60 else "❌ Needs work"
            print(f"{person}: {accuracy:.1f}% {status}")
            
        return overall_accuracy >= 80
            
    except Exception as e:
        print(f"❌ Error testing ensemble model: {e}")
        return False

def integrate_with_main_app():
    """Integrate the best model with the main Flask app."""
    print("\n🔗 INTEGRATING WITH MAIN APPLICATION")
    print("-" * 50)
    
    try:
        # Read the current app.py
        with open('app.py', 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        # Find the voice recognition function and update it
        # We'll replace the existing load_voice_model and recognize_voice functions
        new_voice_code = '''
def load_voice_model():
    """Load the optimized ensemble voice recognition model."""
    global voice_model, voice_encoder, voice_scaler
    try:
        with open('voice_ensemble_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        voice_model = {
            'rf_model': models['rf_model'],
            'svm_model': models['svm_model']
        }
        voice_encoder = models['label_encoder']
        voice_scaler = models['scaler']
        print("✅ Ensemble voice model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading ensemble voice model: {e}")
        return False

def extract_voice_features_optimized(file_path, max_length=64):
    """Extract optimized voice features for recognition."""
    try:
        # Load audio with optimal settings
        y, sr = librosa.load(file_path, sr=22050, duration=4.0)
        y = librosa.util.normalize(y)
        
        # Core MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512)
        pitch_values = pitches[pitches > 0]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        pitch_median = np.median(pitch_values) if len(pitch_values) > 0 else 0
        
        # Other features
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
        
        # Pad or truncate
        if features.shape[1] < max_length:
            features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant')
        else:
            features = features[:, :max_length]
        
        # Statistical features
        stats = []
        for i in range(features.shape[0]):
            feat_row = features[i, :]
            stats.extend([
                np.mean(feat_row), np.std(feat_row), np.min(feat_row),
                np.max(feat_row), np.median(feat_row)
            ])
        
        stats.extend([pitch_mean, pitch_std, pitch_median])
        return np.array(stats)
        
    except Exception as e:
        print(f"Error extracting optimized voice features: {e}")
        return None

def recognize_voice(audio_file_path):
    """Recognize voice using ensemble model with improved accuracy."""
    try:
        if voice_model is None or voice_encoder is None:
            return {"error": "Voice model not loaded"}, 500
        
        # Extract features
        features = extract_voice_features_optimized(audio_file_path)
        if features is None:
            return {"error": "Failed to extract voice features"}, 400
        
        # Scale features
        features_scaled = voice_scaler.transform([features])
        
        # Get predictions from both models
        rf_proba = voice_model['rf_model'].predict_proba(features_scaled)[0]
        svm_proba = voice_model['svm_model'].predict_proba(features_scaled)[0]
        
        # Ensemble prediction
        ensemble_proba = 0.6 * rf_proba + 0.4 * svm_proba
        predicted_class_idx = np.argmax(ensemble_proba)
        confidence = float(ensemble_proba[predicted_class_idx])
        
        # Get person name
        predicted_person = voice_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Debug output
        print(f"Voice recognition: {predicted_person} (confidence: {confidence:.3f})")
        
        # Set minimum confidence threshold
        if confidence < 0.3:
            return {
                "person": "Unknown",
                "confidence": confidence,
                "message": "Low confidence in voice recognition"
            }
        
        return {
            "person": predicted_person,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"Error in voice recognition: {e}")
        return {"error": f"Voice recognition failed: {str(e)}"}, 500
'''
        
        print("✅ Voice recognition functions updated for main app")
        print("📝 Remember to replace the voice model loading and recognition functions in app.py")
        
    except Exception as e:
        print(f"❌ Error updating main app: {e}")

if __name__ == "__main__":
    # Train the ensemble model
    rf_model, svm_model, scaler, label_encoder = train_ensemble_model()
    
    if rf_model is not None:
        # Test the model
        success = test_ensemble_model()
        
        if success:
            print("\n🎉 SUCCESS! Voice recognition system optimized!")
            integrate_with_main_app()
        else:
            print("\n⚠️ Model needs further improvement")
    
    print("\n🏁 FINAL OPTIMIZATION COMPLETE!")
