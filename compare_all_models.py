#!/usr/bin/env python3
"""
Comprehensive test of all voice recognition models to find the best one.
"""

import numpy as np
import librosa
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🏆 COMPREHENSIVE VOICE MODEL EVALUATION")
print("=" * 80)

def extract_features_for_model(file_path, model_type):
    """Extract features based on model type."""
    try:
        if model_type == "ensemble":
            # Extract comprehensive features for ensemble model
            y, sr = librosa.load(file_path, sr=22050, duration=4.0)
            y = librosa.util.normalize(y)
            
            # Core features
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
            return np.array(stats)
            
        elif model_type == "athul_focused":
            # Features for Athul-focused model
            y, sr = librosa.load(file_path, sr=22050, duration=3.0)
            y = librosa.util.normalize(y)
            
            # Enhanced features for CNN model
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # Additional features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512)
            pitch_mean = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)
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
            
            # Pad or truncate to 32 time steps
            if features.shape[1] < 32:
                features = np.pad(features, ((0, 0), (0, 32 - features.shape[1])), mode='constant')
            else:
                features = features[:, :32]
            
            # Add pitch as a separate feature
            pitch_features = np.full((1, 32), pitch_mean)
            features = np.vstack([features, pitch_features])  # Total: 36 features
            
            return features.T  # Return as (time_steps, features)
            
        else:  # Standard model
            # Basic features for standard model
            y, sr = librosa.load(file_path, sr=22050, duration=3.0)
            y = librosa.util.normalize(y)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            features = np.vstack([mfcc, chroma, spectral_contrast])
            
            if features.shape[1] < 130:
                features = np.pad(features, ((0, 0), (0, 130 - features.shape[1])), mode='constant')
            else:
                features = features[:, :130]
            
            return features.T
            
    except Exception as e:
        print(f"Error extracting features for {model_type}: {e}")
        return None

def test_model(model_info):
    """Test a specific model."""
    model_name = model_info['name']
    model_type = model_info['type']
    
    print(f"\n🧪 Testing {model_name}")
    print("-" * 40)
    
    try:
        # Load model based on type
        if model_type == "ensemble":
            with open('voice_ensemble_models.pkl', 'rb') as f:
                models = pickle.load(f)
            model = models
            
        elif model_type == "keras":
            from tensorflow.keras.models import load_model
            model = load_model(model_info['model_file'])
            
            with open(model_info['encoder_file'], 'rb') as f:
                label_encoder = pickle.load(f)
                
            if 'scaler_file' in model_info:
                with open(model_info['scaler_file'], 'rb') as f:
                    scalers = pickle.load(f)
            else:
                scalers = None
        
        voice_data_dir = "voice_data"
        results = {}
        total_correct = 0
        total_samples = 0
        
        for person in ["Athul", "Benison", "Jai Singh", "Nandalal"]:
            person_dir = os.path.join(voice_data_dir, person)
            if not os.path.exists(person_dir):
                continue
                
            correct = 0
            total = 0
            
            for audio_file in os.listdir(person_dir):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(person_dir, audio_file)
                    
                    # Extract features based on model type
                    features = extract_features_for_model(file_path, model_type)
                    if features is None:
                        continue
                    
                    # Make prediction based on model type
                    if model_type == "ensemble":
                        # Scale features
                        features_scaled = model['scaler'].transform([features])
                        
                        # Get predictions from both models
                        rf_proba = model['rf_model'].predict_proba(features_scaled)[0]
                        svm_proba = model['svm_model'].predict_proba(features_scaled)[0]
                        
                        # Ensemble prediction
                        ensemble_proba = 0.6 * rf_proba + 0.4 * svm_proba
                        predicted_class_idx = np.argmax(ensemble_proba)
                        confidence = ensemble_proba[predicted_class_idx]
                        
                        predicted_person = model['label_encoder'].inverse_transform([predicted_class_idx])[0]
                        
                    elif model_type == "keras":
                        if model_name == "Athul-Focused CNN":
                            # Apply normalization for Athul-focused model
                            features_normalized = features.copy()
                            
                            # Apply scalers if available
                            if scalers:
                                features_normalized[:, :13] = scalers['mfcc_scaler'].transform(features_normalized[:, :13])
                                features_normalized[:, 13:25] = scalers['chroma_scaler'].transform(features_normalized[:, 13:25])
                                features_normalized[:, 25:] = scalers['other_scaler'].transform(features_normalized[:, 25:])
                            
                            # Reshape for CNN
                            features_reshaped = features_normalized.reshape(1, 32, 36, 1)
                        else:
                            # Standard model
                            features_reshaped = features.reshape(1, features.shape[0], features.shape[1], 1)
                        
                        prediction = model.predict(features_reshaped, verbose=0)
                        predicted_class_idx = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class_idx]
                        
                        predicted_person = label_encoder.inverse_transform([predicted_class_idx])[0]
                    
                    is_correct = predicted_person == person
                    
                    if is_correct:
                        correct += 1
                    total += 1
            
            accuracy = (correct / total * 100) if total > 0 else 0
            results[person] = accuracy
            total_correct += correct
            total_samples += total
            
            print(f"  {person}: {accuracy:.1f}% ({correct}/{total})")
        
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        
        print(f"  Overall: {overall_accuracy:.1f}%")
        
        return overall_accuracy, results
        
    except Exception as e:
        print(f"  ❌ Error testing {model_name}: {e}")
        return 0, {}

def main():
    """Test all available models."""
    
    # Define all models to test
    models_to_test = [
        {
            'name': 'Ensemble (RF + SVM)',
            'type': 'ensemble'
        },
        {
            'name': 'Athul-Focused CNN',
            'type': 'keras',
            'model_file': 'voice_model_athul_focused.h5',
            'encoder_file': 'voice_encoder_athul_focused.pkl',
            'scaler_file': 'voice_scalers_athul_focused.pkl'
        },
        {
            'name': 'Balanced Voice Model',
            'type': 'keras',
            'model_file': 'balanced_voice_model.h5',
            'encoder_file': 'balanced_voice_encoder.pkl'
        },
        {
            'name': 'Standard Voice Model',
            'type': 'keras',
            'model_file': 'voice_model.h5',
            'encoder_file': 'voice_encoder.pkl'
        }
    ]
    
    all_results = {}
    
    for model_info in models_to_test:
        # Check if model files exist
        if model_info['type'] == 'ensemble':
            if not os.path.exists('voice_ensemble_models.pkl'):
                print(f"⚠️ Skipping {model_info['name']} - file not found")
                continue
        elif model_info['type'] == 'keras':
            if not os.path.exists(model_info['model_file']) or not os.path.exists(model_info['encoder_file']):
                print(f"⚠️ Skipping {model_info['name']} - files not found")
                continue
        
        overall_acc, person_results = test_model(model_info)
        all_results[model_info['name']] = {
            'overall': overall_acc,
            'per_person': person_results
        }
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 FINAL COMPARISON OF ALL MODELS")
    print("=" * 80)
    
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['overall'], reverse=True)
    
    for i, (model_name, results) in enumerate(sorted_models, 1):
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📈"
        print(f"{status} {model_name}: {results['overall']:.1f}% overall")
        
        for person, acc in results['per_person'].items():
            person_status = "✅" if acc >= 80 else "⚠️" if acc >= 60 else "❌"
            print(f"    {person_status} {person}: {acc:.1f}%")
    
    if sorted_models:
        best_model = sorted_models[0]
        print(f"\n🏆 WINNER: {best_model[0]} with {best_model[1]['overall']:.1f}% accuracy!")
        
        # Check if all users have good accuracy
        all_good = all(acc >= 60 for acc in best_model[1]['per_person'].values())
        if all_good:
            print("✅ All users have acceptable accuracy (≥60%)")
        else:
            print("⚠️ Some users still need improvement")

if __name__ == "__main__":
    main()
