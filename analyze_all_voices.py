#!/usr/bin/env python3
"""
Comprehensive voice analysis and improvement script
"""

import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from app import FaceVoiceRecognitionSystem
import traceback

def analyze_voice_samples():
    """Analyze voice samples for each person to identify issues"""
    print("=" * 80)
    print("COMPREHENSIVE VOICE ANALYSIS")
    print("=" * 80)
    
    voice_dir = "voice_data"
    people = [name for name in os.listdir(voice_dir) if os.path.isdir(os.path.join(voice_dir, name))]
    
    print(f"Found {len(people)} people: {people}")
    
    # Initialize system
    system = FaceVoiceRecognitionSystem()
    
    # Load and train the model
    print("\n1. Training the voice recognition system...")
    system.load_voice_data()
    system.train_models()
    
    if system.voice_model is None:
        print("❌ Failed to train voice model!")
        return
    
    print("✅ Voice model trained successfully!")
    
    # Analyze each person's voice characteristics
    print("\n2. Analyzing voice characteristics for each person...")
    
    voice_profiles = {}
    
    for person_name in people:
        print(f"\n--- Analyzing {person_name} ---")
        person_path = os.path.join(voice_dir, person_name)
        
        person_features = []
        sample_results = []
        
        for wav_file in os.listdir(person_path):
            if wav_file.endswith('.wav'):
                wav_path = os.path.join(person_path, wav_file)
                
                try:
                    # Load audio and extract features
                    audio_data, sr = librosa.load(wav_path, sr=22050, duration=3.0)
                    
                    # Extract the same features as the system uses
                    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
                    mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
                    mfccs = mfccs[:, :130]
                    
                    # Chroma features
                    try:
                        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
                    except:
                        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
                        chroma = np.repeat(spectral_centroid, 12, axis=0)
                    
                    chroma = np.pad(chroma, ((0, 0), (0, max(0, 130 - chroma.shape[1]))), mode='constant')
                    chroma = chroma[:, :130]
                    
                    # Spectral contrast
                    try:
                        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
                    except:
                        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
                        contrast = np.repeat(spectral_bandwidth, 7, axis=0)
                    
                    contrast = np.pad(contrast, ((0, 0), (0, max(0, 130 - contrast.shape[1]))), mode='constant')
                    contrast = contrast[:, :130]
                    
                    # Combine features
                    combined_features = np.vstack([mfccs, chroma, contrast])
                    person_features.append(combined_features)
                    
                    # Test recognition for this sample
                    result = system.recognize_voice(audio_data)
                    
                    is_correct = result['name'].lower() == person_name.lower()
                    sample_results.append({
                        'file': wav_file,
                        'predicted': result['name'],
                        'confidence': result['confidence'],
                        'correct': is_correct,
                        'status': result['status']
                    })
                    
                    # Audio quality metrics
                    rms_energy = np.sqrt(np.mean(audio_data**2))
                    zero_crossings = librosa.feature.zero_crossing_rate(audio_data)[0]
                    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
                    
                    print(f"  {wav_file}:")
                    print(f"    Recognition: {'✅' if is_correct else '❌'} {result['name']} ({result['confidence']:.3f})")
                    print(f"    Audio Quality: RMS={rms_energy:.4f}, ZCR={np.mean(zero_crossings):.4f}, SC={spectral_centroid_mean:.0f}Hz")
                    
                except Exception as e:
                    print(f"  ❌ Error processing {wav_file}: {e}")
        
        # Calculate statistics for this person
        if person_features:
            person_features = np.array(person_features)
            correct_predictions = sum(1 for r in sample_results if r['correct'])
            total_samples = len(sample_results)
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            
            avg_confidence = np.mean([r['confidence'] for r in sample_results])
            
            # Feature consistency analysis
            feature_std = np.std(person_features, axis=0)
            feature_consistency = np.mean(feature_std)
            
            voice_profiles[person_name] = {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'feature_consistency': feature_consistency,
                'sample_results': sample_results,
                'features_shape': person_features.shape
            }
            
            print(f"\n  📊 {person_name} Summary:")
            print(f"    Accuracy: {accuracy:.2%} ({correct_predictions}/{total_samples})")
            print(f"    Avg Confidence: {avg_confidence:.3f}")
            print(f"    Feature Consistency: {feature_consistency:.4f} (lower = more consistent)")
            
            # Identify specific issues
            if accuracy < 0.5:
                print(f"    🔴 MAJOR ISSUE: Very low accuracy - voice samples may have quality issues")
            elif accuracy < 0.8:
                print(f"    🟡 MINOR ISSUE: Some misclassifications - could benefit from more training data")
            else:
                print(f"    🟢 GOOD: High accuracy")
    
    # Cross-person analysis
    print("\n" + "=" * 80)
    print("CROSS-PERSON ANALYSIS")
    print("=" * 80)
    
    # Find people with similar voice characteristics
    print("\n3. Analyzing voice similarity between people...")
    
    for i, person1 in enumerate(people):
        for j, person2 in enumerate(people):
            if i < j:  # Avoid duplicate comparisons
                if person1 in voice_profiles and person2 in voice_profiles:
                    acc1 = voice_profiles[person1]['accuracy']
                    acc2 = voice_profiles[person2]['accuracy']
                    
                    # Check if they get confused with each other
                    person1_predictions = [r['predicted'] for r in voice_profiles[person1]['sample_results']]
                    person2_predictions = [r['predicted'] for r in voice_profiles[person2]['sample_results']]
                    
                    person1_confused_with_person2 = person1_predictions.count(person2)
                    person2_confused_with_person1 = person2_predictions.count(person1)
                    
                    if person1_confused_with_person2 > 0 or person2_confused_with_person1 > 0:
                        print(f"  ⚠️ {person1} ↔ {person2}: Confusion detected")
                        print(f"    {person1} misclassified as {person2}: {person1_confused_with_person2} times")
                        print(f"    {person2} misclassified as {person1}: {person2_confused_with_person1} times")
    
    # Generate improvement recommendations
    print("\n" + "=" * 80)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)
    
    for person_name, profile in voice_profiles.items():
        print(f"\n📋 Recommendations for {person_name}:")
        
        accuracy = profile['accuracy']
        avg_confidence = profile['avg_confidence']
        consistency = profile['feature_consistency']
        
        if accuracy < 0.3:
            print("  🔴 CRITICAL: Very low accuracy")
            print("    - Re-record all voice samples with better quality")
            print("    - Ensure clear speech with minimal background noise")
            print("    - Record in a quiet environment")
            print("    - Speak clearly and naturally")
        elif accuracy < 0.7:
            print("  🟡 MODERATE: Needs improvement")
            print("    - Add 5-10 more voice samples")
            print("    - Vary speaking style (fast/slow, different phrases)")
            print("    - Check audio quality of existing samples")
        else:
            print("  🟢 GOOD: Working well")
            print("    - Consider adding a few more samples for robustness")
        
        if avg_confidence < 0.5:
            print("  - Low confidence scores suggest voice similarity with others")
            print("  - Try recording with more distinctive speaking patterns")
        
        if consistency > 0.5:
            print("  - High feature inconsistency - check recording conditions")
            print("  - Ensure similar microphone distance and environment")
    
    # Overall system recommendations
    print(f"\n🎯 OVERALL SYSTEM RECOMMENDATIONS:")
    
    overall_accuracy = np.mean([p['accuracy'] for p in voice_profiles.values()])
    print(f"Current overall accuracy: {overall_accuracy:.2%}")
    
    if overall_accuracy < 0.6:
        print("- Consider re-recording samples for people with 0% accuracy")
        print("- Ensure all recordings are high quality and clear")
        print("- Add more training samples per person (aim for 10-15 samples)")
    elif overall_accuracy < 0.8:
        print("- Add more training samples for low-performing individuals")
        print("- Consider fine-tuning confidence thresholds")
        print("- Ensure recording consistency across all people")
    else:
        print("- System is performing well!")
        print("- Consider adding a few more samples for robustness")
    
    print("\n- Optimal sample count: 10-15 clear voice samples per person")
    print("- Recommended recording: 3-5 seconds of clear speech")
    print("- Environment: Quiet room with minimal echo")
    print("- Microphone: Consistent distance (6-12 inches from mouth)")

if __name__ == "__main__":
    try:
        analyze_voice_samples()
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        traceback.print_exc()
