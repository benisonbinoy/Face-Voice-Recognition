#!/usr/bin/env python3
"""
Test script to evaluate the improved voice recognition system
"""

import sys
import os
import numpy as np
import librosa
from app import FaceVoiceRecognitionSystem

def test_voice_recognition():
    """Test the voice recognition system with existing voice samples"""
    print("=" * 60)
    print("TESTING VOICE RECOGNITION SYSTEM")
    print("=" * 60)
    
    # Initialize the system
    system = FaceVoiceRecognitionSystem()
    
    # Load and train models
    print("\n1. Loading voice data...")
    system.load_voice_data()
    
    print(f"\nVoice data shape: {system.voice_data.shape if len(system.voice_data) > 0 else 'No data'}")
    print(f"Number of voice samples: {len(system.voice_data) if len(system.voice_data) > 0 else 0}")
    print(f"People in voice dataset: {system.voice_encoder.classes_ if hasattr(system.voice_encoder, 'classes_') else 'None'}")
    
    if len(system.voice_data) == 0:
        print("❌ No voice data found! Please ensure voice_data folder contains WAV files.")
        return
    
    print("\n2. Training voice model...")
    system.train_models()
    
    if system.voice_model is None:
        print("❌ Voice model training failed!")
        return
    
    print("✅ Voice model trained successfully!")
    
    # Test with actual voice samples
    print("\n3. Testing recognition with actual voice samples...")
    voice_dir = "voice_data"
    
    test_results = []
    
    for person_name in os.listdir(voice_dir):
        person_path = os.path.join(voice_dir, person_name)
        if os.path.isdir(person_path):
            print(f"\nTesting samples for: {person_name}")
            person_correct = 0
            person_total = 0
            
            for wav_file in os.listdir(person_path):
                if wav_file.endswith('.wav'):
                    wav_path = os.path.join(person_path, wav_file)
                    try:
                        # Load audio (using same sample rate as training)
                        audio_data, sr = librosa.load(wav_path, sr=22050, duration=3.0)
                        
                        # Test recognition
                        result = system.recognize_voice(audio_data)
                        
                        is_correct = result['name'].lower() == person_name.lower()
                        status_icon = "✅" if is_correct else "❌"
                        
                        print(f"  {wav_file}: {status_icon} Predicted: {result['name']} (Confidence: {result['confidence']:.3f})")
                        print(f"    Status: {result['status']}, Message: {result['message']}")
                        
                        person_correct += 1 if is_correct else 0
                        person_total += 1
                        
                    except Exception as e:
                        print(f"  ❌ Error testing {wav_file}: {e}")
            
            if person_total > 0:
                accuracy = person_correct / person_total
                print(f"  📊 Accuracy for {person_name}: {person_correct}/{person_total} ({accuracy:.2%})")
                test_results.append((person_name, person_correct, person_total, accuracy))
            else:
                print(f"  ⚠️ No valid samples found for {person_name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    
    if test_results:
        total_correct = sum(result[1] for result in test_results)
        total_samples = sum(result[2] for result in test_results)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        print(f"Overall Accuracy: {total_correct}/{total_samples} ({overall_accuracy:.2%})")
        print("\nPer-person results:")
        for name, correct, total, accuracy in test_results:
            print(f"  {name}: {correct}/{total} ({accuracy:.2%})")
        
        # Performance analysis
        print("\n📈 PERFORMANCE ANALYSIS:")
        if overall_accuracy >= 0.8:
            print("✅ EXCELLENT: Voice recognition is working very well!")
        elif overall_accuracy >= 0.6:
            print("✅ GOOD: Voice recognition is working reasonably well.")
        elif overall_accuracy >= 0.4:
            print("⚠️ FAIR: Voice recognition needs improvement.")
        else:
            print("❌ POOR: Voice recognition needs significant improvement.")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if overall_accuracy < 0.8:
            print("- Consider adding more voice samples per person")
            print("- Check audio quality (clear speech, minimal background noise)")
            print("- Verify all people have distinctly different voice characteristics")
            
        if any(result[3] < 0.5 for result in test_results):
            worst_person = min(test_results, key=lambda x: x[3])
            print(f"- {worst_person[0]} has the lowest accuracy ({worst_person[3]:.2%}) - check their voice samples")
    else:
        print("❌ No test results available!")

if __name__ == "__main__":
    test_voice_recognition()
