#!/usr/bin/env python3
"""
Test the updated Flask app with enhanced dual-model voice recognition.
"""

import sys
import os
import librosa
import numpy as np

# Add the current directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import FaceVoiceRecognitionSystem
    print("✅ Successfully imported the FaceVoiceRecognitionSystem")
except Exception as e:
    print(f"❌ Failed to import app: {e}")
    sys.exit(1)

def test_voice_recognition():
    """Test the enhanced voice recognition system."""
    print("\n🧪 TESTING ENHANCED VOICE RECOGNITION SYSTEM")
    print("=" * 60)
    
    # Initialize the system
    system = FaceVoiceRecognitionSystem()
    
    # Load voice model
    print("Loading voice model...")
    try:
        system.load_voice_data()
        system.train_voice_model()
        print("✅ Voice model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load voice model: {e}")
        return False
    
    # Test voice recognition with sample files
    voice_data_dir = "voice_data"
    test_results = {}
    
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
                
                try:
                    # Load audio data
                    audio_data, sr = librosa.load(file_path, sr=22050, duration=3.0)
                    
                    # Test recognition
                    result = system.recognize_voice(audio_data)
                    
                    is_correct = result['name'] == person
                    status = "✅" if is_correct else "❌"
                    
                    print(f"  {status} {audio_file}: {result['name']} ({result['confidence']:.3f}) - {result['status']}")
                    print(f"    Model: {result.get('mode', 'unknown')}")
                    
                    if 'model_details' in result and result['model_details'].get('predictions'):
                        predictions = result['model_details']['predictions']
                        print(f"    Models used: {[p['model'] for p in predictions]}")
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    print(f"  ❌ Error testing {audio_file}: {e}")
        
        if total > 0:
            accuracy = (correct / total) * 100
            test_results[person] = accuracy
            print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Overall results
    if test_results:
        overall_accuracy = sum(test_results.values()) / len(test_results)
        print(f"\n🏆 OVERALL RESULTS")
        print(f"Average accuracy: {overall_accuracy:.1f}%")
        
        for person, accuracy in test_results.items():
            status = "🏆 Excellent" if accuracy >= 80 else "✅ Good" if accuracy >= 60 else "⚠️ Needs improvement"
            print(f"  {person}: {accuracy:.1f}% {status}")
        
        return overall_accuracy >= 70
    
    return False

if __name__ == "__main__":
    success = test_voice_recognition()
    
    if success:
        print("\n🎉 Enhanced voice recognition system is working well!")
        print("The Flask app is ready to use with improved accuracy.")
    else:
        print("\n⚠️ Voice recognition system needs further tuning.")
    
    print("\n💡 To run the Flask app: python app.py")
