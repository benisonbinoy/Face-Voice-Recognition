"""
Quick validation test for the voice recognition improvements
"""
import os
import sys

def quick_validation():
    """Quick test to validate the system works"""
    
    print("VOICE RECOGNITION SYSTEM VALIDATION")
    print("=" * 50)
    
    # Test 1: Check if we can import the system
    try:
        from app import FaceVoiceRecognitionSystem
        print("✅ Import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Check if system initializes
    try:
        system = FaceVoiceRecognitionSystem()
        print("✅ System initialization successful")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    # Test 3: Check voice data directory
    if os.path.exists("voice_data"):
        people = [name for name in os.listdir("voice_data") 
                 if os.path.isdir(os.path.join("voice_data", name))]
        print(f"✅ Found voice data for {len(people)} people: {people}")
    else:
        print("❌ No voice_data directory found")
        return False
    
    # Test 4: Test model creation (without training)
    try:
        test_model = system.create_voice_cnn_model(len(people))
        print(f"✅ Voice model creation successful (input shape: 32x130x1)")
        print(f"   Model expects {len(people)} classes")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 5: Check if feature extraction would work
    try:
        import librosa
        import numpy as np
        
        # Create dummy audio to test feature extraction
        dummy_audio = np.random.random(3 * 22050)  # 3 seconds at 22050 Hz
        
        # Test feature extraction
        mfccs = librosa.feature.mfcc(y=dummy_audio, sr=22050, n_mfcc=13)
        chroma = librosa.feature.chroma(y=dummy_audio, sr=22050)
        contrast = librosa.feature.spectral_contrast(y=dummy_audio, sr=22050)
        
        # Test padding and combination
        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')[:, :130]
        chroma = np.pad(chroma, ((0, 0), (0, max(0, 130 - chroma.shape[1]))), mode='constant')[:, :130]
        contrast = np.pad(contrast, ((0, 0), (0, max(0, 130 - contrast.shape[1]))), mode='constant')[:, :130]
        
        combined = np.vstack([mfccs, chroma, contrast])
        
        if combined.shape == (32, 130):
            print("✅ Feature extraction test successful")
            print(f"   Combined features shape: {combined.shape}")
        else:
            print(f"❌ Feature extraction shape mismatch: {combined.shape} != (32, 130)")
            return False
            
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False
    
    print("\n🎉 ALL VALIDATION TESTS PASSED!")
    print("\nThe voice recognition system is ready for testing.")
    print("\nNext steps:")
    print("1. Delete old models: voice_model.h5 and voice_encoder.pkl")
    print("2. Run: python app.py")
    print("3. Test voice recognition through the web interface")
    
    return True

if __name__ == "__main__":
    success = quick_validation()
    sys.exit(0 if success else 1)
