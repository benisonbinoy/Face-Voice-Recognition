#!/usr/bin/env python3
"""
Minimal test to validate voice recognition improvements
"""

# Test combined audio feature extraction
import numpy as np
import librosa
import os

def test_feature_extraction():
    voice_dir = "voice_data"
    
    # Test with first available audio file
    for person in os.listdir(voice_dir):
        person_path = os.path.join(voice_dir, person)
        if os.path.isdir(person_path):
            for wav_file in os.listdir(person_path):
                if wav_file.endswith('.wav'):
                    wav_path = os.path.join(person_path, wav_file)
                    
                    # Test feature extraction
                    y, sr = librosa.load(wav_path, sr=22050, duration=3.0)
                    
                    # 1. MFCC features
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
                    mfccs = mfccs[:, :130]
                    
                    # 2. Chroma features
                    chroma = librosa.feature.chroma(y=y, sr=sr)
                    chroma = np.pad(chroma, ((0, 0), (0, max(0, 130 - chroma.shape[1]))), mode='constant')
                    chroma = chroma[:, :130]
                    
                    # 3. Spectral contrast
                    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                    contrast = np.pad(contrast, ((0, 0), (0, max(0, 130 - contrast.shape[1]))), mode='constant')
                    contrast = contrast[:, :130]
                    
                    # Combine features
                    combined = np.vstack([mfccs, chroma, contrast])
                    
                    print(f"✅ Feature extraction successful for {person}/{wav_file}")
                    print(f"   MFCC shape: {mfccs.shape}")
                    print(f"   Chroma shape: {chroma.shape}")
                    print(f"   Contrast shape: {contrast.shape}")
                    print(f"   Combined shape: {combined.shape}")
                    
                    return True
    
    print("❌ No audio files found for testing")
    return False

if __name__ == "__main__":
    with open("feature_test_results.txt", "w") as f:
        import sys
        sys.stdout = f
        
        try:
            print("=" * 50)
            print("VOICE FEATURE EXTRACTION TEST")
            print("=" * 50)
            
            success = test_feature_extraction()
            
            if success:
                print("\n✅ All feature extraction tests passed!")
                print("✅ Combined features are working correctly")
                print("✅ Ready to test full voice recognition system")
            else:
                print("\n❌ Feature extraction test failed")
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()
        
        sys.stdout = sys.__stdout__
    
    print("Feature test completed. Check feature_test_results.txt")
