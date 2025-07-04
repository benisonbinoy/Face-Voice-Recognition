import os
import sys
import traceback

print("=== VOICE RECOGNITION FEATURE TEST ===")

try:
    print("1. Testing imports...")
    import librosa
    import numpy as np
    print(f"✅ librosa version: {librosa.__version__}")
    print(f"✅ numpy version: {np.__version__}")
    
    print("\n2. Testing basic feature extraction...")
    
    # Find a voice file to test with
    voice_file = None
    for person in os.listdir("voice_data"):
        person_path = os.path.join("voice_data", person)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                if file.endswith('.wav'):
                    voice_file = os.path.join(person_path, file)
                    break
            if voice_file:
                break
    
    if not voice_file:
        print("❌ No voice files found")
        sys.exit(1)
    
    print(f"✅ Using test file: {voice_file}")
    
    # Test loading audio
    y, sr = librosa.load(voice_file, sr=22050, duration=3.0)
    print(f"✅ Audio loaded: shape={y.shape}, sr={sr}")
    
    # Test MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"✅ MFCC extracted: shape={mfccs.shape}")
    
    # Test chroma alternatives
    print("\n3. Testing chroma features...")
    chroma_methods = []
    
    # Method 1: chroma_stft
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_methods.append(("chroma_stft", chroma.shape))
        print(f"✅ chroma_stft: {chroma.shape}")
    except Exception as e:
        print(f"❌ chroma_stft failed: {e}")
    
    # Method 2: spectral_centroid fallback
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        chroma_fallback = np.repeat(spectral_centroid, 12, axis=0)
        chroma_methods.append(("spectral_centroid_x12", chroma_fallback.shape))
        print(f"✅ spectral_centroid fallback: {chroma_fallback.shape}")
    except Exception as e:
        print(f"❌ spectral_centroid fallback failed: {e}")
    
    # Test contrast alternatives
    print("\n4. Testing contrast features...")
    contrast_methods = []
    
    # Method 1: spectral_contrast
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_methods.append(("spectral_contrast", contrast.shape))
        print(f"✅ spectral_contrast: {contrast.shape}")
    except Exception as e:
        print(f"❌ spectral_contrast failed: {e}")
    
    # Method 2: spectral_bandwidth fallback
    try:
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast_fallback = np.repeat(spectral_bandwidth, 7, axis=0)
        contrast_methods.append(("spectral_bandwidth_x7", contrast_fallback.shape))
        print(f"✅ spectral_bandwidth fallback: {contrast_fallback.shape}")
    except Exception as e:
        print(f"❌ spectral_bandwidth fallback failed: {e}")
    
    # Method 3: zero_crossing_rate ultimate fallback
    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_fallback = np.repeat(zcr, 7, axis=0)
        contrast_methods.append(("zcr_x7", zcr_fallback.shape))
        print(f"✅ zero_crossing_rate fallback: {zcr_fallback.shape}")
    except Exception as e:
        print(f"❌ zero_crossing_rate fallback failed: {e}")
    
    print("\n5. Testing combined features...")
    if chroma_methods and contrast_methods:
        # Use the first working method for each
        _, chroma_shape = chroma_methods[0]
        _, contrast_shape = contrast_methods[0]
        
        print(f"   MFCC: {mfccs.shape}")
        print(f"   Chroma: {chroma_shape}")
        print(f"   Contrast: {contrast_shape}")
        
        # Test if we can combine them (just shape check)
        expected_combined_shape = (13 + chroma_shape[0] + contrast_shape[0], min(mfccs.shape[1], chroma_shape[1], contrast_shape[1]))
        print(f"   Expected combined: {expected_combined_shape}")
        
        if expected_combined_shape[0] == 32:
            print("✅ Perfect! Combined features will be (32, X)")
        else:
            print(f"⚠️ Combined features will be ({expected_combined_shape[0]}, X) - not exactly 32")
    
    print("\n🎉 FEATURE EXTRACTION TEST COMPLETED SUCCESSFULLY!")
    print("The system should now work with your librosa version.")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)
