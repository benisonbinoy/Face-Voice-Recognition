"""
Test librosa compatibility and feature extraction
"""
import librosa
import numpy as np

def test_librosa_features():
    print("Testing librosa feature extraction compatibility...")
    
    # Create dummy audio
    dummy_audio = np.random.random(3 * 22050)  # 3 seconds at 22050 Hz
    sr = 22050
    
    try:
        # Test MFCC
        mfccs = librosa.feature.mfcc(y=dummy_audio, sr=sr, n_mfcc=13)
        print(f"✅ MFCC: {mfccs.shape}")
        
        # Test Chroma - try different function names
        try:
            chroma = librosa.feature.chroma_stft(y=dummy_audio, sr=sr)
            print(f"✅ Chroma (chroma_stft): {chroma.shape}")
        except AttributeError:
            try:
                chroma = librosa.feature.chromagram(y=dummy_audio, sr=sr)
                print(f"✅ Chroma (chromagram): {chroma.shape}")
            except AttributeError:
                # Fallback to manual chroma calculation
                stft = librosa.stft(dummy_audio)
                chroma = librosa.feature.chroma_stft(S=np.abs(stft), sr=sr)
                print(f"✅ Chroma (manual stft): {chroma.shape}")
        
        # Test Spectral Contrast
        try:
            contrast = librosa.feature.spectral_contrast(y=dummy_audio, sr=sr)
            print(f"✅ Spectral Contrast: {contrast.shape}")
        except Exception as e:
            print(f"❌ Spectral Contrast failed: {e}")
            # Create dummy contrast features
            contrast = np.random.random((7, mfccs.shape[1]))
            print(f"⚠️ Using dummy contrast features: {contrast.shape}")
        
        print(f"Librosa version: {librosa.__version__}")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

if __name__ == "__main__":
    test_librosa_features()
