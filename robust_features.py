"""
Alternative feature extraction for older librosa versions
"""
import librosa
import numpy as np

def extract_robust_voice_features(y, sr=22050):
    """
    Extract robust voice features that work with different librosa versions
    """
    try:
        # Always start with MFCC (most reliable)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
        mfccs = mfccs[:, :130]
        
        # Try to get additional features
        additional_features = []
        
        # Try chroma
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        except:
            try:
                # Alternative method
                stft = librosa.stft(y)
                chroma = librosa.feature.chroma_stft(S=np.abs(stft), sr=sr)
            except:
                # Use spectral features as substitute
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
                
                # Stack spectral features to create 12-dimensional chroma substitute
                chroma = np.vstack([
                    spectral_centroid,
                    spectral_rolloff, 
                    zero_crossing_rate,
                    np.repeat(spectral_centroid, 3, axis=0),  # Repeat to get 6 features
                    np.repeat(spectral_rolloff, 3, axis=0),   # Repeat to get 9 features  
                    np.repeat(zero_crossing_rate, 3, axis=0)  # Repeat to get 12 features
                ])
        
        chroma = np.pad(chroma, ((0, 0), (0, max(0, 130 - chroma.shape[1]))), mode='constant')
        chroma = chroma[:, :130]
        additional_features.append(chroma)
        
        # Try spectral contrast
        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        except:
            # Use spectral bandwidth and other features as substitute
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            
            # Stack to create 7-dimensional contrast substitute
            contrast = np.vstack([
                spectral_bandwidth,
                spectral_flatness,
                np.repeat(spectral_bandwidth, 2, axis=0),  # Get to 4 features
                np.repeat(spectral_flatness, 3, axis=0)    # Get to 7 features
            ])
        
        contrast = np.pad(contrast, ((0, 0), (0, max(0, 130 - contrast.shape[1]))), mode='constant')
        contrast = contrast[:, :130]
        additional_features.append(contrast)
        
        # Combine all features
        combined_features = np.vstack([mfccs] + additional_features)
        
        return combined_features
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        # Fallback to enhanced MFCC only
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)  # Use more MFCC coefficients
        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
        mfccs = mfccs[:, :130]
        return mfccs

if __name__ == "__main__":
    # Test the robust feature extraction
    dummy_audio = np.random.random(3 * 22050)
    features = extract_robust_voice_features(dummy_audio)
    print(f"Extracted features shape: {features.shape}")
    print("✅ Robust feature extraction working!")
