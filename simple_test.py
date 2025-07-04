import sys
print("Python version:", sys.version)
print("Starting voice recognition test...")

import os
print("Current directory:", os.getcwd())
print("Files in voice_data:")
for item in os.listdir("voice_data"):
    print(f"  {item}")

print("Attempting imports...")
try:
    import numpy as np
    print("✅ numpy imported")
    
    import librosa
    print("✅ librosa imported")
    
    import tensorflow as tf
    print("✅ tensorflow imported")
    
    from app import FaceVoiceRecognitionSystem
    print("✅ FaceVoiceRecognitionSystem imported")
    
    print("Initializing system...")
    system = FaceVoiceRecognitionSystem()
    print("✅ System initialized")
    
    print("Loading voice data...")
    system.load_voice_data()
    print("✅ Voice data loaded")
    print(f"Voice data shape: {system.voice_data.shape if len(system.voice_data) > 0 else 'No data'}")
    print(f"People: {system.voice_encoder.classes_ if hasattr(system.voice_encoder, 'classes_') else 'None'}")
    
    if len(system.voice_data) > 0:
        print("Training models...")
        system.train_models()
        print("✅ Training completed")
    else:
        print("❌ No voice data found")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")
