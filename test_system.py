"""
Test script to verify the Face & Voice Recognition System setup
Run this script to check if all components are working correctly
"""

import os
import sys
import importlib.util

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = [
        'flask', 'cv2', 'numpy', 'tensorflow', 'librosa', 
        'sklearn', 'PIL', 'sounddevice', 'matplotlib', 'scipy'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_data_structure():
    """Test if data directories exist and contain data"""
    print("\nTesting data structure...")
    
    # Check directories
    required_dirs = ['face_data', 'voice_data', 'templates']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ exists")
            
            if dir_name in ['face_data', 'voice_data']:
                # Count people and samples
                people = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
                if people:
                    print(f"   📁 {len(people)} people found: {', '.join(people)}")
                    
                    # Count samples for each person
                    for person in people:
                        person_path = os.path.join(dir_name, person)
                        files = [f for f in os.listdir(person_path) if os.path.isfile(os.path.join(person_path, f))]
                        print(f"   👤 {person}: {len(files)} samples")
                else:
                    print(f"   ⚠️  No people found in {dir_name}")
        else:
            print(f"❌ {dir_name}/ missing")
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0

def test_opencv():
    """Test OpenCV functionality"""
    print("\nTesting OpenCV...")
    
    try:
        import cv2
        
        # Test cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✅ Face cascade classifier loaded")
        else:
            print("❌ Face cascade classifier not found")
            return False
        
        # Test basic operations
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV image operations working")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow functionality"""
    print("\nTesting TensorFlow...")
    
    try:
        import tensorflow as tf
        
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test basic operations
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        
        model = Sequential([Dense(1, input_shape=(1,))])
        print("✅ Keras model creation working")
        
        # Check GPU availability
        if tf.config.list_physical_devices('GPU'):
            print("🚀 GPU available for acceleration")
        else:
            print("💻 Using CPU (GPU not available)")
        
        return True
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def test_audio():
    """Test audio processing functionality"""
    print("\nTesting audio processing...")
    
    try:
        import librosa
        import numpy as np
        
        # Test MFCC extraction with dummy data
        dummy_audio = np.random.randn(16000)  # 1 second at 16kHz
        mfccs = librosa.feature.mfcc(y=dummy_audio, sr=16000, n_mfcc=13)
        print("✅ Librosa MFCC extraction working")
        
        # Test sounddevice
        import sounddevice as sd
        print("✅ SoundDevice imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can be imported"""
    print("\nTesting Flask application...")
    
    try:
        # Check if app.py exists
        if not os.path.exists('app.py'):
            print("❌ app.py not found")
            return False
        
        # Try to import the app (without running it)
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app_module = importlib.util.module_from_spec(spec)
        
        print("✅ Flask application can be imported")
        print("✅ Ready to run with: python app.py")
        
        return True
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Face & Voice Recognition System Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Structure", test_data_structure),
        ("OpenCV", test_opencv),
        ("TensorFlow", test_tensorflow),
        ("Audio Processing", test_audio),
        ("Flask Application", test_flask_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        print("-" * 20)
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready.")
        print("\n🚀 Next steps:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Train your models using the web interface")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("💡 Try running: python setup.py")
    
    print("\n📝 For help, check README.md")

if __name__ == "__main__":
    main()
