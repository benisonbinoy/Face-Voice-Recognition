"""
Face and Voice Recognition System Setup Script
This script helps set up the environment and train initial models
"""

import os
import sys
import subprocess
import platform

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Successfully installed all requirements")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def check_data_structure():
    """Check if data directories exist and have proper structure"""
    print("Checking data structure...")
    
    required_dirs = ["face_data", "voice_data", "templates"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        print("Please ensure your data is organized as follows:")
        print("face_data/")
        print("  ├── Person1/")
        print("  │   ├── img1.jpg")
        print("  │   └── img2.jpg")
        print("  └── Person2/")
        print("      ├── img1.jpg")
        print("      └── img2.jpg")
        print("\nvoice_data/")
        print("  ├── Person1/")
        print("  │   ├── sample1.wav")
        print("  │   └── sample2.wav")
        print("  └── Person2/")
        print("      ├── sample1.wav")
        print("      └── sample2.wav")
        return False
    
    # Check if there's actual data
    face_people = len([d for d in os.listdir("face_data") if os.path.isdir(os.path.join("face_data", d))])
    voice_people = len([d for d in os.listdir("voice_data") if os.path.isdir(os.path.join("voice_data", d))])
    
    print(f"✅ Found {face_people} people in face_data")
    print(f"✅ Found {voice_people} people in voice_data")
    
    if face_people == 0 or voice_people == 0:
        print("⚠️  Warning: No data found. Please add face and voice samples before training.")
    
    return True

def setup_models():
    """Initialize and train models if they don't exist"""
    print("Checking for existing models...")
    
    if os.path.exists("face_model.h5") and os.path.exists("voice_model.h5"):
        print("✅ Existing models found")
        return True
    
    print("No existing models found. You can train new models using the web interface.")
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Face & Voice Recognition System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return
    
    print(f"✅ Python {sys.version}")
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check data structure
    if not check_data_structure():
        return
    
    # Setup models
    setup_models()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nTo run the application:")
    print("python app.py")
    print("\nThen open your browser and go to:")
    print("http://localhost:5000")
    print("\n📝 Remember to:")
    print("1. Add your face images to face_data/[person_name]/")
    print("2. Add your voice samples to voice_data/[person_name]/")
    print("3. Train the models using the web interface")

if __name__ == "__main__":
    main()
