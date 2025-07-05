# Face & Voice Recognition System 🎭

A modern web-based face and voice recognition system using advanced CNN (Convolutional Neural Networks) with enhanced feature extraction and multiple model architectures for improved accuracy.

## 🔥 Key Updates

### 🎯 Advanced Voice Recognition System (Latest)
- **Dual-Model Architecture**: Implemented intelligent switching between multiple specialized models for optimal accuracy
- **Enhanced Feature Extraction**: Upgraded to combined MFCC (13), Chroma (12), and Spectral Contrast (7) features for 32-dimensional audio analysis
- **Specialized Models**: Developed Athul-focused, balanced, and ensemble models for different recognition scenarios
- **Consensus Prediction**: Added smart model combination with confidence-based decision making
- **Robust Error Handling**: Comprehensive fallbacks and improved audio processing reliability

### 🧠 Machine Learning Improvements (Recent)
- **Person-Focused Models**: Created specialized models (e.g., Athul-focused achieving 100% accuracy)
- **Balanced Multi-User Model**: Optimized for 60% overall accuracy across all users  
- **Ensemble Methods**: Integrated Random Forest + SVM models with statistical features
- **Enhanced CNN Architecture**: Updated network capacity with better regularization and dropout
- **High Sample Rate Processing**: Upgraded to 22050 Hz audio processing for improved quality

### 🔧 Analysis & Testing Framework (New)
- **Comprehensive Analysis Tools**: Added multiple voice analysis and model comparison scripts
- **Performance Validation**: Created extensive testing framework with automated comparison capabilities
- **Model Optimization Scripts**: Developed specialized improvement tools for voice recognition
- **Configuration Management**: Centralized config system for easy parameter tuning

### 🎨 Modern UI/UX Interface (Base)
- **Dark/Light Mode Toggle**: Seamless theme switching with beautiful dark mode featuring true black backgrounds and red accent elements
- **Theme Persistence**: Automatically remembers user preference across sessions
- **Glassmorphism Design**: Beautiful glass-like UI with backdrop blur effects and transparency
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Interactive Components**: Real-time feedback, drag-and-drop support, and visual indicators
- **Dedicated Pages**: Specialized interfaces for training, recognition, and system configuration
- **Live Media Integration**: Real-time camera feed and microphone input with visual feedback

## Features ✨

- **Advanced Face Recognition**: CNN-based facial recognition using computer vision with real-time camera integration
- **Enhanced Voice Recognition**: Multi-feature extraction (MFCC, Chroma, Spectral Contrast) with CNN and live recording
- **Multiple Model Architectures**: Standard, focused, balanced, and ensemble models for optimal accuracy
- **Dual-Model System**: Intelligent model switching and consensus for better accuracy
- **Modern Web UI**: Glassmorphism design with dark/light mode toggle, theme persistence, responsive layout and smooth animations
- **Real-time Processing**: Live camera feed and microphone input with visual feedback
- **Interactive Components**: Drag-and-drop file uploads, progress indicators, and status animations
- **Dedicated Pages**: Specialized interfaces for training, recognition, and system settings
- **Robust Feature Extraction**: Enhanced audio processing with fallback support
- **Model Training**: Train multiple custom models with your own data through intuitive interfaces
- **Multi-format Support**: Images (JPG, PNG) and audio files (WAV, MP3) with preview capabilities
- **Performance Analytics**: Detailed testing and validation tools with visual dashboards

## Technologies Used 🔧

- **Backend**: Python Flask, TensorFlow/Keras, OpenCV with advanced CNN architectures
- **Frontend**: HTML5, CSS3, JavaScript with Glassmorphism design and modern animations
- **UI/UX**: Responsive design, Inter typography, gradient backgrounds, interactive components
- **Machine Learning**: Advanced CNN models with multiple architectures and ensemble methods
- **Audio Processing**: Librosa with enhanced feature extraction (MFCC, Chroma, Spectral Contrast)
- **Computer Vision**: OpenCV for face detection and real-time camera processing
- **Model Optimization**: Ensemble methods, Random Forest, SVM with intelligent model switching
- **Configuration Management**: Centralized config system with performance analytics

## Installation 🚀

### Method 1: Quick Start (Windows)
```bash
# Start the application directly
start.bat

# Or run tests
run_test.bat
```

### Method 2: Automatic Setup
```bash
python setup.py
```

### Method 3: Manual Setup
1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Python environment** (if needed):
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. **Organize your training data**:
```
face_data/
├── Person1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── Person2/
    ├── img1.jpg
    ├── img2.jpg
    └── ...

voice_data/
├── Person1/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── Person2/
    ├── sample1.wav
    ├── sample2.wav
    └── ...
```

4. **Run the application**:
```bash
python app.py
```

5. **Access the web interface**:
   - Open your browser and go to: `http://localhost:5000`
   - The modern interface will load with all features available

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended for model training)
- **Storage**: 2GB free space for models and data
- **Browser**: Modern browser with JavaScript enabled
- **Hardware**: Camera and microphone for live recognition features

## Usage 📖

### 🎨 Modern Web Interface
The system features a beautiful, modern interface with:
- **Glassmorphism Design**: Translucent cards with backdrop blur effects
- **Smooth Animations**: Fade-in, slide-up, and hover animations throughout
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Interactive Elements**: Hover effects, progress indicators, and real-time feedback

### 1. Training Models
- Navigate to dedicated training pages through the main interface
- **Face Training**: Upload images through drag-and-drop or file browser
- **Voice Training**: Record samples directly or upload audio files
- Real-time progress tracking with animated progress bars
- Visual feedback for training completion and model status

### 2. Face Recognition
- **Live Camera**: Modern camera interface with real-time preview
  - Click "Start Camera" for live feed
  - "Capture Photo" with instant preview
  - Automatic face detection overlay
- **File Upload**: Drag-and-drop interface with file preview
- Results display with confidence scores and visual indicators
- Smooth transitions between states (scanning, processing, results)

### 3. Voice Recognition
- **Live Recording**: Interactive microphone interface
  - Visual recording indicators with pulse animations
  - Real-time audio level visualization
  - "Start Recording" → speak → "Stop Recording" workflow
- **File Upload**: Modern file browser with audio preview
- Audio waveform visualization during processing
- Results with confidence scores and alternative predictions

### 🎯 Enhanced User Experience Features
- **Visual Feedback**: Loading spinners, progress bars, and status indicators
- **Error Handling**: User-friendly error messages with recovery suggestions
- **Settings Dashboard**: Comprehensive system configuration interface
- **Analytics View**: Performance metrics and recognition history

## Data Requirements 📊

### Face Data
- **Format**: JPG, PNG images
- **Quantity**: At least 5-10 images per person
- **Quality**: Clear, well-lit faces
- **Size**: Any size (automatically resized to 128x128)

### Voice Data
- **Format**: WAV, MP3 audio files
- **Duration**: 2-5 seconds per sample
- **Quantity**: At least 3-5 samples per person
- **Quality**: Clear speech, minimal background noise

## Model Architecture 🧠

### Face Recognition CNN
```
Input: RGB Image (128×128×3)
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Dense(128) → Output
```

### Dual-Model Voice Recognition System
The system implements an intelligent dual-model approach with multiple specialized architectures:

#### Enhanced CNN Models
```
Input: Multi-feature audio (32×130×1)
Features:
- MFCC (13): Spectral characteristics and timbre
- Chroma (12): Pitch class profiles and harmonic content  
- Spectral Contrast (7): Timbral texture differences
Total: 32 features per time frame

Architecture:
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Dense(128) → Output
```

#### Specialized Model Types
1. **Standard Model (`voice_model.h5`)**: General-purpose voice recognition
2. **Focused Models (`voice_model_athul_focused.h5`)**: Person-specific optimization
   - Athul-focused: 100% accuracy for Athul recognition
3. **Balanced Model (`balanced_voice_model.h5`)**: Multi-user optimization
   - Optimized for 60% overall accuracy across all users
4. **Ensemble Model (`voice_ensemble_models.pkl`)**: Random Forest + SVM combination
   - Uses statistical features for additional discrimination

#### Intelligent Model Selection
- **Consensus Prediction**: Combines outputs from multiple models
- **Confidence-based Decision**: Uses highest confidence predictions
- **Fallback System**: Ensures robust recognition even if individual models fail
- **Smart Switching**: Automatically selects best model for each prediction

## Recent Improvements 🚀

### � Advanced Voice Recognition System
- **Dual-Model Architecture**: Implemented intelligent model switching and consensus prediction
- **Multi-Feature Extraction**: Enhanced audio processing with MFCC, Chroma, and Spectral Contrast (32 features total)
- **Specialized Models**: Created person-focused models with up to 100% accuracy for specific users
- **Ensemble Methods**: Integrated Random Forest and SVM models for additional discrimination
- **Robust Error Handling**: Comprehensive fallbacks and improved audio processing reliability

### 🧠 Machine Learning Enhancements
- **Enhanced CNN Architecture**: Updated network capacity with better regularization and dropout layers
- **Feature-wise Normalization**: Optimized preprocessing maintaining feature relationships while standardizing ranges
- **High Sample Rate Processing**: Upgraded to 22050 Hz for better audio quality and detail capture
- **Model Performance Optimization**: Achieved 100% accuracy for specific users with focused models
- **Intelligent Model Selection**: Smart switching between models based on confidence scores and user patterns

### 🔧 Technical Optimizations
- **JSON Serialization**: Fixed numpy type conversion errors for reliable data handling
- **Memory Management**: Optimized model loading and prediction processes
- **Configuration System**: Centralized config management for easy parameter tuning
- **Comprehensive Testing**: Created extensive analysis and validation framework
- **Performance Analytics**: Detailed testing tools with automated comparison capabilities

### 🎨 Modern UI/UX Interface
- **Glassmorphism Design**: Beautiful glass-like UI with backdrop blur effects and transparency
- **Responsive Layout**: Fully responsive design adapting to all screen sizes and devices
- **Interactive Components**: Real-time feedback, drag-and-drop support, and visual indicators
- **Dedicated Pages**: Specialized interfaces for training, recognition, and system configuration
- **Live Media Integration**: Real-time camera feed and microphone input with visual feedback
- **Professional Design**: Modern typography, gradient backgrounds, and smooth animations

## API Endpoints 🔌

### Web Interface Routes
- `GET /` - Main dashboard with modern glassmorphism design
- `GET /home` - Alternative home interface
- `GET /face-recognition` - Face verification interface with live camera
- `GET /voice-recognition` - Voice verification interface with real-time audio
- `GET /train-face` - Face model training interface
- `GET /train-voice` - Voice model training interface  
- `GET /settings` - System configuration and analytics dashboard
- `GET /results` - Recognition results display

### API Endpoints
- `POST /train` - Train face and voice models with current data
- `POST /recognize_face` - Face recognition from uploaded image or camera capture
- `POST /recognize_voice` - Voice recognition using dual-model system with ensemble methods
- `POST /upload_face_data` - Upload face training images
- `POST /upload_voice_data` - Upload voice training samples

### Advanced Features
- **Dual-Model Voice Recognition**: Automatically uses multiple models for consensus prediction
- **Real-time Processing**: Live camera feed and microphone input with instant feedback
- **Model Analytics**: Performance metrics and confidence scoring
- **Smart Model Selection**: Automatic switching between specialized models based on confidence

## 🎨 UI/UX Design Philosophy

### Modern Glassmorphism Aesthetic
Our interface embraces the latest design trends with:
- **Glass-like transparency effects** using CSS `backdrop-filter: blur()`
- **Layered depth** with shadows and gradient overlays
- **Smooth micro-interactions** that respond to user input
- **Professional color palette** with carefully chosen gradients

### User-Centric Design Principles
- **Intuitive Navigation**: Clear visual hierarchy and logical flow
- **Immediate Feedback**: Real-time visual responses to user actions
- **Accessibility**: High contrast ratios and readable typography
- **Progressive Enhancement**: Graceful degradation across devices

### Interactive Components
- **Animated State Changes**: Smooth transitions between UI states
- **Hover Effects**: Subtle animations that guide user attention
- **Loading States**: Engaging progress indicators and spinners
- **Error Recovery**: Clear messaging with actionable solutions

### Responsive Architecture
- **Mobile-First Design**: Optimized for touch interactions
- **Flexible Grid System**: Adapts to any screen size seamlessly
- **Touch-Friendly Controls**: Appropriately sized interactive elements
- **Performance Optimized**: Lightweight CSS with hardware acceleration

## Performance Tips 💡

1. **Face Recognition**:
   - Use well-lit, clear images with the enhanced camera interface
   - Include various angles and expressions through the training interface
   - Ensure faces are clearly visible with automatic detection feedback
   - Utilize the real-time camera preview for optimal positioning

2. **Voice Recognition**:
   - Record in quiet environments using the visual audio level indicators
   - Use consistent microphone/recording setup with the built-in recorder
   - Include various speaking styles through the enhanced training interface
   - Ensure 2-5 second audio samples for best results with waveform preview

3. **Model Training**:
   - More data generally improves accuracy - track progress with visual indicators
   - Balance the number of samples per person using the analytics dashboard
   - Use multiple model types for different scenarios through the settings panel
   - Retrain models when adding new people with guided workflows
   - Consider person-specific focused models for difficult cases

4. **System Optimization**:
   - The dual-model system automatically selects the best approach
   - Ensemble methods provide fallback for edge cases
   - Enhanced feature extraction improves discrimination
   - Monitor performance through the settings dashboard

5. **UI/UX Optimization**:
   - Use drag-and-drop functionality for faster file uploads
   - Monitor real-time feedback during training and recognition
   - Access quick settings through the modern interface
   - Utilize keyboard shortcuts and touch gestures where available

## Troubleshooting 🔧

### Common Issues

1. **Camera not working**:
   - Check browser permissions
   - Ensure no other applications are using the camera

2. **Microphone not working**:
   - Check browser permissions
   - Verify microphone is properly connected

3. **Poor recognition accuracy**:
   - Add more training samples
   - Ensure good quality data
   - Retrain the models

4. **Installation issues**:
   - Update pip: `python -m pip install --upgrade pip`
   - Use virtual environment
   - Check Python version (3.8+ required)

## File Structure 📁

```
Face Recognition/
├── app.py                          # Main Flask application with dual-model voice recognition
├── setup.py                       # Automated setup script
├── config.py                      # Configuration management system
├── deploy.py                      # Deployment utilities
├── requirements.txt               # Python dependencies
├── README.md                      # This documentation file
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
├── start.bat                      # Windows startup script
├── run_test.bat                   # Testing automation script
├── Templates/                     # Modern UI Templates
│   ├── index.html                 # Main dashboard with glassmorphism design
│   ├── home.html                  # Alternative home interface
│   ├── face_recognition.html      # Face verification interface
│   ├── voice_recognition.html     # Voice verification interface
│   ├── train_face.html           # Face model training interface
│   ├── train_voice.html          # Voice model training interface
│   ├── settings.html             # System configuration dashboard
│   └── results.html              # Recognition results display
├── Data Directories/              # Training Data (gitignored)
│   ├── face_data/                 # Face training images
│   ├── voice_data/                # Voice training samples
│   └── heic/                      # HEIC format images
├── Trained Models/                # AI Models & Encoders (gitignored)
│   ├── face_model.h5             # Face recognition CNN model
│   ├── face_encoder.pkl          # Face label encoder
│   ├── face_encodings.pkl        # Processed face encodings
│   ├── voice_model.h5            # Standard voice CNN model
│   ├── voice_encoder.pkl         # Voice label encoder
│   ├── voice_features.pkl        # Processed voice features
│   ├── balanced_voice_model.h5   # Balanced multi-user model
│   ├── balanced_voice_encoder.pkl # Balanced model encoder
│   ├── focused_voice_model.h5    # General focused model
│   ├── focused_voice_encoder.pkl # Focused model encoder
│   ├── voice_model_athul_focused.h5 # Athul-specific model (100% accuracy)
│   ├── voice_encoder_athul_focused.pkl # Athul-focused encoder
│   ├── voice_scalers_athul_focused.pkl # Athul-focused feature scalers
│   └── voice_ensemble_models.pkl # Ensemble RF+SVM models
├── Analysis & Testing Scripts/    # Development & Analysis Tools
│   ├── test_system.py            # Complete system validation
│   ├── validate_system.py        # Model testing framework
│   ├── test_enhanced_app.py      # Enhanced app testing
│   ├── test_voice_recognition.py # Voice recognition testing
│   ├── test_features_only.py     # Feature extraction testing
│   ├── test_librosa.py           # Librosa compatibility testing
│   ├── analyze_all_voices.py     # Comprehensive voice analysis
│   ├── compare_all_models.py     # Model comparison tools
│   ├── improve_all_voices.py     # Voice improvement scripts
│   ├── fix_voice_confusion.py    # Confusion resolution tools
│   ├── fix_athul_recognition.py  # Athul-specific improvements
│   ├── balanced_voice_system.py  # Balanced training system
│   ├── final_voice_optimization.py # Ensemble optimization
│   ├── robust_features.py        # Feature extraction utilities
│   ├── feature_test.py           # Feature testing tools
│   └── simple_test.py            # Basic testing script
├── Documentation/                 # Project Documentation
│   ├── FINAL_PROJECT_STATUS.md   # Complete project status report
│   ├── IMPROVEMENTS_SUMMARY.md   # Detailed improvement documentation
│   └── USAGE_SUMMARY.md         # Usage guidelines and best practices
├── Output Logs/                  # Analysis & Test Results
│   ├── voice_test_output.txt     # Voice testing results
│   ├── analysis_output.txt       # Comprehensive analysis logs
│   ├── improvement_output.txt    # Improvement process logs
│   └── feature_test_output.txt   # Feature extraction test results
└── Archive/                      # Legacy Files
    └── old/                      # Previous versions and prototypes
        ├── main.py               # Original implementation
        ├── server.py             # Legacy server code
        ├── front.html            # Original frontend
        └── *.py                  # Other legacy scripts
```

## Security Notes 🔒

- Models and data are stored locally
- No data is transmitted to external servers
- Camera and microphone access requires user permission
- Consider implementing authentication for production use

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License 📄

This project is open source and available under the MIT License.

## Credits 👏

- TensorFlow/Keras for deep learning framework
- OpenCV for computer vision operations
- Librosa for audio processing
- Flask for web framework
