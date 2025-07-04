# Face & Voice Recognition System 🎭

A modern web-based face and voice recognition system using advanced CNN (Convolutional Neural Networks) with enhanced feature extraction and multiple model architectures for improved accuracy.

## 🎉 Latest UI/UX Updates & Key Features

### ✨ Modern Interface Overhaul
- **Glassmorphism Design**: Beautiful glass-like UI with blur effects and transparency
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Smooth Animations**: Fade-in, slide-up, pulse, and hover effects throughout the interface
- **Interactive Components**: Real-time feedback, progress indicators, and visual state changes

### 🚀 Enhanced User Experience
- **Dedicated Pages**: Specialized interfaces for each function (training, recognition, settings)
- **Live Camera Integration**: Real-time camera feed with automatic face detection overlay
- **Audio Visualization**: Dynamic microphone indicators and waveform displays
- **Drag & Drop Support**: Modern file upload with preview capabilities
- **Progress Tracking**: Animated progress bars and status indicators for all operations

### 🎨 Professional Design Elements
- **Modern Typography**: Inter font family for clean, professional appearance
- **Gradient Backgrounds**: Beautiful color gradients and dynamic visual elements
- **Card-Based Layout**: Organized content in modern cards with shadow effects
- **Interactive Buttons**: Hover effects and smooth transitions for better user engagement

## Features ✨

- **Advanced Face Recognition**: CNN-based facial recognition using computer vision with real-time camera integration
- **Enhanced Voice Recognition**: Multi-feature extraction (MFCC, Chroma, Spectral Contrast) with CNN and live recording
- **Multiple Model Architectures**: Standard, focused, balanced, and ensemble models for optimal accuracy
- **Dual-Model System**: Intelligent model switching and consensus for better accuracy
- **Modern Web UI**: Glassmorphism design with responsive layout and smooth animations
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

### Method 1: Automatic Setup
```bash
python setup.py
```

### Method 2: Manual Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your data:
```
face_data/
├── Person1/
│   ├── img1.jpg
│   └── img2.jpg
└── Person2/
    ├── img1.jpg
    └── img2.jpg

voice_data/
├── Person1/
│   ├── sample1.wav
│   └── sample2.wav
└── Person2/
    ├── sample1.wav
    └── sample2.wav
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and go to: `http://localhost:5000`

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
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Dense(128) → Output
```

### Voice Recognition System
The system includes multiple model architectures:

#### Enhanced CNN Models
```
Input: Multi-feature audio (32×130×1 or 32×36×1)
- MFCC features (13): Spectral characteristics
- Chroma features (12): Pitch class profiles  
- Spectral Contrast (7): Timbral texture
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Dense(128) → Output
```

#### Specialized Models
- **Standard Model**: General-purpose voice recognition
- **Focused Models**: Person-specific optimization (e.g., Athul-focused)
- **Balanced Model**: Multi-user optimization
- **Ensemble Model**: Random Forest + SVM with statistical features

#### Dual-Model System
- Intelligent model switching based on confidence scores
- Consensus prediction from multiple models
- Enhanced accuracy through model combination

## Recent Improvements 🚀

### 🎨 Modern UI/UX Enhancements
- **Glassmorphism Design**: Modern glass-like UI with backdrop blur effects and transparency layers
- **Responsive Layout**: Fully responsive design that adapts to all screen sizes and devices
- **Advanced Animations**: Smooth CSS animations including fade-in, slide-up, pulse, and hover effects
- **Interactive Elements**: Enhanced hover states with transform effects and dynamic color changes
- **Modern Typography**: Inter font family for clean, professional appearance
- **Gradient Backgrounds**: Beautiful gradient overlays and dynamic visual elements
- **Card-Based Layout**: Organized content in modern card components with shadow effects
- **Professional Navigation**: Intuitive navigation with back buttons and smooth transitions

### 📱 Enhanced User Interface Features
- **Dedicated Pages**: Separate specialized pages for each functionality:
  - `face_recognition.html` - Face verification with live camera feed
  - `voice_recognition.html` - Voice verification with real-time audio processing
  - `train_face.html` - Face model training interface
  - `train_voice.html` - Voice model training interface
  - `settings.html` - System configuration and analytics dashboard
  - `results.html` - Recognition results display with confidence scores

### 🎯 Interactive UI Components
- **Live Camera Integration**: Real-time camera feed with capture functionality
- **Audio Visualization**: Dynamic microphone visualization with recording indicators
- **Progress Indicators**: Training progress bars and status animations
- **Drag & Drop Support**: Modern file upload with drag-and-drop functionality
- **Real-time Feedback**: Instant visual feedback for user actions
- **Status Indicators**: Clear visual states for recording, processing, and results

### Voice Recognition Enhancements
- **Multi-Feature Extraction**: Combined MFCC, Chroma, and Spectral Contrast features (32 total vs. 13 previously)
- **Advanced Model Architectures**: Multiple specialized models for different scenarios
- **Dual-Model System**: Intelligent switching between models based on confidence
- **Ensemble Methods**: Random Forest and SVM models for statistical analysis
- **Enhanced Preprocessing**: Improved normalization and feature-type specific processing
- **Robust Error Handling**: Comprehensive fallbacks for different librosa versions

### System Optimizations
- **Configuration Management**: Centralized config system for easy parameter tuning
- **Performance Analytics**: Detailed testing and validation tools
- **Memory Optimization**: Efficient model loading and prediction processes
- **JSON Serialization**: Fixed numpy type conversion issues
- **Sample Rate Consistency**: Standardized audio processing across all models

### Model Performance
- **Focused Models**: Up to 100% accuracy for specific individuals
- **Balanced Models**: Optimized for multi-user scenarios (60% overall accuracy)
- **Consensus Prediction**: Combines multiple model outputs for better reliability

## API Endpoints 🔌

- `GET /` - Main web interface with modern dashboard
- `POST /train` - Train models with current data
- `POST /recognize_face` - Recognize face from image
- `POST /recognize_voice` - Recognize voice from audio
- `GET /face-recognition` - Face verification interface
- `GET /voice-recognition` - Voice verification interface
- `GET /train-face` - Face model training interface
- `GET /train-voice` - Voice model training interface
- `GET /settings` - System configuration dashboard

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
├── app.py                          # Main Flask application with dual-model system
├── setup.py                       # Setup script
├── config.py                      # Configuration management
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
├── templates/                     # Modern UI Templates
│   ├── index.html                 # Main dashboard with glassmorphism design
│   ├── face_recognition.html      # Face verification interface
│   ├── voice_recognition.html     # Voice verification interface
│   ├── train_face.html           # Face model training interface
│   ├── train_voice.html          # Voice model training interface
│   ├── settings.html             # System configuration dashboard
│   ├── results.html              # Recognition results display
│   └── home.html                 # Alternative home interface
├── face_data/                     # Face training images
├── voice_data/                    # Voice training samples
├── Models & Encoders/             # Trained Models
│   ├── face_model.h5             # Trained face model
│   ├── face_encoder.pkl          # Face label encoder
│   ├── voice_model.h5            # Standard voice model
│   ├── voice_encoder.pkl         # Voice label encoder
│   ├── balanced_voice_model.h5   # Balanced voice model
│   ├── balanced_voice_encoder.pkl # Balanced model encoder
│   ├── focused_voice_model.h5    # Person-focused models
│   ├── voice_ensemble_models.pkl # Ensemble models
│   └── voice_features.pkl        # Processed voice features
├── Analysis & Testing Scripts/    # Development Tools
│   ├── test_system.py            # System validation
│   ├── validate_system.py        # Model testing
│   ├── analyze_all_voices.py     # Voice analysis tools
│   ├── compare_all_models.py     # Model comparison
│   ├── improve_all_voices.py     # Voice improvement scripts
│   ├── fix_voice_confusion.py    # Confusion resolution
│   └── final_voice_optimization.py # Ensemble optimization
├── Documentation/                 # Project Documentation
│   ├── FINAL_PROJECT_STATUS.md   # Project status report
│   ├── IMPROVEMENTS_SUMMARY.md   # Improvement documentation
│   └── USAGE_SUMMARY.md         # Usage guidelines
└── Output Logs/                  # Analysis Results
    ├── voice_test_output.txt     # Test results
    ├── analysis_output.txt       # Analysis logs
    └── improvement_output.txt    # Improvement logs
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
