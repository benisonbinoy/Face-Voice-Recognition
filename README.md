# Face & Voice Recognition System 🎭

A modern web-based face and voice recognition system using advanced CNN (Convolutional Neural Networks) with enhanced feature extraction and multiple model architectures for improved accuracy.

> **🚀 Version 1.5 Update**: Major overhaul with 32-feature voice recognition, dual-model system, person-specific models achieving 100% accuracy, and 20+ new testing tools for comprehensive validation.

## Features ✨

### 🎭 Core Recognition Capabilities
- **Advanced Face Recognition**: CNN-based facial recognition using computer vision
- **Revolutionary Voice Recognition**: Multi-feature extraction with 32 combined features (MFCC, Chroma, Spectral Contrast)
- **Dual-Model System**: Intelligent model switching and consensus prediction for maximum accuracy
- **Real-time Processing**: Live camera feed and microphone input with instant results

### 🧠 Multiple AI Model Architectures
- **Standard CNN Models**: General-purpose recognition for both face and voice
- **Focused Models**: Person-specific optimization achieving up to 100% accuracy for individuals
- **Balanced Models**: Multi-user optimization with 60% overall accuracy
- **Ensemble Models**: Random Forest + SVM with statistical features for robust fallback

### 🎨 User Experience
- **Modern Web UI**: Responsive design with drag-and-drop file uploads
- **Multi-format Support**: Images (JPG, PNG) and audio files (WAV, MP3)
- **Progress Tracking**: Real-time training progress and detailed result analytics
- **Confidence Scoring**: Transparent confidence levels for all predictions

### 🔧 Advanced Technical Features
- **Robust Feature Extraction**: Enhanced audio processing with comprehensive fallback support
- **Configuration Management**: Centralized config system for easy parameter tuning
- **Performance Analytics**: 20+ specialized testing and validation tools
- **Model Training**: Train multiple custom models with your own data
- **Memory Optimization**: Efficient processing for production environments

## Technologies Used 🔧

- **Backend**: Python Flask, TensorFlow/Keras, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: Advanced CNN models with multiple architectures
- **Audio Processing**: Librosa with enhanced feature extraction (MFCC, Chroma, Spectral Contrast)
- **Computer Vision**: OpenCV for face detection and processing
- **Model Optimization**: Ensemble methods, Random Forest, SVM
- **Configuration Management**: Centralized config system

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

### 1. Training Models
- Click "Train Models" button in the web interface
- The system will automatically process your face_data and voice_data
- Training progress will be displayed with a progress bar
- Models are saved automatically for future use

### 2. Face Recognition
- **Live Camera**: Click "Start Camera" → "Capture Photo"
- **File Upload**: Drag and drop an image or click to browse
- Results show recognized person with confidence score

### 3. Voice Recognition
- **Live Recording**: Click "Start Recording" → speak → "Stop Recording"
- **File Upload**: Drag and drop an audio file or click to browse
- Results show recognized speaker with confidence score

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

## Version History & Key Updates 📈

### Version 1.5 (Current) - Major Voice Recognition Overhaul
**Released: July 2025**

#### 🚀 Major Enhancements

**Advanced Voice Recognition System**
- **Multi-Feature Audio Processing**: Upgraded from 13 MFCC features to 32 combined features
  - MFCC (13): Spectral characteristics and voice timbre
  - Chroma (12): Pitch class profiles and harmonic content
  - Spectral Contrast (7): Timbral texture differences
- **Dual-Model Recognition System**: Intelligent model switching and consensus prediction
- **Multiple Specialized Models**:
  - Standard CNN Model: General-purpose recognition
  - Focused Models: Person-specific optimization (e.g., Athul-focused with 100% accuracy)
  - Balanced Model: Multi-user optimization
  - Ensemble Model: Random Forest + SVM with statistical features

**Enhanced Model Architecture**
- Updated CNN architecture with better capacity and regularization
- Feature-type specific normalization for better discrimination
- Early stopping and learning rate reduction for optimal training
- Robust error handling and fallback support for different librosa versions

**System Optimizations**
- **Configuration Management**: Centralized config system for easy parameter tuning
- **Performance Analytics**: Comprehensive testing and validation tools (20+ new test files)
- **Memory Optimization**: Efficient model loading and prediction processes
- **JSON Serialization Fixes**: Resolved numpy type conversion issues
- **Sample Rate Consistency**: Standardized to 22050 Hz across all models

#### 🔧 Technical Improvements

**New Analysis & Testing Tools**
- `analyze_all_voices.py`: Comprehensive voice analysis
- `compare_all_models.py`: Model performance comparison
- `balanced_voice_system.py`: Balanced multi-user training
- `final_voice_optimization.py`: Advanced optimization techniques
- `validate_system.py`: System validation and testing
- Multiple specialized test files for different scenarios

**Enhanced Processing**
- Higher sample rate (22050 Hz vs 16000 Hz) for better audio quality
- Improved padding and normalization strategies
- Voice activity detection and pitch analysis
- Robust feature extraction with comprehensive error handling

#### 📊 Performance Results
- **Focused Models**: Up to 100% accuracy for specific individuals
- **Balanced Models**: 60% overall accuracy optimized for multi-user scenarios
- **Consensus Prediction**: Combines multiple model outputs for better reliability
- **Reduced Misclassification**: Better discrimination between similar voices

### Version 1.0 - Initial Release
**Basic Features**
- Simple face recognition using CNN
- Basic voice recognition with MFCC features only
- Single model approach
- Basic web interface
- Limited testing capabilities

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
Hey, Cortana. Honestly, it's a big creepy. Hey, Cortana. 
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

### 1. Training Models
- Click "Train Models" button in the web interface
- The system will automatically process your face_data and voice_data
- Training progress will be displayed with a progress bar
- Models are saved automatically for future use

### 2. Face Recognition
- **Live Camera**: Click "Start Camera" → "Capture Photo"
- **File Upload**: Drag and drop an image or click to browse
- Results show recognized person with confidence score

### 3. Voice Recognition
- **Live Recording**: Click "Start Recording" → speak → "Stop Recording"
- **File Upload**: Drag and drop an audio file or click to browse
- Results show recognized speaker with confidence score

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

- `GET /` - Main web interface
- `POST /train` - Train models with current data
- `POST /recognize_face` - Recognize face from image
- `POST /recognize_voice` - Recognize voice from audio

## Performance Tips 💡

1. **Face Recognition**:
   - Use well-lit, clear images
   - Include various angles and expressions
   - Ensure faces are clearly visible

2. **Voice Recognition**:
   - Record in quiet environments
   - Use consistent microphone/recording setup
   - Include various speaking styles
   - Ensure 2-5 second audio samples for best results

3. **Model Training**:
   - More data generally improves accuracy
   - Balance the number of samples per person
   - Use multiple model types for different scenarios
   - Retrain models when adding new people
   - Consider person-specific focused models for difficult cases

4. **System Optimization**:
   - The dual-model system automatically selects the best approach
   - Ensemble methods provide fallback for edge cases
   - Enhanced feature extraction improves discrimination

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
├── templates/
│   └── index.html                 # Web interface
├── face_data/                     # Face training images
├── voice_data/                    # Voice training samples
├── face_model.h5                  # Trained face model
├── face_encoder.pkl               # Face label encoder
├── voice_model.h5                 # Standard voice model
├── voice_encoder.pkl              # Voice label encoder
├── balanced_voice_model.h5        # Balanced voice model
├── balanced_voice_encoder.pkl     # Balanced model encoder
├── focused_voice_model.h5         # Person-focused models
├── voice_ensemble_models.pkl      # Ensemble models
├── voice_features.pkl             # Processed voice features
├── test_system.py                 # System validation
├── validate_system.py             # Model testing
├── analyze_all_voices.py          # Voice analysis tools
├── compare_all_models.py          # Model comparison
├── FINAL_PROJECT_STATUS.md        # Project status report
├── IMPROVEMENTS_SUMMARY.md        # Improvement documentation
└── USAGE_SUMMARY.md              # Usage guidelines
Hey, Cortana. ```

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
