# Face & Voice Recognition System 🎭

A modern web-based face and voice recognition system using CNN (Convolutional Neural Networks) with a beautiful, responsive UI.

## Features ✨

- **Face Recognition**: CNN-based facial recognition using computer vision
- **Voice Recognition**: MFCC feature extraction with CNN for voice identification
- **Real-time Processing**: Live camera feed and microphone input
- **Modern Web UI**: Responsive design with drag-and-drop file uploads
- **Model Training**: Train custom models with your own data
- **Multi-format Support**: Images (JPG, PNG) and audio files (WAV, MP3)

## Technologies Used 🔧

- **Backend**: Python Flask, TensorFlow/Keras, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: CNN models for both face and voice recognition
- **Audio Processing**: Librosa for feature extraction
- **Computer Vision**: OpenCV for face detection and processing

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

### Voice Recognition CNN
```
Input: MFCC features (13×130)
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Dense(128) → Output
```

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

3. **Model Training**:
   - More data generally improves accuracy
   - Balance the number of samples per person
   - Retrain models when adding new people

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
├── app.py                 # Main Flask application
├── setup.py              # Setup script
├── requirements.txt      # Dependencies
├── README.md            # This file
├── templates/
│   └── index.html       # Web interface
├── face_data/           # Face training images
├── voice_data/          # Voice training samples
├── face_model.h5        # Trained face model (generated)
├── voice_model.h5       # Trained voice model (generated)
├── face_encoder.pkl     # Face label encoder (generated)
└── voice_encoder.pkl    # Voice label encoder (generated)
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
