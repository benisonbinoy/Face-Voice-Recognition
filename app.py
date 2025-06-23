import os
import pickle
import numpy as np
import cv2
import librosa
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sounddevice as sd
import threading
import time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class FaceVoiceRecognitionSystem:
    def __init__(self):
        self.face_model = None
        self.voice_model = None
        self.face_encoder = LabelEncoder()
        self.voice_encoder = LabelEncoder()
        self.face_data = []
        self.voice_data = []
        self.face_labels = []
        self.voice_labels = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.voice_people_count = 0
        self.face_people_count = 0
        
    def load_face_data(self, data_dir="face_data"):
        """Load and preprocess face images using CNN"""
        print("Loading face data...")
        face_images = []
        labels = []
        
        for person_name in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_name)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Detect faces and resize
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        for (x, y, w, h) in faces:
                            face = gray[y:y+h, x:x+w]
                            face = cv2.resize(face, (128, 128))
                            face = face.reshape(128, 128, 1)
                            face_images.append(face)
                            labels.append(person_name)
        
        if face_images:
            self.face_data = np.array(face_images, dtype='float32') / 255.0
            self.face_labels = self.face_encoder.fit_transform(labels)
            print(f"Loaded {len(self.face_data)} face images for {len(set(labels))} people")
        
    def load_voice_data(self, data_dir="voice_data"):
        """Load and preprocess voice data"""
        print("Loading voice data...")
        voice_features = []
        labels = []
        
        for person_name in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_name)
            if os.path.isdir(person_path):
                for wav_file in os.listdir(person_path):
                    wav_path = os.path.join(person_path, wav_file)
                    try:
                        # Extract MFCC features
                        y, sr = librosa.load(wav_path, sr=16000, duration=3.0)
                        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
                        mfccs = mfccs[:, :130]  # Standardize to 130 time steps
                        voice_features.append(mfccs)
                        labels.append(person_name)
                    except Exception as e:
                        print(f"Error processing {wav_path}: {e}")
        
        if voice_features:
            self.voice_data = np.array(voice_features)
            self.voice_labels = self.voice_encoder.fit_transform(labels)
            print(f"Loaded {len(self.voice_data)} voice samples for {len(set(labels))} people")
    
    def create_face_cnn_model(self, num_classes):
        """Create CNN model for face recognition"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def create_voice_cnn_model(self, num_classes):
        """Create CNN model for voice recognition (fixed for small input size)"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(13, 130, 1), padding='same'),
            MaxPooling2D((1, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((1, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def train_models(self):
        """Train both face and voice recognition models"""
        # Train face model
        if len(self.face_data) > 0:
            print("Training face recognition model...")
            num_face_classes = len(np.unique(self.face_labels))
            self.face_model = self.create_face_cnn_model(num_face_classes)
            
            y_face_categorical = to_categorical(self.face_labels, num_face_classes)
            X_train, X_test, y_train, y_test = train_test_split(
                self.face_data, y_face_categorical, test_size=0.2, random_state=42
            )
            
            self.face_model.fit(X_train, y_train, epochs=20, batch_size=32, 
                              validation_data=(X_test, y_test), verbose=1)
            
            # Save model
            self.face_model.save('face_model.h5')
            with open('face_encoder.pkl', 'wb') as f:
                pickle.dump(self.face_encoder, f)
          # Train voice model
        if len(self.voice_data) > 0:
            num_voice_classes = len(np.unique(self.voice_labels))
            
            if num_voice_classes > 1:
                print("Training voice recognition model...")
                self.voice_model = self.create_voice_cnn_model(num_voice_classes)
                
                voice_data_reshaped = self.voice_data.reshape(-1, 13, 130, 1)
                y_voice_categorical = to_categorical(self.voice_labels, num_voice_classes)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    voice_data_reshaped, y_voice_categorical, test_size=0.2, random_state=42
                )
                
                self.voice_model.fit(X_train, y_train, epochs=30, batch_size=16,
                                   validation_data=(X_test, y_test), verbose=1)
                
                # Save model
                self.voice_model.save('voice_model.h5')
                with open('voice_encoder.pkl', 'wb') as f:
                    pickle.dump(self.voice_encoder, f)
            else:
                print(f"✅ Voice data loaded for {num_voice_classes} person: {self.voice_encoder.classes_[0]}")
                print("   Voice recognition will work when you add more people.")
                print("   Creating simple voice template for future use...")
                
                # Create a simple template model that can be expanded later
                self.voice_model = self.create_voice_cnn_model(max(2, num_voice_classes))
                
                # Train with dummy data to prepare the model architecture
                voice_data_reshaped = self.voice_data.reshape(-1, 13, 130, 1)
                  # Create temporary expanded dataset for training
                if num_voice_classes == 1:
                    # Add dummy "unknown" samples for training (same number as real samples)
                    num_samples = voice_data_reshaped.shape[0]
                    dummy_voice = np.random.normal(0, 0.1, voice_data_reshaped.shape)
                    expanded_voice_data = np.concatenate([voice_data_reshaped, dummy_voice])
                    expanded_labels = np.concatenate([self.voice_labels, np.ones(num_samples)])  # 1 for "unknown"
                    
                    y_voice_categorical = to_categorical(expanded_labels, 2)
                    
                    # Train with minimal epochs just to establish the model
                    self.voice_model.fit(expanded_voice_data, y_voice_categorical, 
                                       epochs=5, batch_size=16, verbose=1)
                
                # Save model and encoder for future expansion
                self.voice_model.save('voice_model.h5')
                with open('voice_encoder.pkl', 'wb') as f:
                    pickle.dump(self.voice_encoder, f)
                    
                print("   Voice model template saved. Add more people to enable full recognition.")
        else:
            print("⚠️  No voice data found for training.")
    
    def retrain_voice_model(self):
        """Retrain voice model when new voice data is added"""
        print("Retraining voice recognition model...")
        self.load_voice_data()
        
        if len(self.voice_data) > 0:
            num_voice_classes = len(np.unique(self.voice_labels))
            
            if num_voice_classes > 1:
                print(f"Training voice model with {num_voice_classes} people...")
                self.voice_model = self.create_voice_cnn_model(num_voice_classes)
                
                voice_data_reshaped = self.voice_data.reshape(-1, 13, 130, 1)
                y_voice_categorical = to_categorical(self.voice_labels, num_voice_classes)
                
                if len(voice_data_reshaped) > 2:  # Need at least some data for train/test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        voice_data_reshaped, y_voice_categorical, test_size=0.2, random_state=42
                    )
                    
                    self.voice_model.fit(X_train, y_train, epochs=30, batch_size=16,
                                       validation_data=(X_test, y_test), verbose=1)
                else:
                    # Train on all data if we don't have enough for splitting
                    self.voice_model.fit(voice_data_reshaped, y_voice_categorical, 
                                       epochs=20, batch_size=16, verbose=1)
                  # Save updated model
                self.voice_model.save('voice_model.h5')
                with open('voice_encoder.pkl', 'wb') as f:
                    pickle.dump(self.voice_encoder, f)
                    
                print("✅ Voice model retrained successfully!")
                return True
            else:
                print("Still only one person - voice recognition limited")
                return False
        else:
            print("No voice data found")
            return False
    
    def load_trained_models(self):
        """Load pre-trained models"""
        try:
            if os.path.exists('face_model.h5'):
                self.face_model = load_model('face_model.h5')
                with open('face_encoder.pkl', 'rb') as f:
                    self.face_encoder = pickle.load(f)
                print("Face model loaded successfully")
            
            if os.path.exists('voice_model.h5'):
                self.voice_model = load_model('voice_model.h5')
                with open('voice_encoder.pkl', 'rb') as f:
                    self.voice_encoder = pickle.load(f)
                print("Voice model loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def recognize_face(self, image):
        """Recognize face from image"""
        if self.face_model is None:
            return "No face model loaded"
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        recognized_names = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128))
            face = face.reshape(1, 128, 128, 1)
            face = face.astype('float32') / 255.0
            
            prediction = self.face_model.predict(face, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            if confidence > 0.7:  # Confidence threshold
                name = self.face_encoder.inverse_transform([predicted_class])[0]
                recognized_names.append(f"{name} ({confidence:.2f})")
            else:
                recognized_names.append("Unknown")
        
        return recognized_names if recognized_names else ["No faces detected"]
    
    def recognize_voice(self, audio_data):
        """Recognize voice from audio data with enhanced multi-person support"""
        if self.voice_model is None:
            return {
                "name": "Error",
                "confidence": 0.0,
                "mode": "error",
                "message": "Voice model not available (add more people to enable)",
                "status": "error"
            }
        
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
            mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 130 - mfccs.shape[1]))), mode='constant')
            mfccs = mfccs[:, :130]
            mfccs = mfccs.reshape(1, 13, 130, 1)
            
            prediction = self.voice_model.predict(mfccs, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Enhanced multi-person recognition logic
            num_people = len(self.voice_encoder.classes_)
            
            if num_people == 1:
                # Single person mode - more lenient threshold
                if confidence > 0.4:
                    name = self.voice_encoder.classes_[0]
                    return {
                        "name": name,
                        "confidence": confidence,
                        "mode": "single_person",
                        "message": f"✅ {name} (Single person mode - {confidence:.2%} confidence)",
                        "status": "success"
                    }
                else:
                    return {
                        "name": "Unknown",
                        "confidence": confidence,
                        "mode": "single_person", 
                        "message": f"❓ Voice not recognized (Low confidence: {confidence:.2%})",
                        "status": "unknown"
                    }
            else:
                # Multi-person mode - stricter threshold
                if confidence > 0.6:
                    name = self.voice_encoder.inverse_transform([predicted_class])[0]
                    return {
                        "name": name,
                        "confidence": confidence,
                        "mode": "multi_person",
                        "message": f"✅ {name} (Multi-person mode - {confidence:.2%} confidence)",
                        "status": "success"
                    }
                elif confidence > 0.4:
                    name = self.voice_encoder.inverse_transform([predicted_class])[0]
                    return {
                        "name": name,
                        "confidence": confidence,
                        "mode": "multi_person",
                        "message": f"⚠️ Possibly {name} (Low confidence: {confidence:.2%})",
                        "status": "uncertain"
                    }
                else:
                    return {
                        "name": "Unknown",
                        "confidence": confidence,
                        "mode": "multi_person",
                        "message": f"❌ Voice not recognized (Very low confidence: {confidence:.2%})",
                        "status": "unknown"
                    }
                    
        except Exception as e:
            return {
                "name": "Error",
                "confidence": 0.0,
                "mode": "error",
                "message": f"Voice recognition error: {str(e)}",
                "status": "error"
            }
    
    def get_system_status(self):
        """Get current system status for both face and voice recognition"""
        # Count people in face_data
        face_people = []
        if os.path.exists("face_data"):
            face_people = [name for name in os.listdir("face_data") 
                          if os.path.isdir(os.path.join("face_data", name))]
        
        # Count people in voice_data  
        voice_people = []
        if os.path.exists("voice_data"):
            voice_people = [name for name in os.listdir("voice_data") 
                           if os.path.isdir(os.path.join("voice_data", name))]
        
        self.face_people_count = len(face_people)
        self.voice_people_count = len(voice_people)
        
        status = {
            "face_recognition": {
                "enabled": self.face_model is not None,
                "people_count": self.face_people_count,
                "people_names": face_people,
                "status": "ready" if self.face_model is not None else "needs_training"
            },
            "voice_recognition": {
                "enabled": self.voice_model is not None,
                "people_count": self.voice_people_count,
                "people_names": voice_people,
                "status": "full_recognition" if self.voice_people_count > 1 else "limited_recognition" if self.voice_people_count == 1 else "needs_training"
            }
        }
        
        return status
    
    def get_voice_data_info(self):
        """Get detailed information about voice data"""
        voice_info = {}
        if os.path.exists("voice_data"):
            for person_name in os.listdir("voice_data"):
                person_path = os.path.join("voice_data", person_name)
                if os.path.isdir(person_path):
                    files = [f for f in os.listdir(person_path) if f.endswith(('.wav', '.mp3', '.m4a'))]
                    voice_info[person_name] = {
                        "sample_count": len(files),
                        "files": files
                    }
        return voice_info

# Initialize the recognition system
recognition_system = FaceVoiceRecognitionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_models():
    """Train the models with current data"""
    try:
        recognition_system.load_face_data()
        recognition_system.load_voice_data()
        recognition_system.train_models()
        return jsonify({"status": "success", "message": "Models trained successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    """Recognize face from uploaded image"""
    try:
        data = request.json
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = recognition_system.recognize_face(image)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/recognize_voice', methods=['POST'])
def recognize_voice():
    """Recognize voice from uploaded audio"""
    try:
        audio_file = request.files['audio']
        audio_data, sr = librosa.load(audio_file, sr=16000)
        
        result = recognition_system.recognize_voice(audio_data)
        
        # Handle both old string format and new dict format
        if isinstance(result, dict):
            return jsonify({"status": "success", "result": result})
        else:
            # Backward compatibility for old string format
            return jsonify({"status": "success", "result": {"message": result}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/recognize_face_realtime', methods=['POST'])
def recognize_face_realtime():
    """Recognize face from camera frame in real-time"""
    try:
        data = request.json
        print("Received face recognition request...")  # Debug log
        
        # Handle base64 image data
        if 'image' not in data:
            return jsonify({"status": "error", "message": "No image data received"})
        
        img_data_str = data['image']
        if img_data_str.startswith('data:image'):
            img_data_str = img_data_str.split(',')[1]
        
        img_data = base64.b64decode(img_data_str)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"status": "error", "message": "Failed to decode image"})
        
        print(f"Image shape: {image.shape}")  # Debug log
        
        result = recognition_system.recognize_face(image)
        print(f"Recognition result: {result}")  # Debug log
        
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        print(f"Error in face recognition: {str(e)}")  # Debug log
        return jsonify({"status": "error", "message": str(e)})

@app.route('/retrain_voice', methods=['POST'])
def retrain_voice():
    """Retrain voice recognition model when new voice data is added"""
    try:
        success = recognition_system.retrain_voice_model()
        if success:
            return jsonify({"status": "success", "message": "Voice model retrained successfully"})
        else:
            return jsonify({"status": "info", "message": "Voice model updated but still needs more people"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_system_status', methods=['GET'])
def get_system_status():
    """Get current system status"""
    try:
        status = recognition_system.get_system_status()
        return jsonify({"status": "success", "data": status})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_voice_info', methods=['GET'])
def get_voice_info():
    """Get voice data information"""
    try:
        voice_info = recognition_system.get_voice_data_info()
        return jsonify({"status": "success", "data": voice_info})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Load existing models or train new ones
    recognition_system.load_trained_models()
    
    # If no models exist, load data and train
    if recognition_system.face_model is None or recognition_system.voice_model is None:
        print("No trained models found. Training new models...")
        recognition_system.load_face_data()
        recognition_system.load_voice_data()
        recognition_system.train_models()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
