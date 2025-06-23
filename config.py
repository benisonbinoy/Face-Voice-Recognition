# Configuration settings for Face & Voice Recognition System

# Model settings
FACE_MODEL_CONFIG = {
    'input_shape': (128, 128, 1),
    'confidence_threshold': 0.7,
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001
}

VOICE_MODEL_CONFIG = {
    'input_shape': (13, 130, 1),
    'confidence_threshold': 0.6,
    'batch_size': 16,
    'epochs': 30,
    'learning_rate': 0.001,
    'n_mfcc': 13,
    'max_time_steps': 130,
    'sample_rate': 16000,
    'audio_duration': 3.0
}

# Data paths
DATA_PATHS = {
    'face_data': 'face_data',
    'voice_data': 'voice_data',
    'face_model': 'face_model.h5',
    'voice_model': 'voice_model.h5',
    'face_encoder': 'face_encoder.pkl',
    'voice_encoder': 'voice_encoder.pkl'
}

# Web server settings
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}

# Camera settings
CAMERA_CONFIG = {
    'width': 640,
    'height': 480,
    'fps': 30
}

# Audio recording settings
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'channels': 1,
    'chunk_size': 1024,
    'record_duration': 3.0
}
