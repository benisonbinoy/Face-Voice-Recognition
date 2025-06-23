import os
import cv2
import numpy as np
import face_recognition
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
import pickle

# --------- FACE RECOGNITION SETUP ---------
def load_face_encodings(face_data_folder):
    known_encodings = []
    known_names = []
    for person_name in os.listdir(face_data_folder):
        person_folder = os.path.join(face_data_folder, person_name)
        if os.path.isdir(person_folder):
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
    return known_encodings, known_names

# --------- VOICE RECOGNITION SETUP ---------
def load_voice_features(voice_data_folder):
    encoder = VoiceEncoder()
    voice_features = {}
    for person_name in os.listdir(voice_data_folder):
        person_folder = os.path.join(voice_data_folder, person_name)
        if os.path.isdir(person_folder):
            embeddings = []
            for wav_name in os.listdir(person_folder):
                wav_path = os.path.join(person_folder, wav_name)
                try:
                    wav = preprocess_wav(wav_path)
                    embed = encoder.embed_utterance(wav)
                    embeddings.append(embed)
                except Exception as e:
                    print(f"Failed to process {wav_path}: {e}")
            if embeddings:
                voice_features[person_name] = np.mean(embeddings, axis=0)
    return encoder, voice_features

def recognize_speaker_realtime(encoder, voice_features, duration=2, fs=16000):
    try:
        print("Recording voice for recognition...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        wav = np.squeeze(recording)
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32) / np.max(np.abs(wav))
        embed = encoder.embed_utterance(wav)
        best_score = -1
        best_person = "Unknown"
        for person, ref_embed in voice_features.items():
            score = np.dot(embed, ref_embed) / (np.linalg.norm(embed) * np.linalg.norm(ref_embed))
            if score > best_score:
                best_score = score
                best_person = person
        print(f"Speaker recognized as: {best_person}")
        return best_person
    except Exception as e:
        print(f"Speaker recognition failed: {e}")
        return "Unknown"

# --------- MAIN PROGRAM ---------
def main():
    face_data = "face_data"
    voice_data = "voice_data"

    # --- FACE ENCODINGS CACHE ---
    if os.path.exists("face_encodings.pkl"):
        print("Loading cached face encodings...")
        with open("face_encodings.pkl", "rb") as f:
            known_encodings, known_names = pickle.load(f)
    else:
        print("Loading known faces...")
        known_encodings, known_names = load_face_encodings(face_data)
        with open("face_encodings.pkl", "wb") as f:
            pickle.dump((known_encodings, known_names), f)
    print(f"Loaded {len(known_encodings)} face encodings.")

    # --- VOICE FEATURES CACHE ---
    if os.path.exists("voice_features.pkl"):
        print("Loading cached voice features...")
        with open("voice_features.pkl", "rb") as f:
            voice_features = pickle.load(f)
        encoder = VoiceEncoder()  # Still need to instantiate encoder
    else:
        print("Loading known voices...")
        encoder, voice_features = load_voice_features(voice_data)
        with open("voice_features.pkl", "wb") as f:
            pickle.dump(voice_features, f)
    print(f"Loaded {len(voice_features)} voice profiles.")

    cap = cv2.VideoCapture(0)
    window_name = 'Face & Voice Recognition'
    cv2.namedWindow(window_name)
    speaker_name = "Press 'v' for voice"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        names_in_frame = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
            names_in_frame.append(name)

        # Draw boxes and names
        for (top, right, bottom, left), name in zip(face_locations, names_in_frame):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show speaker name
        cv2.putText(frame, f"Voice: {speaker_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord('v'):
            speaker_name = recognize_speaker_realtime(encoder, voice_features, duration=2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()