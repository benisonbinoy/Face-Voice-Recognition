import os
from collections import defaultdict
import cv2
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd

person_features = defaultdict(list)
voice_features = {}
orb = cv2.ORB_create()

def load_person_features():
    face_data = "face_data"
    for person_name in os.listdir(face_data):
        person_folder = os.path.join(face_data, person_name)
        if os.path.isdir(person_folder):
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    keypoints, descriptors = orb.detectAndCompute(image, None)
                    if descriptors is not None:
                        person_features[person_name].append(descriptors)

def load_voice_features():
    encoder = VoiceEncoder()
    voice_data = "voice_data"
    for person_name in os.listdir(voice_data):
        person_folder = os.path.join(voice_data, person_name)
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

def recognize_person(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        best_match_count = 0
        recognized_person = "Unknown"
        for person, feature_sets in person_features.items():
            for features in feature_sets:
                try:
                    matches = bf.match(descriptors, features)
                    matches = sorted(matches, key=lambda x: x.distance)
                    if len(matches) > best_match_count:
                        best_match_count = len(matches)
                        recognized_person = person
                except:
                    continue
        return recognized_person
    return "No face detected"

def recognize_speaker_realtime(encoder, duration=2, fs=16000):
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

def main():
    print("Loading face features...")
    load_person_features()
    print("Loading voice features...")
    load_voice_features()
    encoder = VoiceEncoder()
    print("Loaded the voice encoder\n")
    cap = cv2.VideoCapture(0)
    window_name = 'Face & Voice Recognition'
    cv2.namedWindow(window_name)
    speaker_name = "Press 'v' for voice"
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        person_name = recognize_person(frame)
        cv2.putText(frame, f"Face: {person_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Voice: {speaker_name}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord('v'):
            speaker_name = recognize_speaker_realtime(encoder, duration=2)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()