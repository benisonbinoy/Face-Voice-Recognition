
    face_data = "face_data"
    print("Loading known faces...")
    known_encodings, known_names = load_face_encodings(face_data)
    print(f"Loaded {len(known_encodings)} face encodings.")