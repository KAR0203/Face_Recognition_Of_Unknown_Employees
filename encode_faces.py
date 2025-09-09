import os
import face_recognition
import pickle

EMPLOYEE_DIR = "data/employees"
ENCODINGS_FILE = "data/encodings.pkl"
known_encodings = []
known_names = []

for person_name in os.listdir(EMPLOYEE_DIR):
    person_folder = os.path.join(EMPLOYEE_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    for file in os.listdir(person_folder):
        file_path = os.path.join(person_folder, file)
       
        image = face_recognition.load_image_file(file_path)
       
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]
            known_encodings.append(encoding)
            known_names.append(person_name)
            print(f"[INFO] Encoded {file} for {person_name}")

data = {"encodings": known_encodings, "names": known_names}
os.makedirs("data", exist_ok=True)
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encodings saved to {ENCODINGS_FILE}")
