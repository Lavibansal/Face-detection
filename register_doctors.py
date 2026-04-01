import os
import pickle

import cv2
import numpy as np

KNOWN_DOCTORS_DIR = "Known_Doctors"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
OUTPUT_MODEL = "trainer.yml"
OUTPUT_LABELS = "labels.pkl"

if not os.path.isdir(KNOWN_DOCTORS_DIR):
    raise FileNotFoundError(
        f"Directory not found: {KNOWN_DOCTORS_DIR}.\n"
        "Create the folder or update KNOWN_DOCTORS_DIR to the correct path."
    )

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
if face_detector.empty():
    raise RuntimeError(f"Failed to load Haar cascade from {CASCADE_PATH}")

faces = []
labels = []
label_ids = {}
next_id = 0

print("[INFO] Preparing training data...")

for doctor_name in sorted(os.listdir(KNOWN_DOCTORS_DIR)):
    doctor_path = os.path.join(KNOWN_DOCTORS_DIR, doctor_name)
    if not os.path.isdir(doctor_path):
        continue

    if doctor_name not in label_ids:
        label_ids[doctor_name] = next_id
        next_id += 1

    for image_name in sorted(os.listdir(doctor_path)):
        image_path = os.path.join(doctor_path, image_name)
        if not os.path.isfile(image_path):
            continue

        print(f"[INFO] Processing {doctor_name} - {image_name}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Unable to read image: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(rects) == 0:
            print(f"[WARNING] No face found in {image_path}")
            continue

        x, y, w, h = rects[0]
        faces.append(gray[y:y+h, x:x+w])
        labels.append(label_ids[doctor_name])

if len(faces) == 0:
    raise RuntimeError("No faces were found in Known_Doctors. Add clear face images in subfolders.")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.write(OUTPUT_MODEL)

with open(OUTPUT_LABELS, "wb") as f:
    pickle.dump(label_ids, f)

print(f"[SUCCESS] Trained model saved to {OUTPUT_MODEL}")
print(f"[SUCCESS] Label mapping saved to {OUTPUT_LABELS}")
