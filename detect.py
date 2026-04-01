import os
import pickle
from datetime import datetime, timedelta

import cv2
import pandas as pd

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.pkl"
ATTENDANCE_FILE = "attendance.csv"

# Use 0 for laptop webcam. For phone camera, replace with the stream URL.
CAPTURE_SOURCE = 0
EXIT_TIMEOUT_SECONDS = 30
# LBPH confidence is lower for better matches. The live camera can return values up to 120 for real doctors,
# so increase this if valid doctors are still being marked Unknown.
CONFIDENCE_THRESHOLD = 120
MIN_CONSECUTIVE_FRAMES_FOR_ENTRY = 3
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
if face_detector.empty():
    raise RuntimeError(f"Failed to load Haar cascade from {CASCADE_PATH}")

if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(
        f"Missing model or labels. Run register_doctors.py first to create {MODEL_FILE} and {LABELS_FILE}."
    )

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

with open(LABELS_FILE, "rb") as f:
    label_ids = pickle.load(f)

labels = {value: key for key, value in label_ids.items()}

try:
    df = pd.read_csv(ATTENDANCE_FILE)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Name", "Entry", "Exit", "Duration"])

active_entries = {}
pending_entries = {}

# Use DirectShow on Windows for better webcam compatibility.
cap = cv2.VideoCapture(CAPTURE_SOURCE, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError(f"Unable to open video source: {CAPTURE_SOURCE}")

print("[INFO] Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to read frame from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_WIDTH, MIN_FACE_HEIGHT),
    )

    current_recognized = {}
    seen_names = set()

    for (x, y, w, h) in rects:
        face = gray[y : y + h, x : x + w]
        if face.size == 0:
            continue

        face = cv2.equalizeHist(face)
        label, confidence = recognizer.predict(face)
        name = "Unknown"

        if confidence < CONFIDENCE_THRESHOLD and label in labels:
            name = labels[label]
            current_recognized[name] = current_recognized.get(name, 0) + 1
            seen_names.add(name)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{name} ({int(confidence)})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    now = datetime.now()

    for name, count in current_recognized.items():
        if name in active_entries:
            active_entries[name]["last_seen"] = now
        else:
            pending_entries[name] = pending_entries.get(name, 0) + 1
            if pending_entries[name] >= MIN_CONSECUTIVE_FRAMES_FOR_ENTRY:
                active_entries[name] = {"entry": now, "last_seen": now}
                print(f"[ENTRY] {name} at {now.strftime('%H:%M:%S')}")
                pending_entries.pop(name, None)

    for name in list(pending_entries):
        if name not in current_recognized:
            pending_entries.pop(name, None)

    for name in list(active_entries):
        if name not in seen_names:
            last_seen = active_entries[name]["last_seen"]
            if now - last_seen > timedelta(seconds=EXIT_TIMEOUT_SECONDS):
                entry_time = active_entries[name]["entry"]
                exit_time = last_seen
                duration = exit_time - entry_time
                print(
                    f"[EXIT] {name} at {exit_time.strftime('%H:%M:%S')} | Duration: {duration}"
                )
                df.loc[len(df)] = [
                    name,
                    entry_time.strftime("%H:%M:%S"),
                    exit_time.strftime("%H:%M:%S"),
                    str(duration),
                ]
                df.to_csv(ATTENDANCE_FILE, index=False)
                del active_entries[name]

    cv2.imshow("Doctor Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
