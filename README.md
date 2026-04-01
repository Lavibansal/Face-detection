# Face Attendance System

A simple face recognition attendance system built with Python, OpenCV, and Flask.

## Features

- Train face recognition from `Known_Doctors/` image folders
- Real-time webcam attendance logging with OpenCV LBPH recognizer
- Flask dashboard to view attendance logs
- Attendance stored in `attendance.csv`

## Requirements

- Python 3.8+
- `opencv-contrib-python`
- `pandas`
- `flask`

## Setup

1. Create and activate your virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install opencv-contrib-python pandas flask
   ```

3. Train the recognizer from the known doctor images:

   ```powershell
   python register_doctors.py
   ```

## Run the app

- Start the Flask dashboard:

  ```powershell
  python app.py
  ```

  Then open `http://127.0.0.1:5000` in your browser.

- Run webcam attendance detection:

  ```powershell
  python detect.py
  ```

  Press `Esc` to stop.

## Files

- `app.py` - Flask dashboard for attendance logs
- `detect.py` - Real-time face recognition and attendance capture
- `register_doctors.py` - Train known face labels from `Known_Doctors/`
- `Known_Doctors/` - Directory for doctor image folders
- `attendance.csv` - Generated attendance log
- `templates/index.html` - Dashboard HTML template

## Notes

- If you do not have a working webcam, you can still use the dashboard to view existing `attendance.csv` data.
- Make sure `trainer.yml` and `labels.pkl` are created before running `detect.py`.

## GitHub

This project can be pushed to `https://github.com/Lavibansal/Face-detection`.
