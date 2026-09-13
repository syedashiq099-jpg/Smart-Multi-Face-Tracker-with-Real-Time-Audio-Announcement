import cv2
import time
import math
import pyttsx3
import threading

# Initialize Voice Engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Bolne ki speed

def speak(text):
    """Voice announcement in background thread so camera doesn't freeze"""
    def run_speech():
        try:
            # Har thread ke liye safe instance
            local_engine = pyttsx3.init()
            local_engine.setProperty('rate', 160)
            local_engine.say(text)
            local_engine.runAndWait()
        except Exception:
            pass

    t = threading.Thread(target=run_speech, daemon=True)
    t.start()

# Load Haar Cascade
haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if haar_cascade.empty():
    print("Error: Haar Cascade file not found.")
    exit()

# Initialize camera (0 for inbuilt webcam, 1 for external USB)
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

# Tracking variables
next_id = 1
tracked_faces = {}
max_distance = 80
max_missing_frames = 10

# FPS & Voice Alert Timers
prev_time = 0
last_spoken_count = -1
last_spoken_time = 0
speech_delay = 2.5  # Har 2.5 second baad naya count bolega (agar count badla ho)

while True:
    success, img = cam.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade.detectMultiScale(
        grayImg,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40)
    )

    current_faces = []

    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2

        current_faces.append({
            "box": (x, y, w, h),
            "center": (center_x, center_y)
        })

    # Centroid Tracking Logic
    used_ids = set()

    for face in current_faces:
        x, y, w, h = face["box"]
        center_x, center_y = face["center"]

        best_id = None
        best_distance = max_distance

        for face_id, data in tracked_faces.items():
            if face_id in used_ids:
                continue

            old_x, old_y = data["center"]
            distance = math.sqrt((center_x - old_x) ** 2 + (center_y - old_y) ** 2)

            if distance < best_distance:
                best_distance = distance
                best_id = face_id

        if best_id is not None:
            face_id = best_id
            used_ids.add(face_id)
            tracked_faces[face_id]["center"] = (center_x, center_y)
            tracked_faces[face_id]["box"] = (x, y, w, h)
            tracked_faces[face_id]["missing"] = 0
        else:
            face_id = next_id
            next_id += 1
            tracked_faces[face_id] = {
                "center": (center_x, center_y),
                "box": (x, y, w, h),
                "missing": 0
            }
            used_ids.add(face_id)

        # Draw box and labels
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"ID: {face_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(img, (center_x, center_y), 4, (0, 0, 255), -1)

    # Missing frames update
    for face_id in list(tracked_faces.keys()):
        if face_id not in used_ids:
            tracked_faces[face_id]["missing"] += 1
            if tracked_faces[face_id]["missing"] > max_missing_frames:
                del tracked_faces[face_id]

    face_count = len(faces)
    current_time = time.time()

    # Voice Announcement Logic (Bolne ka logic)
    if (face_count != last_spoken_count or (current_time - last_spoken_time > 5)) and face_count > 0:
        if (current_time - last_spoken_time) > speech_delay:
            message = f"{face_count} face detected" if face_count == 1 else f"{face_count} faces detected"
            speak(message)
            last_spoken_count = face_count
            last_spoken_time = current_time
    elif face_count == 0:
        last_spoken_count = 0

    # FPS Calculation
    if prev_time != 0:
        fps = 1 / (current_time - prev_time)
    else:
        fps = 0
    prev_time = current_time

    # UI Information
    cv2.putText(img, f"Faces: {face_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, f"FPS: {fps:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, f"Tracked IDs: {len(tracked_faces)}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Detection & Tracking with Voice Alert", img)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
