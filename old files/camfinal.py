import cv2
import subprocess
import os
import numpy as np
from collections import deque

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize camera
cap = cv2.VideoCapture(0)

# Set thresholds for detection
left_threshold = 100
right_threshold = 100
center_threshold = 50
similarity_threshold = 5000  # Threshold for matching target face
rotation_limit = 90  # Motor rotation limit in degrees (half of 180)
cumulative_rotation = 0

# Path to save the target face
TARGET_FACE_PATH = 'target_face.jpg'
target_face = None
face_saved = False

# Initialize position queue for smoothing
position_queue = deque(maxlen=5)

# Start the motor control program
motor_process = subprocess.Popen(["./motorfinal"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def save_target_face(face_image):
    cv2.imwrite(TARGET_FACE_PATH, face_image)
    print("Target face saved.")

def load_target_face():
    if os.path.exists(TARGET_FACE_PATH):
        return cv2.imread(TARGET_FACE_PATH, cv2.IMREAD_GRAYSCALE)
    return None

# Load previously saved target face if available
target_face = load_target_face()
if target_face is not None:
    face_saved = True

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Define the frame center
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2

        best_match_value = float('inf')
        best_match_index = -1

        for i, (x, y, w, h) in enumerate(faces):
            detected_face = gray_frame[y:y + h, x:x + w]

            if not face_saved:
                # Save the first detected face as the target
                save_target_face(detected_face)
                target_face = detected_face
                face_saved = True

            if face_saved and target_face is not None:
                # Resize detected face to match the target face size
                detected_face_resized = cv2.resize(detected_face, (target_face.shape[1], target_face.shape[0]))

                # Calculate similarity (match value) between detected face and target face
                match_value = cv2.norm(target_face, detected_face_resized, cv2.NORM_L2)

                # Update best match if this face is closer to target than previous matches
                if match_value < best_match_value and match_value < similarity_threshold:
                    best_match_value = match_value
                    best_match_index = i

        # Second pass to label faces based on the best match
        for i, (x, y, w, h) in enumerate(faces):
            face_center_x = x + w // 2

            if i == best_match_index:
                # Label the closest matching face as "Target"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Target", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Smooth the position by adding it to the position queue
                position_queue.append(face_center_x)
                smoothed_center_x = int(np.mean(position_queue))

                # Motor control logic based on smoothed center position
                if smoothed_center_x < frame_center_x - left_threshold and cumulative_rotation > -rotation_limit:
                    print("Target on the left side")
                    motor_process.stdin.write("ccw\n")
                    motor_process.stdin.flush()
                    cumulative_rotation -= 1
                elif smoothed_center_x > frame_center_x + right_threshold and cumulative_rotation < rotation_limit:
                    print("Target on the right side")
                    motor_process.stdin.write("cw\n")
                    motor_process.stdin.flush()
                    cumulative_rotation += 1
                elif frame_center_x - center_threshold < smoothed_center_x < frame_center_x + center_threshold:
                    print("Target centered")
                    motor_process.stdin.write("stop\n")
                    motor_process.stdin.flush()
            else:
                # Label all other faces as "Unknown"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection with Motor Control', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except BrokenPipeError:
    print("Motor process terminated unexpectedly.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    if motor_process.poll() is None:
        motor_process.terminate()
