import cv2
import ctypes
import time
from picamera2 import Picamera2
import numpy as np

# Load the shared library
motor_lib = ctypes.CDLL('/home/imt/src/om/libmotor_control.so')

# Initialize GPIOs for motor control
if motor_lib.initGPIO() < 0:
    print("Failed to initialize GPIOs.")
    exit(1)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('/home/imt/src/car/haarcascade_frontalface_default.xml')

# Initialize Picamera2
picam2 = Picamera2()
picam2.start()

# Load the pre-trained LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to train the face recognizer with a sample image
def train_face_recognizer(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("No face detected in the training image.")
        exit(1)
    (x, y, w, h) = faces[0]
    face = image[y:y + h, x:x + w]  # Crop the face from the image
    labels = np.array([1])  # Label "1" for "Om"
    recognizer.train([face], labels)

# Train the recognizer with the "me.jpg" image
train_face_recognizer('/home/imt/src/om/me.jpg')

# Set thresholds for detection
left_threshold = 150
right_threshold = 150
center_threshold = 30
no_face_time_limit = 10  # 10 seconds without face detection before rotating motor
last_face_detection_time = time.time()  # Track the last face detection time

# Function to smoothly move the motor
def smooth_motor_move(direction, duration=1, step_time=0.1):
    steps = int(duration / step_time)
    for _ in range(steps):
        motor_lib.runMotor(direction)
        time.sleep(step_time)
    motor_lib.stopMotor()

try:
    while True:
        # Capture frame from the camera
        frame = picam2.capture_array()

        # If you want to work with grayscale, convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame (which is appropriate for face detection)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Define the frame center
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2

        # Variables to store the best face and its confidence
        best_face = None
        best_confidence = float('inf')
        best_label = None

        # Loop through all detected faces to find the best match
        for (x, y, w, h) in faces:
            face_center_x = x + w // 2
            face = gray_frame[y:y + h, x:x + w]

            # Try to recognize the face using the trained model
            label, confidence = recognizer.predict(face)
            if confidence < best_confidence:  # Choose the face with the lowest confidence (most reliable)
                best_face = (x, y, w, h)
                best_confidence = confidence
                best_label = label

        if best_face is not None and best_label == 1 and best_confidence < 100:
            # We found the recognized face with enough confidence
            x, y, w, h = best_face
            name = "Om"  # Recognized as Om

            # Draw rectangle and put the name label on the original (colored) frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Reset the timer when a face is detected
            last_face_detection_time = time.time()

            # Motor control logic based on target's position
            face_center_x = x + w // 2
            if face_center_x < frame_center_x - left_threshold:
                print(f"{name} is on the left side")
                motor_lib.runMotor(0)  # Move counterclockwise
            elif face_center_x > frame_center_x + right_threshold:
                print(f"{name} is on the right side")
                motor_lib.runMotor(1)  # Move clockwise
            elif frame_center_x - center_threshold < face_center_x < frame_center_x + center_threshold:
                print(f"{name} is centered")
                motor_lib.stopMotor()  # Stop the motor
        else:
            # No recognized face (or not Om), treat as unknown
            print("Unknown or no face detected.")

            # Check if 10 seconds have passed without detecting a face
            if time.time() - last_face_detection_time > no_face_time_limit:
                print("No face detected for 10 seconds, motor searching...")

                # Rotate motor to search (smoothly rotate to both sides)
                print("Searching to the left...")
                smooth_motor_move(0, duration=2, step_time=0.1)  # Move left smoothly
                time.sleep(0.5)  # Pause
                print("Searching to the right...")
                smooth_motor_move(1, duration=2, step_time=0.1)  # Move right smoothly
                time.sleep(0.5)  # Pause
                motor_lib.stopMotor()

                # Reset the timer after the search
                last_face_detection_time = time.time()
            else:
                print("No face detected, stopping motor.")
                motor_lib.stopMotor()

        # Display the resulting frame (colored)
        cv2.imshow('Face Detection with Motor Control', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    motor_lib.stopMotor()
