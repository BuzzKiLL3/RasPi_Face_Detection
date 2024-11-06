import cv2
import ctypes
import os
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
left_threshold = 100
right_threshold = 100
center_threshold = 30

try:
    while True:
        # Capture frame from the camera
        frame = picam2.capture_array()

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Correct the color balance: Apply a simple manual color correction (tune these values)
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=0)  # Increase brightness and contrast if needed
        # Alternatively, apply white balance correction
        # frame[:, :, 0] = frame[:, :, 0] + 20  # Add to the Blue channel
        # frame[:, :, 1] = frame[:, :, 1] - 10  # Subtract from the Green channel
        # frame[:, :, 2] = frame[:, :, 2] - 10  # Subtract from the Red channel

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Define the frame center
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Assume first detected face
            face_center_x = x + w // 2

            # Crop the detected face from the frame
            face = gray_frame[y:y + h, x:x + w]

            # Try to recognize the face using the trained model
            label, confidence = recognizer.predict(face)
            name = "Unknown"  # Default to Unknown

            # If the confidence is below a threshold, we recognize the face as "Om"
            if label == 1 and confidence < 100:
                name = "Om"
            
            # Draw rectangle and put the name label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Motor control logic based on target's position
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
            print("No face detected, stopping motor.")
            motor_lib.stopMotor()

        # Display the resulting frame
        cv2.imshow('Face Detection with Motor Control', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    motor_lib.stopMotor()
