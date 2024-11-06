import cv2
import numpy as np
from picamzero import Camera
from time import sleep

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('/home/imt/src/final/haarcascade_frontalface_default.xml')

# Load and prepare your image for training
your_image_path = '/home/imt/src/final/frame.jpg'  # Replace with your image path
your_label = 0  # This label represents "XYZ" in the recognizer

# Load your training image
your_image = cv2.imread(your_image_path)

# Check if the image was loaded successfully
if your_image is None:
    print(f"Error: Could not load image at '{your_image_path}'. Please check the path.")
    exit()

# Convert the loaded image to grayscale
gray_image = cv2.cvtColor(your_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Train the recognizer if a face is detected
if len(faces) == 1:
    (x, y, w, h) = faces[0]
    your_face = gray_image[y:y+h, x:x+w]
    print("Training completed with your image.")
else:
    print("Error: No face or multiple faces detected in your image.")
    exit()

# Function to handle the camera and real-time face detection
def start_camera():
    with Camera() as cam:
        cam.start_preview()
        sleep(2)  # Allow time for the camera to warm up

        while True:
            frame = cam.get_frame()  # Get the current frame from the camera
            if frame is None:
                print("Error: Unable to read from camera.")
                break

            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the current frame
            detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            # Process each detected face
            for (x, y, w, h) in detected_faces:
                face_region = gray_frame[y:y+h, x:x+w]

                # Here, you could add face recognition logic as needed
                # For now, just draw a rectangle around detected faces
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Show the frame with detection
            cv2.imshow('Face Recognition', frame)

            # Exit loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# To run the camera in real-time detection
if __name__ == "__main__":
    start_camera()
