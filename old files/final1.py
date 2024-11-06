import cv2
import numpy as np
import os
import subprocess
import threading

# Function to run the C program
def run_c_program(direction):
    subprocess.run(["./motor_start", direction])

# Paths and labels
your_image_path = 'frame.jpg'  # Replace with your image path
your_label = 0  # This label represents "XYZ" in the recognizer

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load and prepare your image for training
your_image = cv2.imread(your_image_path)
gray_image = cv2.cvtColor(your_image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# If a face is detected in your image, train the recognizer
if len(faces) == 1:
    (x, y, w, h) = faces[0]
    your_face = gray_image[y:y+h, x:x+w]
    recognizer.train([your_face], np.array([your_label]))
    print("Training completed with your image.")
else:
    print("Error: No face or multiple faces detected in your image.")
    exit()

# Start the real-time video capture
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection and recognition
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the current frame
    detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    # Assume the frame width is the center reference
    frame_center_x = frame.shape[1] // 2

    # Process each detected face
    for (x, y, w, h) in detected_faces:
        face_region = gray_frame[y:y+h, x:x+w]
        
        # Predict the label of the face
        label, confidence = recognizer.predict(face_region)
        
        # Check if the detected face matches "XYZ" or is "Unknown"
        if label == your_label and confidence < 70:  # Adjust confidence threshold as needed
            label_text = "XYZ"
            color = (0, 255, 0)  # Green for recognized face

            # Calculate the position of the face in relation to the center
            face_center_x = x + w // 2
            
            if face_center_x < frame_center_x - 50:  # Move left
                direction = "counterclockwise"
            elif face_center_x > frame_center_x + 50:  # Move right
                direction = "clockwise"
            else:
                direction = "stop"
                
            # Start the C program in a new thread with the appropriate direction
            threading.Thread(target=run_c_program, args=(direction,)).start()

        else:
            label_text = "Unknown"
            color = (0, 0, 255)  # Red for unrecognized face
        
        # Draw rectangle around the face and put the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the frame with detection and recognition
    cv2.imshow('Face Recognition', frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
