import cv2
import numpy as np
import subprocess
import threading

# Initialize the LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
    recognizer.train([your_face], np.array([your_label]))
    print("Training completed with your image.")
else:
    print("Error: No face or multiple faces detected in your image.")
    exit()

# Function to handle the camera and real-time face detection
def start_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    frame_center_x = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the current frame
        detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        # Set frame_center_x only once to avoid recalculating it repeatedly
        if frame_center_x is None:
            frame_center_x = frame.shape[1] // 2

        # Process each detected face
        for (x, y, w, h) in detected_faces:
            face_region = gray_frame[y:y+h, x:x+w]
            
            # Predict the label of the face
            label, confidence = recognizer.predict(face_region)
            
            # Send command based on detection
            if label == your_label and confidence < 70:  # Adjust confidence threshold as needed
                command = determine_command(x, w, frame_center_x)
                threading.Thread(target=run_motor_command, args=(command,)).start()
            else:
                print("Unknown face detected.")

            # Draw rectangle around the face
            color = (0, 255, 0) if label == your_label else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Show the frame with detection
        cv2.imshow('Face Recognition', frame)
        
        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()

# Function to determine motor command based on face position
def determine_command(x, w, frame_center_x):
    face_center_x = x + w // 2
    if face_center_x < frame_center_x - 50:  # Move left
        return "counterclockwise"
    elif face_center_x > frame_center_x + 50:  # Move right
        return "clockwise"
    else:
        return "stop"

# This function will be called from the main script to run motor commands
def run_motor_command(command):
    from motor_control import run_c_program  # Import here to avoid circular import
    run_c_program(command)

# To run the camera in real-time detection
if __name__ == "__main__":
    start_camera()
