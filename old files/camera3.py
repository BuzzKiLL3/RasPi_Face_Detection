import cv2
import numpy as np
from rapicam import Camera

# Initialize the camera
camera = Camera(resolution=(640, 480), framerate=30)

# Start the camera
camera.start()

# Path to the Haar cascade file for face detection
haarcascade_path = '/home/imt/src/haarcascade_frontalface_default.xml'

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Initialize the LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load and prepare your image for training
your_image_path = 'frame.jpg'  # Replace with your image path
your_label = 0  # This label represents "XYZ" in the recognizer

your_image = cv2.imread(your_image_path)
gray_image = cv2.cvtColor(your_image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

if len(faces) == 1:
    (x, y, w, h) = faces[0]
    your_face = gray_image[y:y+h, x:x+w]
    recognizer.train([your_face], np.array([your_label]))
    print("Training completed with your image.")
else:
    print("Error: No face or multiple faces detected in your image.")
    exit()

# Start real-time face detection
while True:
    frame = camera.capture_array()  # Capture frame from the camera
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        face_region = gray_frame[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_region)

        if label == your_label and confidence < 70:
            label_text = "XYZ"
            color = (0, 255, 0)  # Green for recognized face
        else:
            label_text = "Unknown"
            color = (0, 0, 255)  # Red for unrecognized face

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.close()
cv2.destroyAllWindows()
