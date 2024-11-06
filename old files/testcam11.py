import cv2
import numpy as np
import subprocess

# Define the face cascade for detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_faces(frame):
    """Detect faces in a frame and return the frame with rectangles around detected faces."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame, len(faces)

def main():
    # Start libcamera-vid in subprocess for continuous video capture
    libcamera_process = subprocess.Popen(
        ["libcamera-vid", "--inline", "--width", "320", "--height", "240", "--nopreview", "-o", "-"],
        stdout=subprocess.PIPE
    )

    # Read video stream and detect faces
    while True:
        # Capture frame-by-frame
        raw_frame = libcamera_process.stdout.read(320 * 240 * 3)  # Read one frame at a time
        if len(raw_frame) == 320 * 240 * 3:  # Ensure frame is correctly captured
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((240, 320, 3))
            
            # Perform face detection
            frame_with_faces, face_count = detect_faces(frame)
            print(f"Detected {face_count} face(s).")

            # Display the frame with detected faces
            cv2.imshow("Face Detection", frame_with_faces)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Cleanup
    libcamera_process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
