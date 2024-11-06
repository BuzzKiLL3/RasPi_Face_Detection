import cv2
import subprocess
import numpy as np

# Define the face cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def capture_frame():
    """Capture a frame using libcamera and return it as a numpy array."""
    # Run libcamera-still to capture a frame as raw data
    result = subprocess.run(
        ["libcamera-still", "-t", "1", "--width", "640", "--height", "480", "--nopreview", "-o", "-"],
        stdout=subprocess.PIPE
    )
    # Decode the image data as a numpy array
    frame = np.frombuffer(result.stdout, dtype=np.uint8)
    image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    return image

def detect_faces(frame):
    """Detect faces in a frame and return the frame with rectangles around detected faces."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame, len(faces)

def main():
    while True:
        frame = capture_frame()
        if frame is not None:
            # Detect faces in the frame
            frame_with_faces, face_count = detect_faces(frame)
            print(f"Detected {face_count} face(s).")

            # Show the frame with detected faces
            cv2.imshow("Face Detection", frame_with_faces)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Failed to capture image.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
