import cv2
import subprocess

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize camera
cap = cv2.VideoCapture(0)

# Set thresholds for detection
left_threshold = 100
right_threshold = 100
center_threshold = 30

# Start the motor control program
motor_process = subprocess.Popen(["./motorfinal"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Assume first detected face
            face_center_x = x + w // 2

            # Draw rectangle for detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Motor control logic based on target's position
            if face_center_x < frame_center_x - left_threshold:
                print("Face is on the left side")
                motor_process.stdin.write("ccw\n")  # Move counterclockwise
                motor_process.stdin.flush()
            elif face_center_x > frame_center_x + right_threshold:
                print("Face is on the right side")
                motor_process.stdin.write("cw\n")  # Move clockwise
                motor_process.stdin.flush()
            elif frame_center_x - center_threshold < face_center_x < frame_center_x + center_threshold:
                print("Face is centered")
                motor_process.stdin.write("stop\n")  # Stop the motor
                motor_process.stdin.flush()

        else:
            print("No face detected, stopping motor.")
            if motor_process.poll() is None:  # Check if motor process is still running
                motor_process.stdin.write("stop\n")
                motor_process.stdin.flush()

        # Display the resulting frame
        cv2.imshow('Face Detection with Motor Control', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except BrokenPipeError:
    print("Motor process has terminated unexpectedly.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    if motor_process.poll() is None:  # Ensure motor process is terminated
        motor_process.terminate()