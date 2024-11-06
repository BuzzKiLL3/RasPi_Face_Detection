import cv2
import cProfile
import time
import ctypes

# Load the shared library for GPIO control functions
gpio_lib = ctypes.CDLL('./myfunctions.so')  # Adjust path to the shared library

# Set GPIO pins for motor control
GPIO_PIN1 = 20  # Clockwise motor pin
GPIO_PIN2 = 21  # Counterclockwise motor pin

# Define function mappings for GPIO
gpio_lib.GPIOInit.argtypes = [ctypes.c_int]
gpio_lib.GPIOInit.restype = ctypes.c_int
gpio_lib.GPIOSetDir.argtypes = [ctypes.c_int, ctypes.c_int]
gpio_lib.GPIOSetDir.restype = ctypes.c_int
gpio_lib.GPIOWrite.argtypes = [ctypes.c_int, ctypes.c_int]
gpio_lib.GPIOWrite.restype = ctypes.c_int

# Initialize GPIO pins for motor control
gpio_lib.GPIOInit(GPIO_PIN1)
gpio_lib.GPIOInit(GPIO_PIN2)
gpio_lib.GPIOSetDir(GPIO_PIN1, 0)  # Set as output
gpio_lib.GPIOSetDir(GPIO_PIN2, 0)  # Set as output

# Function to control motor
def run_motor(clockwise):
    if clockwise:
        gpio_lib.GPIOWrite(GPIO_PIN1, 1)
        gpio_lib.GPIOWrite(GPIO_PIN2, 0)
    else:
        gpio_lib.GPIOWrite(GPIO_PIN1, 0)
        gpio_lib.GPIOWrite(GPIO_PIN2, 1)

# Initialize face detector
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def keep_face_centered(frame):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Screen center coordinates
    screen_center_x = int(frame.shape[1] / 2)

    if len(faces) == 0:
        print("No face detected")
        return  # No face detected, no motor action needed

    # Assuming the first detected face is the target face
    (x, y, w, h) = faces[0]
    face_center_x = x + w // 2

    # Calculate offset from screen center
    offset_x = screen_center_x - face_center_x

    # Determine motor direction based on offset
    if abs(offset_x) > 15:  # Threshold to reduce unnecessary adjustments
        if offset_x > 0:
            print("Moving motor clockwise to center face")
            run_motor(clockwise=True)
        else:
            print("Moving motor counterclockwise to center face")
            run_motor(clockwise=False)
        time.sleep(0.1)  # Adjust motor delay to avoid rapid movement
    else:
        print("Face is centered")
        gpio_lib.GPIOWrite(GPIO_PIN1, 0)
        gpio_lib.GPIOWrite(GPIO_PIN2, 0)  # Stop motor

    # Draw rectangle around face and display frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Face Tracker', frame)

def main():
    if not cap.isOpened():
        print("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        keep_face_centered(frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    gpio_lib.GPIOWrite(GPIO_PIN1, 0)
    gpio_lib.GPIOWrite(GPIO_PIN2, 0)

cProfile.run('main()')
