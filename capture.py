import time
import cv2
from picamera2 import Picamera2
import numpy as np

# Initialize Picamera2
picam2 = Picamera2()
picam2.start()

# Function to capture image when 'c' is pressed
def capture_image():
    # Capture image from the camera (RGB format)
    image = picam2.capture_array()
    
    # Convert the RGB image to BGR format (for OpenCV)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Display the image on screen
    cv2.imshow("Captured Image", image_bgr)

    # Save the captured image to file
    cv2.imwrite("me.jpg", image_bgr)
    print("Picture taken and saved as 'me.jpg'")

try:
    while True:
        # Capture a frame from the camera (RGB format)
        frame = picam2.capture_array()

        # Convert the frame from RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow("Camera Feed", frame_bgr)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Capture image when 'c' is pressed
        if key == ord('c'):
            capture_image()
        
        # Exit the loop if 'q' is pressed
        if key == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    # Stop the camera and close any OpenCV windows
    picam2.stop()
    cv2.destroyAllWindows()
