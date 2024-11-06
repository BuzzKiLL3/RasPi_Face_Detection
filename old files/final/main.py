# main.py

import threading
from camera_face_detection import start_camera
from motor_control import run_c_program

if __name__ == "__main__":
    # Start the camera and face detection in a separate thread
    camera_thread = threading.Thread(target=start_camera)
    camera_thread.start()
    
    # Join the thread to ensure the main script waits for it
    camera_thread.join()
