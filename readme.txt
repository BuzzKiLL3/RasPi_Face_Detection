Before running the program follow the steps below: 
Steps to run the Face Detection
Step 1 : Run the "capture.py" to capture your image and save so that the face detection has a face.
Step 2 : Once the program runs align your face and press 'c' to click your piture. The image will be automatically saved. Press 'q' to quit.
Step 3 : Run this command 'gcc -shared -o libmotor_control.so -fPIC motor_control.c' the .so sharing file from .c to .py will be created.
Step 4 : Now Open "main.py" as to edit and change the path of the HaarCode.xml , me.jpg and the .so file. (If you dont change the path the program will not work)
Step 5 : Run the "main.py" file, which internally runs the motor_control.c and the face detection.
Step 6 : Thank you!  