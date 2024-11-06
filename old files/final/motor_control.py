# motor_control.py

import subprocess

# Function to run the C program
def run_c_program(direction):
    subprocess.run(["./motor_start", direction])
