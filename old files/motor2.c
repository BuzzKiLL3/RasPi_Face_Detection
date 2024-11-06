#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

/* constants */
#define ROOT_GPIO_DEVICES "/sys/class/gpio"
#define EXPORT "export"
#define UNEXPORT "unexport"
#define DIRECTION "direction"
#define VALUE "value"
#define OUT 0
#define OUT_STR "out"
#define STR_LEN 256
#define GPIO_PIN_CW 529  // GPIO17 controls the motor clockwise
#define GPIO_PIN_CCW 539  // GPIO27 controls the motor counterclockwise

/* function declarations */
int GPIOInit(int);        // Initialize a GPIO pin
int GPIOSetDir(int, int); // Set the direction of the GPIO pin (output/input)
int GPIOWrite(int, int);  // Write a value to the GPIO pin
void runMotor(int);       // Run the motor clockwise or counterclockwise
void stopMotor();         // Stop the motor

/* Main program */
int main(void)
{
    // Initialize both GPIO pins
    if (GPIOInit(GPIO_PIN_CW) < 0 || GPIOInit(GPIO_PIN_CCW) < 0)
    {
        fprintf(stderr, "ERROR: Failed to initialize GPIO pins\n");
        return -1;
    }

    // Set both pins as output
    if (GPIOSetDir(GPIO_PIN_CW, OUT) < 0 || GPIOSetDir(GPIO_PIN_CCW, OUT) < 0)
    {
        fprintf(stderr, "ERROR: Failed to set direction for GPIO pins\n");
        return -1;
    }

    char command[10];

    while (1) {
        printf("Enter command (cw, ccw, stop, exit): ");
        fgets(command, sizeof(command), stdin);

        // Check for command to run the motor
        if (strncmp(command, "cw", 2) == 0) {
            printf("Running motor clockwise\n");
            runMotor(1); // 1 for clockwise
        } else if (strncmp(command, "ccw", 3) == 0) {
            printf("Running motor counterclockwise\n");
            runMotor(0); // 0 for counterclockwise
        } else if (strncmp(command, "stop", 4) == 0) {
            printf("Stopping motor\n");
            stopMotor(); // Stop the motor
        } else if (strncmp(command, "exit", 4) == 0) {
            printf("Exiting program\n");
            break;
        } else {
            printf("Unknown command. Use 'cw', 'ccw', 'stop', or 'exit'.\n");
        }
    }

    // Clean up by unexporting the GPIO pins
    GPIOInit(-GPIO_PIN_CW);
    GPIOInit(-GPIO_PIN_CCW);

    return 0;
}

/* GPIO initialization */
int GPIOInit(int iGPIONumber)
{
    char szAccessPath[STR_LEN];
    FILE *fOut;

    if (iGPIONumber < 0) {
        // Unexport GPIO if iGPIONumber is negative
        iGPIONumber = -iGPIONumber;
        sprintf(szAccessPath, "%s/%s", ROOT_GPIO_DEVICES, UNEXPORT);
    } else {
        sprintf(szAccessPath, "%s/%s", ROOT_GPIO_DEVICES, EXPORT);
    }

    if ((fOut = fopen(szAccessPath, "w")) == NULL)
    {
        fprintf(stderr, "ERROR: GPIOInit() -> fopen(%s,..)\n", szAccessPath);
        fprintf(stderr, "       error code %d (%s)\n", errno, strerror(errno));
        return -errno;
    }

    fprintf(fOut, "%d", iGPIONumber);
    fclose(fOut);
    return 0;
}

/* Set GPIO direction (output) */
int GPIOSetDir(int iGPIONumber, int iDataDirection)
{
    char szAccessPath[STR_LEN];
    FILE *fOut;

    sprintf(szAccessPath, "%s/gpio%d/%s", ROOT_GPIO_DEVICES, iGPIONumber, DIRECTION);
    if ((fOut = fopen(szAccessPath, "w")) == NULL)
    {
        fprintf(stderr, "ERROR: GPIOSetDir() -> fopen(%s,..)\n", szAccessPath);
        fprintf(stderr, "       error code %d (%s)\n", errno, strerror(errno));
        return -errno;
    }

    if (iDataDirection == OUT)
    {
        fprintf(fOut, "%s", OUT_STR);
    }
    else
    {
        fclose(fOut);
        return -1;
    }

    fclose(fOut);
    return 0;
}

/* Write value to GPIO */
int GPIOWrite(int iGPIONumber, int iValue)
{
    char szAccessPath[STR_LEN];
    FILE *fOut;

    sprintf(szAccessPath, "%s/gpio%d/%s", ROOT_GPIO_DEVICES, iGPIONumber, VALUE);
    if ((fOut = fopen(szAccessPath, "w")) == NULL)
    {
        fprintf(stderr, "ERROR: GPIOWrite() -> fopen(%s,..)\n", szAccessPath);
        fprintf(stderr, "       error code %d (%s)\n", errno, strerror(errno));
        return -errno;
    }

    fprintf(fOut, "%d", iValue); // Writing 0 or 1 directly
    fclose(fOut);
    return 0;
}

/* Run the motor */
void runMotor(int direction)
{
    if (direction == 1) // Clockwise
    {
        GPIOWrite(GPIO_PIN_CW, 1);
        GPIOWrite(GPIO_PIN_CCW, 0);
    }
    else if (direction == 0) // Counterclockwise
    {
        GPIOWrite(GPIO_PIN_CW, 0);
        GPIOWrite(GPIO_PIN_CCW, 1);
    }
}

/* Stop the motor */
void stopMotor()
{
    GPIOWrite(GPIO_PIN_CW, 0);
    GPIOWrite(GPIO_PIN_CCW, 0);
}
