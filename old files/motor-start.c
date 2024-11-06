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
#define GPIO_PIN1 20  // GPIO20 controls the motor clockwise
#define GPIO_PIN2 21  // GPIO21 controls the motor counterclockwise

/* function declarations */
int GPIOInit(int);        // Initialize a GPIO pin
int GPIOSetDir(int, int); // Set the direction of the GPIO pin (output/input)
int GPIOWrite(int, int);  // Write a value to the GPIO pin
void runMotor(int);       // Run the motor clockwise or counterclockwise

/* Main program */
int main(void)
{
    // Initialize both GPIO pins
    if (GPIOInit(GPIO_PIN1) < 0 || GPIOInit(GPIO_PIN2) < 0)
    {
        fprintf(stderr, "ERROR: Failed to initialize GPIO pins\n");
        return -1;
    }

    // Set both pins as output
    if (GPIOSetDir(GPIO_PIN1, OUT) < 0 || GPIOSetDir(GPIO_PIN2, OUT) < 0)
    {
        fprintf(stderr, "ERROR: Failed to set direction for GPIO pins\n");
        return -1;
    }

    // Run the motor clockwise
    printf("Clockwise\n");
    runMotor(1);
    sleep(50); 

    // Run the motor counterclockwise
    printf("Counterclockwise\n");
    runMotor(0);
    sleep(50); 

    // Stop the motor
    printf("Stopping motor\n");
    GPIOWrite(GPIO_PIN1, 0);
    GPIOWrite(GPIO_PIN2, 0);

    return 0;
}

/* GPIO initialization */
int GPIOInit(int iGPIONumber)
{
    char szAccessPath[STR_LEN];
    FILE *fOut;

    sprintf(szAccessPath, "%s/%s", ROOT_GPIO_DEVICES, EXPORT);
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

    if (iValue == 0)
    {
        fprintf(fOut, "0");
    }
    else if (iValue == 1)
    {
        fprintf(fOut, "1");
    }
    else
    {
        fprintf(stderr, "ERROR: GPIOWrite() -> output value must be 0 or 1\n");
        fclose(fOut);
        return -1;
    }

    fclose(fOut);
    return 0;
}

/* Run the motor */
void runMotor(int direction)
{
    if (direction == 1) // Clockwise
    {
        GPIOWrite(GPIO_PIN1, 1);
        GPIOWrite(GPIO_PIN2, 0);
    }
    else if (direction == 0) // Counterclockwise
    {
        GPIOWrite(GPIO_PIN1, 0);
        GPIOWrite(GPIO_PIN2, 1);
    }
}
