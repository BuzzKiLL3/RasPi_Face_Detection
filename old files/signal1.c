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
#define GPIO_PIN 26  // GPIO26 corresponds to pin 37 on the Raspberry Pi

/* function declarations */
int GPIOInit(int);        // Initialize a GPIO pin
int GPIOSetDir(int, int); // Set the direction of the GPIO pin (output/input)
int GPIOWrite(int, int);  // Write a value to the GPIO pin
void usage(char *);       // Usage message

/* Main program */
int main(void)
{
    int gpioNumber = GPIO_PIN; // GPIO pin 26 (corresponding to pin 37)
    
    // Initialize the GPIO pin
    if (GPIOInit(gpioNumber) < 0)
    {
        fprintf(stderr, "ERROR: Failed to initialize GPIO %d\n", gpioNumber);
        return -1;
    }

    // Set GPIO direction to output
    if (GPIOSetDir(gpioNumber, OUT) < 0)
    {
        fprintf(stderr, "ERROR: Failed to set direction of GPIO %d to output\n", gpioNumber);
        return -1;
    }

    // Write a value of 1 to the GPIO pin (set it HIGH)
    if (GPIOWrite(gpioNumber, 1) < 0)
    {
        fprintf(stderr, "ERROR: Failed to write to GPIO %d\n", gpioNumber);
        return -1;
    }

    printf("GPIO %d (pin 37) is set to HIGH\n", gpioNumber);

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
