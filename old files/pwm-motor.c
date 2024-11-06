#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

/* constants */
#define ROOT_PWM_DEVICES "/sys/class/pwm/pwmchip0"
#define EXPORT "export"
#define UNEXPORT "unexport"
#define PERIOD "period"
#define DUTY_CYCLE "duty_cycle"
#define ENABLE "enable"

#define ROOT_GPIO_DEVICES "/sys/class/gpio"
#define GPIO_PIN1 20  // GPIO20 controls the motor direction
#define OUT 0
#define OUT_STR "out"
#define STR_LEN 256

/* function declarations */
int GPIOInit(int);                 // Initialize a GPIO pin
int GPIOSetDir(int, int);          // Set the direction of the GPIO pin (output/input)
int GPIOWrite(int, int);           // Write a value to the GPIO pin
int PWMInit(int);                  // Initialize the PWM channel
int PWMSetDutyCycle(int);          // Set the duty cycle of the PWM channel
void setMotorSpeed(int, int);      // Set motor speed and direction

/* Main program */
int main(void)
{
    // Initialize GPIO pin for motor direction
    if (GPIOInit(GPIO_PIN1) < 0 || GPIOSetDir(GPIO_PIN1, OUT) < 0)
    {
        fprintf(stderr, "ERROR: Failed to initialize GPIO pin\n");
        return -1;
    }

    // Initialize PWM
    if (PWMInit(1000000) < 0) // Set a 1ms period (1kHz frequency)
    {
        fprintf(stderr, "ERROR: Failed to initialize PWM\n");
        return -1;
    }

    // Set motor speed and direction
    printf("Setting motor speed to 50%% (clockwise)\n");
    setMotorSpeed(50, 1);  // 50% speed, clockwise
    sleep(5);

    printf("Setting motor speed to 75%% (counterclockwise)\n");
    setMotorSpeed(75, 0);  // 75% speed, counterclockwise
    sleep(5);

    printf("Stopping motor\n");
    setMotorSpeed(0, 1);   // Stop motor

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

    sprintf(szAccessPath, "%s/gpio%d/direction", ROOT_GPIO_DEVICES, iGPIONumber);
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
    fclose(fOut);
    return 0;
}

/* Write value to GPIO */
int GPIOWrite(int iGPIONumber, int iValue)
{
    char szAccessPath[STR_LEN];
    FILE *fOut;

    sprintf(szAccessPath, "%s/gpio%d/value", ROOT_GPIO_DEVICES, iGPIONumber);
    if ((fOut = fopen(szAccessPath, "w")) == NULL)
    {
        fprintf(stderr, "ERROR: GPIOWrite() -> fopen(%s,..)\n", szAccessPath);
        fprintf(stderr, "       error code %d (%s)\n", errno, strerror(errno));
        return -errno;
    }

    fprintf(fOut, "%d", iValue);
    fclose(fOut);
    return 0;
}

/* Initialize PWM */
int PWMInit(int period_ns)
{
    FILE *fOut;

    // Export PWM0
    if ((fOut = fopen(ROOT_PWM_DEVICES "/" EXPORT, "w")) == NULL)
    {
        fprintf(stderr, "ERROR: PWMInit() -> export failed\n");
        return -1;
    }
    fprintf(fOut, "0");  // Export PWM0
    fclose(fOut);

    // Set the period
    if ((fOut = fopen(ROOT_PWM_DEVICES "/pwm0/" PERIOD, "w")) == NULL)
    {
        fprintf(stderr, "ERROR: PWMInit() -> period failed\n");
        return -1;
    }
    fprintf(fOut, "%d", period_ns);
    fclose(fOut);

    // Enable PWM
    if ((fOut = fopen(ROOT_PWM_DEVICES "/pwm0/" ENABLE, "w")) == NULL)
    {
        fprintf(stderr, "ERROR: PWMInit() -> enable failed\n");
        return -1;
    }
    fprintf(fOut, "1");
    fclose(fOut);

    return 0;
}

/* Set PWM duty cycle */
int PWMSetDutyCycle(int duty_cycle_percent)
{
    FILE *fOut;
    int duty_cycle_ns = 10000 * duty_cycle_percent; // Scale to ns assuming 1ms period

    if ((fOut = fopen(ROOT_PWM_DEVICES "/pwm0/" DUTY_CYCLE, "w")) == NULL)
    {
        fprintf(stderr, "ERROR: PWMSetDutyCycle() -> duty_cycle failed\n");
        return -1;
    }

    fprintf(fOut, "%d", duty_cycle_ns);
    fclose(fOut);

    return 0;
}

/* Set motor speed and direction */
void setMotorSpeed(int speed_percent, int direction)
{
    GPIOWrite(GPIO_PIN1, direction);   // Set direction (1 for clockwise, 0 for counterclockwise)
    PWMSetDutyCycle(speed_percent);    // Set PWM duty cycle for speed
}
