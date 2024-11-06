#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/time.h>
#include <errno.h>

/* GPIO handling */
#define ROOT_GPIO_DEVICES        "/sys/class/gpio"
#define EXPORT                   "export"
#define UNEXPORT                 "unexport"
#define DIRECTION                "direction"
#define VALUE                    "value"
#define OUT                      0
#define OUT_STR                  "out"
#define STR_LEN                  256

/* globals */
int iGPIONumber;            /* GPIO number that will be controlled */
int iState = 1;             /* State of the GPIO pin (1 = HIGH, 0 = LOW) */

/* declarations */
void SignalHandler(int);     /* Signal handler for timer interrupts */
int GPIOInit(int);           /* Initialize GPIO pin */
int GPIODeInit(int);         /* Deinitialize GPIO pin */
int GPIOSetDir(int, int);    /* Set GPIO direction (OUT) */
int GPIOWrite(int, int);     /* Write value (0 or 1) to GPIO pin */

/* GPIO initialization function */
int GPIOInit(int iGPIONumber)
{
    char szAccessPath[STR_LEN];
    FILE *fOut;

    sprintf(szAccessPath, "%s/%s", ROOT_GPIO_DEVICES, EXPORT);
    if ((fOut = fopen(szAccessPath, "w")) == NULL)
    {
        fprintf(stderr, "ERROR : GPIOInit() ---> call to fopen(%s,..)\n", szAccessPath);
        return (-errno);
    }
    fprintf(fOut, "%d", iGPIONumber);
    fclose(fOut);
    return (0);
}

/* GPIO deinitialization function */
int GPIODeInit(int iGPIONumber)
{
    char szAccessPath[STR_LEN];
    FILE *fOut;

    sprintf(szAccessPath, "%s/%s", ROOT_GPIO_DEVICES, UNEXPORT);
    if ((fOut = fopen(szAccessPath, "w")) == NULL)
    {
        fprintf(stderr, "ERROR : GPIODeInit() ---> call to fopen(%s,..)\n", szAccessPath);
        return (-errno);
    }
    fprintf(fOut, "%d", iGPIONumber);
    fclose(fOut);
    return (0);
}

/* Set GPIO direction (OUT) */
int GPIOSetDir(int iGPIONumber, int iDataDirection)
{
    char szAccessPath[STR_LEN];
    FILE *fOut;

    sprintf(szAccessPath, "%s/gpio%d/%s", ROOT_GPIO_DEVICES, iGPIONumber, DIRECTION);
    if ((fOut = fopen(szAccessPath, "w")) == NULL)
    {
        fprintf(stderr, "ERROR : GPIOSetDir() ---> call to fopen(%s,..)\n", szAccessPath);
        return (-errno);
    }
    fprintf(fOut, "%s", OUT_STR);
    fclose(fOut);
    return (0);
}

/* Write value (0 or 1) to GPIO pin */
int GPIOWrite(int iGPIONumber, int iValue)
{
    char szAccessPath[STR_LEN];
    FILE *fOut;

    sprintf(szAccessPath, "%s/gpio%d/%s", ROOT_GPIO_DEVICES, iGPIONumber, VALUE);
    if ((fOut = fopen(szAccessPath, "w")) == NULL)
    {
        fprintf(stderr, "ERROR : GPIOWrite() ---> call to fopen(%s,..)\n", szAccessPath);
        return (-errno);
    }
    fprintf(fOut, "%d", iValue);
    fclose(fOut);
    return (0);
}

/* Signal handler for periodic activity */
void SignalHandler(int signal)
{
    if (signal == SIGALRM)
    {
        /* Toggle the GPIO state */
        GPIOWrite(iGPIONumber, iState);
        iState = !iState;  /* Toggle state */
    }
}

int main(int argc, char *argv[])
{
    struct sigaction sa;
    struct itimerval t;

    /* Ensure correct argument usage */
    if (argc != 2)
    {
        printf("Usage: %s <GPIO number>\n", argv[0]);
        return (0);
    }

    /* Parse GPIO number from arguments */
    if (sscanf(argv[1], "%d", &iGPIONumber) == 0)
    {
        printf("Invalid GPIO number.\n");
        return (0);
    }

    /* Initialize GPIO */
    if (GPIOInit(iGPIONumber) < 0)
    {
        fprintf(stderr, "ERROR : main() ---> call to GPIOInit()\n");
        return (0);
    }

    /* Set GPIO direction to output */
    if (GPIOSetDir(iGPIONumber, OUT) < 0)
    {
        fprintf(stderr, "ERROR : main() ---> call to GPIOSetDir()\n");
        return (0);
    }

    /* Setup signal handler for periodic activity (SIGALRM) */
    memset(&sa, 0, sizeof(struct sigaction));
    sa.sa_handler = SignalHandler;
    sigaction(SIGALRM, &sa, NULL);

    /* Initialize timer to trigger every 0.0025 seconds (200Hz square wave) */
    t.it_interval.tv_sec = 0;
    t.it_interval.tv_usec = 2500;  /* 0.0025 s */
    t.it_value.tv_sec = 0;
    t.it_value.tv_usec = 2500;     /* 0.0025 s */
    setitimer(ITIMER_REAL, &t, NULL);

    /* Keep the program running, toggle GPIO state via SignalHandler */
    while (1)
    {
        pause();  /* Wait for signal to toggle GPIO */
    }

    return (0);
}
