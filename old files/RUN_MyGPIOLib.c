/* here we try to write a simple set of function aimed to easily access the Raspberry Pi */
/* GPIO from the text file abstraction provided by the Operating System                  */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>

extern char *strerror( int );
/* constants */
#define ROOT_GPIO_DEVICES        "/sys/class/gpio"
#define EXPORT                   "export"
#define UNEXPORT                 "unexport"
#define DIRECTION                "direction"
#define VALUE                    "value"
#define OUT                      0
#define IN                       1
#define OUT_STR                  "out"
#define IN_STR                   "in"
#define STR_LEN                  256
#define NB_ARGS                  2

#define SLEEP_TIME               500000
/* declarations */
void usage( char *);        /* ->help message                                        */
int GPIOInit( int );        /* ->init a GPIO pin using the corresponding export file */   
int GPIODeInit( int );      /* ->deinitialize a GPIO pin using the corresponding unexport file */
int GPIOSetDir( int, int ); /* ->set data direction (0 --> out,  1 --> in )*/
int GPIOWrite( int, int );  /* ->write a value (0 or 1 ) to a GPIO pin */
int GPIORead( int );        /* ->read a value (0 or 1) from a GPIO pin */
/*&&&&&&*/
/* help */
/*&&&&&&*/
void usage( char *szPgmName)
{
    if( szPgmName == NULL )
    {
        exit( -1 );
    };
    printf("%s <GPIO number>\n", szPgmName);
    printf("change peridicaly the state of the specified GPIO\n");
}
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
/* init a GPIO pin without setting */
/* the data direction              */
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
int GPIOInit( int iGPIONumber)
{
    char    szAccessPath[STR_LEN];
    FILE    *fOut;
    /* build the path to the export file : */
    sprintf(szAccessPath,"%s/%s",ROOT_GPIO_DEVICES, EXPORT );   
    /* try to open the export file :*/
    if( (fOut = fopen(szAccessPath,"w")) == NULL)
    {
        fprintf(stderr,"ERROR : GPIOInit() ---> call to fopen(%s,..)\n", szAccessPath);
        fprintf(stderr,"        error code %d (%s)\n", errno, (char *)(strerror(errno)));
        return( -errno );
    };
    /* write the GPIO number in the export file */
    fprintf(fOut,"%d", iGPIONumber);
    fclose( fOut );
    /* OK */
    return( 0 );
}
/*&&&&&&&&&&&&&&&&&&&&&&*/
/* deinitialize a GPIO  */
/*&&&&&&&&&&&&&&&&&&&&&&*/
int GPIODeInit( int iGPIONumber)
{
    char    szAccessPath[STR_LEN];
    FILE    *fOut;
    /* build the path to the export file : */
    sprintf(szAccessPath,"%s/%s",ROOT_GPIO_DEVICES, UNEXPORT );   
    /* try to open the export file :*/
    if( (fOut = fopen(szAccessPath,"w")) == NULL)
    {
        fprintf(stderr,"ERROR : GPIODeInit() ---> call to fopen(%s,..)\n", szAccessPath);
        fprintf(stderr,"        error code %d (%s)\n", errno, (char *)(strerror(errno)));
        return( -errno );
    };
    /* write the GPIO number in the export file */
    fprintf(fOut,"%d", iGPIONumber);
    fclose( fOut );
    /* OK */
    return( 0 );
}
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
/* set the data direction for a given GPIO number */
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
int GPIOSetDir( int     iGPIONumber,            /* ->GPIO number direction has to be set */
                int     iDataDirection )        /* ->data direction                      */
{
    char    szAccessPath[STR_LEN];
    FILE    *fOut;
    /* build the path to the export file : */
    sprintf(szAccessPath,"%s/gpio%02d/%s",ROOT_GPIO_DEVICES, iGPIONumber, DIRECTION );   
    /* try to open the data direction  file :*/
    if( (fOut = fopen(szAccessPath,"w")) == NULL)
    {
        fprintf(stderr,"ERROR : GPIOSetDir() ---> call to fopen(%s,..)\n", szAccessPath);
        fprintf(stderr,"        error code %d (%s)\n", errno, (char *)(strerror(errno)));
        return( -errno );
    };
    switch( iDataDirection )
    {
        case OUT:
            fprintf(fOut,"%s", OUT_STR);
            break;
        case IN:
            fprintf(fOut,"%s", IN_STR); 
            break;
        default:
            fclose( fOut );
            return( -1 );
            break;
    };
    fclose( fOut );
    /* OK */
    return( 0 );
}
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
/* write a value to a given GPIO */
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
int GPIOWrite(  int iGPIONumber,            /* ->number or the GPIO to be accessed */
                int iValue         )        /* ->value (0 or 1)                    */
{
char    szAccessPath[STR_LEN];
    FILE    *fOut;
    /* build the path to the export file : */
    sprintf(szAccessPath,"%s/gpio%02d/%s",ROOT_GPIO_DEVICES, iGPIONumber, VALUE );   
    /* try to open the data direction  file :*/
    if( (fOut = fopen(szAccessPath,"w")) == NULL)
    {
        fprintf(stderr,"ERROR : GPIOWrite() ---> call to fopen(%s,..)\n", szAccessPath);
        fprintf(stderr,"        error code %d (%s)\n", errno, (char *)(strerror(errno)));
        return( -errno );
    };
    switch( iValue )
    {
        case 0:
            fprintf(fOut,"0");
            break;
        case 1:
            fprintf(fOut,"1");
            break;
        default:
            fprintf(stderr,"ERROR : GPIOWrite() ---> output value must be 0 or 1\n");
            fprintf(stderr,"        (%d provided )\n", iValue );
            fclose( fOut );
            return( -1 );
            break;
    };
    fclose( fOut );
    /* OK */
    return( 0 );   
}
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
/* write a value to a given GPIO */
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
int GPIORead(  int iGPIONumber )            /* ->number or the GPIO to be accessed */
{
    char    szAccessPath[STR_LEN];
    FILE    *fOut;
    int     iInValue;
    /* build the path to the export file : */
    sprintf(szAccessPath,"%s/gpio%02d/%s",ROOT_GPIO_DEVICES, iGPIONumber, VALUE );   
    /* try to open the data direction  file :*/
    if( (fOut = fopen(szAccessPath,"w")) == NULL)
    {
        fprintf(stderr,"ERROR : GPIORead() ---> call to fopen(%s,..)\n", szAccessPath);
        fprintf(stderr,"        error code %d (%s)\n", errno, (char *)(strerror(errno)));
        return( -errno );
    };
    if( fscanf(fOut,"%d", &iInValue) == 0 )
    {
        fprintf(stderr,"ERROR : GPIORead() ---> call to fscan()\n");
        fprintf(stderr,"        cannot read an integer value.\n");
        return( -errno );
    }
    fclose( fOut );
    /* OK */
    return( iInValue );   
}
/*######*/
/* main */
/*######*/
int main( int argc, char *argv[] )
{
    int     iState = 1;
    int     iGPIONumber;

    if( argc != NB_ARGS)
    {
        usage( argv[0]);
        return( 0 );
    };

    if( sscanf(argv[1],"%d", &iGPIONumber) == 0 )
    {
        usage(argv[0]);
        return( 0 );
    };
    if( GPIOInit( iGPIONumber) < 0 )
    {
        fprintf(stderr,"ERROR : %s.main() ---> call to GPIOInit()\n", argv[0]);
        return( 0 );
    };
    if( GPIOSetDir(iGPIONumber, OUT) < 0 )
    {
        fprintf(stderr,"ERROR : %s.main() ---> call to GPIOSetDir()\n", argv[0]);
        return( 0 );
    };

    for(;;)
    {
        GPIOWrite( iGPIONumber, iState);
        usleep( SLEEP_TIME);
        if( iState == 1)
        {
            iState = 0;
        }
        else
        {
            iState = 1;
        };
    };
    return( 0 );
}