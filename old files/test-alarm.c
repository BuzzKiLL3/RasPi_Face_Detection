/* test program to generate peridic activity */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/time.h>
/* globals */
int hh,mm,ss,dd;
/* declaration */
void SignalHandler( int );
/* signal handler (digital clock...) */
void SignalHandler( int signal )
{
  if( signal == SIGALRM )
  {
    dd++;
    if( dd == 10 )
    {
      ss++;
      dd = 0;
      if( ss == 60 )
      {
	mm++;
	ss = 0;
	if( mm == 60 )
	{
	  hh++;
	  mm = 0;
	};
      };
    };
  };
}
/* program  */
int main( void )
{
  struct sigaction sa, sa_old;
  struct itimerval t;
  sigset_t mask;
  hh = mm = ss = dd = 0;
  /* init timer values : */
  t.it_interval.tv_sec = 0;
  t.it_interval.tv_usec = 1e5;  /* 0.1 s */
  t.it_value.tv_sec = 0;
  t.it_value.tv_usec = 1e5;  /* 0.1 s */
  /* install signal handler */
  memset(&sa,0, sizeof(struct sigaction));
  memset(&sa_old,0, sizeof(struct sigaction));
  sigemptyset( &mask );
  sa.sa_handler = SignalHandler;
  sa.sa_mask = mask;
  sigaction( SIGALRM, &sa, &sa_old );
  /* start the timer */
  setitimer( ITIMER_REAL, &t, NULL );
  /* display the clock value (start = 00:00:00.00 ) */
  do
  {
    pause();
    printf("%02d:%02d:%02d.%02d\n", hh,mm,ss,dd );
  }
  while( 1 );
  return( 0 );
}
  
	
