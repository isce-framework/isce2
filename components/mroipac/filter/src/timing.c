#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <limits.h>

struct tms buffer;
int user_time, system_time, start_time;

void start_timing()
{
  start_time = (int) times(&buffer);
  user_time = (int) buffer.tms_utime;
  system_time = (int) buffer.tms_stime;
}

void stop_timing()
{
  int  end_time,elapsed_time;
  int clk_tck;

  clk_tck = (int)sysconf(_SC_CLK_TCK);

  end_time = (int) times(&buffer);
  user_time = (int) (buffer.tms_utime - user_time);
  system_time = (int) (buffer.tms_stime - system_time);
  elapsed_time = (end_time - start_time);

  fprintf(stdout,"\n\nuser time    (s):  %10.3f\n", (double)user_time/clk_tck);
  fprintf(stdout,"system time  (s):  %10.3f\n", (double)system_time/clk_tck); 
  fprintf(stdout,"elapsed time (s):  %10.3f\n\n", (double) elapsed_time/clk_tck);
}
