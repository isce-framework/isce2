#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

// This function concatenates, potentially non-continuous frames together along track using a last-in-last-out paradigm.
// output is the name of the file, 
// Width should be the number of bytes per line, not the number of samples.
int
frame_concatenate(char *output, int * pwidth, int * plast_line_num, int * pnumber_of_frames, char **input, int *start_line)
{
    FILE *  in;
    FILE *  out;
    int lineAddr;
    char *inData,*outData;
    int width,last_line_num,number_of_frames;
    int i,j,k,number_of_lines,number_valid_lines,total_line_num;
    width= (*pwidth);
    last_line_num = (*plast_line_num);
    number_of_frames = (*pnumber_of_frames);
    // Create a memory map of the output file
    out = fopen(output,"w");
    char * oneLine = (char *) malloc(width*sizeof(char)); // turned out that is better to use one line at the time instead of a bunch of them so we don not 
    // interfere with  the OS  I/O buffering
    int cnt = 0;
    for(i=0;i<number_of_frames;i++)
    {
        printf("Adding frame %d of %d\n",(i+1),number_of_frames);
        // Get the file size
        in = fopen(input[i],"r");
        printf("Starting concatenation at line %d of output file\n",start_line[i]);
        if( i == number_of_frames - 1) // last one
        {
            while(1)
            {
                int actualRead = fread(oneLine,sizeof(char),width,in);
                if((actualRead  == 0))
                {
                    break;
                }
                else
                {
                    fwrite(oneLine,sizeof(char),actualRead,out);
                    ++cnt;
                }

            }
        }
        else
        {
            int tot_line  = start_line[i+1] - start_line[i];  
            for( k = 0; k < tot_line; ++k)
            {

                fread(oneLine,sizeof(char),width,in);
                fwrite(oneLine,sizeof(char),width,out);
                 ++cnt;
            }
        }
        fclose(in);
    }
    free(oneLine);
    fclose(out);
}
