#include <stdio.h>
#include <stdlib.h>

// Given an input file, and the number of samples to left pad and right pad 
// and a pad value, create an output file with the new dimensions and padded 
// with the given pad value.
    int
swst_resample(char *input, char *output, int * pwidth, int * pleft_pad, int * pright_pad, unsigned char * ppad_value)
{
    int i,count,line_num;
    char *in_line, *out_line;
    FILE *in,*out;
    int width = (*pwidth);
    int left_pad = (*pleft_pad);
    int right_pad = (*pright_pad);
    unsigned char pad_value = (*ppad_value);
    in = fopen(input,"rb");
    out = fopen(output,"wb");

    in_line = (char *)malloc(width*sizeof(char));
    out_line = (char *)malloc((left_pad+width+right_pad)*sizeof(char));
    line_num = 0;
    while(1)
    {
        if ((line_num % 1000) == 0)
        {
            printf("Line: %d\n",line_num);
        }
        count = fread(in_line,sizeof(char),width,in);
        if ( count != width )
        {
            printf("%d Total Lines %d\n",line_num,count);
            break;
        }
        // Add the left pad
        for(i=0;i<left_pad;i++)
        {
            out_line[i] = pad_value;
        }
        // Add the actual data
        for(i=left_pad;i<(left_pad+width);i++)
        {
            out_line[i] = in_line[i-left_pad];
        }
        // Add the right pad
        for(i=(left_pad+width);i<(left_pad+width+right_pad);i++)
        {
            out_line[i] = pad_value;
        }
        fwrite(out_line,sizeof(char),(left_pad+width+right_pad),out);
        line_num++;
    }

    free(in_line);
    free(out_line);
    fclose(in);
    fclose(out);

    return EXIT_SUCCESS;
}
