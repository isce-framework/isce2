//////////////////////////////////////
// Cunren Liang, NASA JPL/Caltech
// Copyright 2017
//////////////////////////////////////


#include "resamp.h"

FILE *openfile(char *filename, char *pattern){
  FILE *fp;
  
  fp=fopen(filename, pattern);
  if (fp==NULL){
    fprintf(stderr,"Error: cannot open file: %s\n", filename);
    exit(1);
  }

  return fp;
}

void readdata(void *data, size_t blocksize, FILE *fp){
  if(fread(data, blocksize, 1, fp) != 1){
    fprintf(stderr,"Error: cannot read data\n");
    exit(1);
  }
}

void writedata(void *data, size_t blocksize, FILE *fp){
  if(fwrite(data, blocksize, 1, fp) != 1){
    fprintf(stderr,"Error: cannot write data\n");
    exit(1);
  }
}

long file_length(FILE* fp, long width, long element_size){
  long length;
  
  fseeko(fp,0L,SEEK_END);
  length = ftello(fp) / element_size / width;
  rewind(fp);
  
  return length;
}
