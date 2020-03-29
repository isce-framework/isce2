//////////////////////////////////////
// Cunren Liang, NASA JPL/Caltech
// Copyright 2017
//////////////////////////////////////


//a program for taking looks for single band file

#include "resamp.h"

int look(char *inputfile, char *outputfile, long nrg, int nrlks, int nalks, int ft, int sum, int avg){

  FILE *infp;
  FILE *outfp;
  
  signed char *in0;
  int *in1;
  float *in2;
  double *in3;
  fcomplex *in4;
  dcomplex *in5;

  signed char *out0;
  int *out1;
  float *out2;
  double *out3;
  fcomplex *out4;
  dcomplex *out5;

  int sum_nz;
  double sum1, sum2;

  long index;

  long naz;
  long nrg1, naz1;
  //int nrlks, nalks;

  //int ft, sum, avg;

  int i, j;
  int ii, jj;


  // if(argc < 4){
  //   fprintf(stderr, "\nUsage: %s infile outfile nrg [nrlks] [nalks] [ft] [sum] [avg]\n", argv[0]);
  //   fprintf(stderr, "  infile:  input file\n");
  //   fprintf(stderr, "  outfile: output file\n");
  //   fprintf(stderr, "  nrg:     file width\n");
  //   fprintf(stderr, "  nrlks:   number of looks in range (default: 4)\n");
  //   fprintf(stderr, "  nalks:   number of looks in azimuth (default: 4)\n");
  //   fprintf(stderr, "  ft:      file type (default: 0)\n");
  //   fprintf(stderr, "             0: signed char\n");
  //   fprintf(stderr, "             1: int\n");
  //   fprintf(stderr, "             2: float\n");
  //   fprintf(stderr, "             3: double\n");
  //   fprintf(stderr, "             4: complex (real and imagery: float)\n");
  //   fprintf(stderr, "             5: complex (real and imagery: double)\n");
  //   fprintf(stderr, "  sum:     sum method (default: 0)\n");
  //   fprintf(stderr, "             0: simple sum\n");
  //   fprintf(stderr, "             1: power sum (if complex, do this for each channel seperately)\n");
  //   fprintf(stderr, "  avg:     take average (default: 0)\n");
  //   fprintf(stderr, "             0: no\n");
  //   fprintf(stderr, "             1: yes\n\n");

  //   exit(1);
  // }

  infp  = openfile(inputfile, "rb");
  outfp = openfile(outputfile, "wb");

  //nrg = atoi(argv[3]);

  //if(argc > 4)
  //  nrlks = atoi(argv[4]);
  //else
  //  nrlks = 4;

  //if(argc > 5)
  //  nalks = atoi(argv[5]);
  //else
  //  nalks = 4;
  
  //if(argc > 6)
  //  ft = atoi(argv[6]);
  //else
  //  ft = 0;

  //if(argc > 7)
  //  sum = atoi(argv[7]);
  //else
  //  sum = 0;

  //if(argc > 8)
  //  avg = atoi(argv[8]);
  //else
  //  avg = 0;

  nrg1 = nrg / nrlks;

  if(ft == 0){
    in0 = array1d_char(nrg*nalks);
    out0 = array1d_char(nrg1);
    naz = file_length(infp, nrg, sizeof(signed char));
  }
  else if(ft == 1){
    in1 = array1d_int(nrg*nalks);
    out1 = array1d_int(nrg1);
    naz = file_length(infp, nrg, sizeof(int));
  }
  else if(ft == 2){
    in2 = array1d_float(nrg*nalks);
    out2 = array1d_float(nrg1);
    naz = file_length(infp, nrg, sizeof(float));
  }
  else if(ft == 3){
    in3 = array1d_double(nrg*nalks);
    out3 = array1d_double(nrg1);
    naz = file_length(infp, nrg, sizeof(double));
  }
  else if(ft == 4){
    in4 = array1d_fcomplex(nrg*nalks);
    out4 = array1d_fcomplex(nrg1);
    naz = file_length(infp, nrg, sizeof(fcomplex));
  }
  else if(ft == 5){
    in5 = array1d_dcomplex(nrg*nalks);
    out5 = array1d_dcomplex(nrg1);
    naz = file_length(infp, nrg, sizeof(dcomplex));
  }
  else{
    fprintf(stderr, "Error: file type not supported.\n\n");
    exit(1);
  }

  naz1 = naz / nalks;

  for(i = 0; i < naz1; i++){

    if((i + 1) % 100 == 0)
      fprintf(stderr,"processing line: %6d of %6d\r", i+1, naz1);

    //read data
    if(ft == 0){
      readdata((signed char *)in0, (size_t)nalks * (size_t)nrg * sizeof(signed char), infp);
    }
    else if(ft == 1){
      readdata((int *)in1, (size_t)nalks * (size_t)nrg * sizeof(int), infp);
    }
    else if(ft == 2){
      readdata((float *)in2, (size_t)nalks * (size_t)nrg * sizeof(float), infp);
    }
    else if(ft == 3){
      readdata((double *)in3, (size_t)nalks * (size_t)nrg * sizeof(double), infp);
    }
    else if(ft == 4){
      readdata((fcomplex *)in4, (size_t)nalks * (size_t)nrg * sizeof(fcomplex), infp);
    }
    else if(ft == 5){
      readdata((dcomplex *)in5, (size_t)nalks * (size_t)nrg * sizeof(dcomplex), infp);
    }

    //process data
    for(j = 0; j < nrg1; j++){
      //get sum
      sum_nz = 0;
      sum1 = 0.0;
      sum2 = 0.0;
      for(ii = 0; ii < nalks; ii++){
        for(jj = 0; jj < nrlks; jj++){
          index = ii * nrg + j * nrlks + jj;
          if(ft == 0){
            if(in0[index] != 0){
              if(sum == 0)
                sum1 += in0[index];
              else
                sum1 += in0[index] * in0[index];
              sum_nz += 1;
            }
          }
          else if(ft == 1){
            if(in1[index] != 0){
              if(sum == 0)
                sum1 += in1[index];
              else
                sum1 += in1[index] * in1[index];
              sum_nz += 1;
            }
          }
          else if(ft == 2){
            if(in2[index] != 0){
              if(sum == 0)
                sum1 += in2[index];
              else
                sum1 += in2[index] * in2[index];
              sum_nz += 1;
            }
          }
          else if(ft == 3){
            if(in3[index] != 0){
              if(sum == 0)
                sum1 += in3[index];
              else
                sum1 += in3[index] * in3[index];
              sum_nz += 1;
            }
          }
          else if(ft == 4){
            if(in4[index].re != 0 || in4[index].im != 0){
              if(sum ==0){
                sum1 += in4[index].re;
                sum2 += in4[index].im;
              }
              else{
                sum1 += in4[index].re * in4[index].re;
                sum2 += in4[index].im * in4[index].im;
              }
              sum_nz += 1;
            }
          }
          else if(ft == 5){
            if(in5[index].re != 0 || in5[index].im != 0){
              if(sum == 0){
                sum1 += in5[index].re;
                sum2 += in5[index].im;
              }
              else{
                sum1 += in5[index].re * in5[index].re;
                sum2 += in5[index].im * in5[index].im;
              }
              sum_nz += 1;
            }
          }
        }
      }

      //preprocessing
      if(avg == 1){
        if(sum_nz != 0){
          sum1 /= sum_nz;
          if(ft == 4 || ft == 5)
            sum2 /= sum_nz;
        }
      }
      if(sum == 1){
        if(sum_nz != 0){
          sum1 = sqrt(sum1);
          if(ft == 4 || ft ==5)
            sum2 = sqrt(sum2);
        }
      }

      //get data
      if(ft == 0){
        out0[j] = (signed char)(roundfi(sum1));
      }
      else if(ft == 1){
        out1[j] = (int)(roundfi(sum1));
      }
      else if(ft == 2){
        out2[j] = sum1;
      }
      else if(ft == 3){
        out3[j] = sum1;
      }
      else if(ft == 4){
        out4[j].re = sum1;
        out4[j].im = sum2;
      }
      else if(ft == 5){
        out5[j].re = sum1;
        out5[j].im = sum2;
      }
    }

    //write data
    if(ft == 0){
      writedata((signed char *)out0, nrg1 * sizeof(signed char), outfp);
    }
    else if(ft == 1){
      writedata((int *)out1, nrg1 * sizeof(int), outfp);
    }
    else if(ft == 2){
      writedata((float *)out2, nrg1 * sizeof(float), outfp);
    }
    else if(ft == 3){
      writedata((double *)out3, nrg1 * sizeof(double), outfp);
    }
    else if(ft == 4){
      writedata((fcomplex *)out4, nrg1 * sizeof(fcomplex), outfp);
    }
    else if(ft == 5){
      writedata((dcomplex *)out5, nrg1 * sizeof(dcomplex), outfp);
    }
  }
  fprintf(stderr,"processing line: %6d of %6d\n", naz1, naz1);

  //clear up
  if(ft == 0){
    free_array1d_char(in0);
    free_array1d_char(out0);
  }
  else if(ft == 1){
    free_array1d_int(in1);
    free_array1d_int(out1);
  }
  else if(ft == 2){
    free_array1d_float(in2);
    free_array1d_float(out2);
  }
  else if(ft == 3){
    free_array1d_double(in3);
    free_array1d_double(out3);
  }
  else if(ft == 4){
    free_array1d_fcomplex(in4);
    free_array1d_fcomplex(out4);
  }
  else if(ft == 5){
    free_array1d_dcomplex(in5);
    free_array1d_dcomplex(out5);
  }
  fclose(infp);
  fclose(outfp);

  return 0;
}
