//////////////////////////////////////
// Cunren Liang, NASA JPL/Caltech
// Copyright 2017
//////////////////////////////////////


// program for mosaicking multiple consecutive subswaths
// Cunren Liang, 03-JUN-2015
// JPL/Caltech

//////////////////////////////////////////////////////////////////
//update history
//12-APR-2016, CL. output data of both adjacent subswaths as BIL
//                 file, instead of output the difference.
//////////////////////////////////////////////////////////////////


#include "resamp.h"

//int main(int argc, char *argv[]){
int mosaicsubswath(char *outputfile, int nrgout, int nazout, int delta, int diffflag, int n, char **inputfile, int *nrgin, int *nrgoff, int *nazoff, float *phc, int *oflag){

  FILE **infp;
  FILE *outfp;
  
  fcomplex **in;
  fcomplex *out, out1, out2;
  //fcomplex *leftoverlap;
  //fcomplex *rightoverlap;
  fcomplex tmp;
  int cnt;
  
  //int n;
  //int *nrgin;
  int *nazin;
  //int *nrgoff;
  //int *nazoff;
  //int *oflag;
  //int nrgout;
  //int nazout;
  int nrginmax;

  int los, loe, low; //start, end and width of left overlap area
  int ros, roe, row; //start, end and width of right overlap area
  int cnw; //width of center area

  int paroff;
  int parcyc;

  char diffname[256][256];
  FILE **difffp;
  fcomplex **diff;
  fcomplex **diff2;
  //int diffflag;
  //diffflag = 0;

  int ns;
  float r;

  int i, j, k, l;

  //int delta; //edge to be removed of the overlap area (number of samples)
  //delta = 20;


  // if(argc < 5){
  //   fprintf(stderr, "\nUsage: %s outputfile nrgout nazout delta diffflag n [inputfile0] [nrgin0] [nrgoff0] [nazoff0] [oflag0] (repeat...)\n\n", argv[0]);
  //   fprintf(stderr, "  for each frame\n");
  //   fprintf(stderr, "  range offset is relative to the first sample of last subswath\n");
  //   fprintf(stderr, "  azimuth offset is relative to the uppermost line\n\n");
  //   exit(1);
  // }


  //read mandatory parameters
  outfp    = openfile(outputfile, "wb");
  //nrgout   = atoi(argv[2]);
  //nazout   = atoi(argv[3]);
  //delta    = atoi(argv[4]);
  //diffflag = atoi(argv[5]);
  //n        = atoi(argv[6]);


  //allocate memory
  infp   = array1d_FILE(n);
  //nrgin  = array1d_int(n);
  nazin  = array1d_int(n);
  //nrgoff = array1d_int(n); //nrgoff must be <= 0
  //nazoff = array1d_int(n); //nazoff must be <= 0
  //oflag  = array1d_int(n);

  difffp = array1d_FILE(n - 1);

  //read optional parameters
  paroff = 6;
  parcyc = 5;
  for(i = 0; i < n; i++){
    infp[i] = openfile(inputfile[i], "rb");
    //nrgin[i] = atoi(argv[paroff + parcyc*i + 2]);
    //nrgoff[i] = atoi(argv[paroff + parcyc*i + 3]);
    //nazoff[i] = atoi(argv[paroff + parcyc*i + 4]);
    //oflag[i] = atoi(argv[paroff + parcyc*i + 5]);
    nazin[i] = file_length(infp[i], nrgin[i], sizeof(fcomplex));
    if(nrgoff[i] > 0){
      fprintf(stderr,"Error: positive range offset: %d\n\n", nrgoff[i]);
      exit(1);
    }
    if(nazoff[i] > 0){
      fprintf(stderr,"Error: positive azimuth offset: %d\n\n", nazoff[i]);
      exit(1);
    }
    if(nazout < nazin[i] - nazoff[i]){
      fprintf(stderr,"Error: ouput length < nazin[%d] - nazoff[%d], %d, %d\n\n", i, i, nazout, nazin[i] - nazoff[i]);
      exit(1);
    }
  }

  //find max width
  nrginmax = nrgin[0];
  for(i = 0; i < n; i++)
    if(nrgin[i] > nrginmax)
      nrginmax = nrgin[i];

  in   = array2d_fcomplex(n, nrginmax);
  out  = array1d_fcomplex(nrgout);
  //out1 = array1d_fcomplex(nrginmax);
  //out2 = array1d_fcomplex(nrginmax);
  diff = array2d_fcomplex(n-1, nrginmax);
  diff2 = array2d_fcomplex(n-1, nrginmax);

  if(diffflag == 0)
    for(i = 0; i < n - 1; i++){
      sprintf(diffname[i], "%d-%d.int", i, i+1);
      difffp[i] = openfile(diffname[i], "wb");
    }


  for(i = 0; i < nazout; i++){

    if((i + 1) % 1000 == 0)
      fprintf(stderr,"processing line: %6d of %6d\r", i + 1, nazout);
    if(i + 1 == nazout)
      fprintf(stderr,"processing line: %6d of %6d\n\n", i + 1, nazout);

    //prepare for writing data
    for(j = 0; j < nrgout; j++){
      out[j].re = 0.0;
      out[j].im = 0.0;
    }

    //prepare for reading data
    for(j = 0; j < n; j++){
      for(k = 0; k < nrginmax; k++){
        in[j][k].re = 0.0;
        in[j][k].im = 0.0;
      }
    }

    for(j = 0; j < n; j++){
      if(i + nazoff[j] >= 0 && i + nazoff[j] <= nazin[j] - 1)
        readdata((fcomplex *)in[j], nrgin[j] * sizeof(fcomplex), infp[j]);
      
      if(phc[j]!=0.0){
        tmp.re = cos(phc[j]);
        tmp.im = sin(phc[j]);
        for(k = 0; k < nrgin[j]; k++)
          in[j][k] = cmul(in[j][k], tmp);
      }
    }


    cnt = 0;
    for(j = 0; j < n; j++){

      //we follow the following convention: line and column number start with 0.
      //left overlap area of subswath j
      if(j != 0){
        los = - nrgoff[j];
        loe = nrgin[j-1] - 1;
        low = loe - los + 1;
        if(low < delta * 2){
          fprintf(stderr,"Error: not enough overlap area between subswath: %d and %d\n\n", j-1, j);
          exit(1);
        }
      }
      else{
        los = 0;
        loe = 0;
        low = 0;
      }

      //right overlap area of subswath j
      if(j != n - 1){
        ros = - nrgoff[j+1];
        roe = nrgin[j] - 1;
        row = roe - ros + 1;
        if(row < delta * 2){
          fprintf(stderr,"Error: not enough overlap area between subswath: %d and %d\n\n", j, j+1);
          exit(1);
        }
      }
      else{
        ros = 0;
        roe = 0;
        row = 0;
      }

      //center non-overlap area of subswath j
      //should add a check here?
      cnw = nrgin[j] - low - row;

      //deal with center non-overlap area.
      //this only excludes the right overlap area for the first subswath
      //this only excludes the left overlap area for the last subswath
      for(k = 0; k < cnw; k++){
        out[cnt + k].re = in[j][low + k].re;
        out[cnt + k].im = in[j][low + k].im;
      }
      cnt += cnw;

      //deal with right overlap area of subswath j, which is also the left overlap area
      //of subswath j + 1 (next subswath)
      
      //for last subswath, just skip
      if(j == n - 1){
        break;
      }


      for(k = 0; k < nrginmax; k++){
        diff[j][k].re = 0.0;
        diff[j][k].im = 0.0;
        diff2[j][k].re = 0.0;
        diff2[j][k].im = 0.0;
      }

      for(k = 0; k < row; k++){

        
        out1.re = in[j][low + cnw + k].re;
        out1.im = in[j][low + cnw + k].im;
        out2.re = in[j+1][k].re;
        out2.im = in[j+1][k].im;

        //left edge of overlap area
        //use current subswath: subswath j
        if(k < delta){
          out[cnt + k].re = out1.re;
          out[cnt + k].im = out1.im;
        }
        else if(k >= delta && k < row - delta){

          //output difference of overlap area
          //diffflag 0: subswath j phase - subswath j+1 phase
          if(diffflag == 0){
            if(out1.re != 0.0 && out1.im != 0.0 && out2.re != 0.0 && out2.im != 0.0){
              //diff[j][k - delta] = cmul(out1, cconj(out2));
              diff[j][k - delta] = out1;
              diff2[j][k - delta] = out2;
            }
          }
          //diffflag 1: subswath j - subswath j+1
          //else if(diffflag == 1){
          //  if(out1.re != 0.0 && out1.im != 0.0 && out2.re != 0.0 && out2.im != 0.0){
          //    diff[j][k - delta].re = out1.re - out2.re;
          //    diff[j][k - delta].im = out1.im - out2.im;
          //  }
          //}
          else{
            ;
          }

          //mosaic overlap area
          //case 0: mosaic at the center of overlap area
          if(oflag[j] == 0){
            if(k < row / 2){
              //avoid holes, Cunren Liang, Dec. 18, 2015.
              if(out1.re != 0.0 && out1.im != 0.0){
                out[cnt + k].re = out1.re;
                out[cnt + k].im = out1.im;
              }
              else{
                out[cnt + k].re = out2.re;
                out[cnt + k].im = out2.im;                
              }
            }
            else{
              //avoid holes, Cunren Liang, Dec. 18, 2015.
              if(out2.re != 0.0 && out2.im != 0.0){
                out[cnt + k].re = out2.re;
                out[cnt + k].im = out2.im;
              }
              else{
                out[cnt + k].re = out1.re;
                out[cnt + k].im = out1.im;                
              }
            }
          }
          //case 1: mosaic at the right egde of overlap area
          else if(oflag[j] == 1){
            out[cnt + k].re = out1.re;
            out[cnt + k].im = out1.im;
          }
          //case 2: mosaic at the left edge of overlap area
          else if(oflag[j] == 2){
            out[cnt + k].re = out2.re;
            out[cnt + k].im = out2.im;
          }
          //case 3: add overlap area
          else if(oflag[j] == 3){
            out[cnt + k].re = out1.re + out2.re;
            out[cnt + k].im = out1.im + out2.im;

            if(out1.re != 0.0 && out1.im != 0.0 && out2.re != 0.0 && out2.im != 0.0){
              out[cnt + k].re /= 2.0;
              out[cnt + k].im /= 2.0;
            }

          }
          //case 4: add by weight determined by distance to overlap center
          //perform overlapp area smoothing using a method discribed in:
          //C. Liang, Q. Zeng, J. Jia, J. Jiao, and X. Cui, ScanSAR interferometric processing
          //using existing standard InSAR software for measuring large scale land deformation
          //Computers & Geosciences, 2013.
          else{
            l = k - delta + 1; // l start with 1
            ns = row - 2 * delta;
            
            if(out1.re != 0.0 && out1.im != 0.0 && out2.re != 0.0 && out2.im != 0.0){
              r = sqrt((out1.re * out1.re + out1.im * out1.im) / (out2.re * out2.re + out2.im * out2.im));
              out[cnt + k].re = ((ns - l + 0.5) * out1.re + r * (l - 0.5) * out2.re) / ns;
              out[cnt + k].im = ((ns - l + 0.5) * out1.im + r * (l - 0.5) * out2.im) / ns;
            }
            else{
              out[cnt + k].re = out1.re + out2.re;
              out[cnt + k].im = out1.im + out2.im;
            }
          }
          //cnt += row - 2 * delta;
        }
        //right edge of overlap area
        //use next subswath: subswath j+1
        //else if(k >= row - delta){
        else{
          out[cnt + k].re = out2.re;
          out[cnt + k].im = out2.im;
        }
        //cnt += 1;
      }
      cnt += row;

      if(diffflag == 0){
        writedata((fcomplex *)diff[j], (row - 2 * delta) * sizeof(fcomplex), difffp[j]);
        writedata((fcomplex *)diff2[j], (row - 2 * delta) * sizeof(fcomplex), difffp[j]);
      }

    } //loop of j, subswath
    writedata((fcomplex *)out, nrgout * sizeof(fcomplex), outfp);
  } //loop of i, output line

  for(i = 0; i < n; i++)
    fclose(infp[i]);
  fclose(outfp);

  if(diffflag == 0)
    for(i = 0; i < n - 1; i++)
      fclose(difffp[i]);

  free_array1d_FILE(infp);
  free_array1d_FILE(difffp);

  free_array2d_fcomplex(in);
  free_array1d_fcomplex(out);

  //free_array1d_int(nrgin);
  free_array1d_int(nazin);
  //free_array1d_int(nrgoff); //nrgoff must be <= 0
  //free_array1d_int(nazoff); //nazoff must be <= 0
  //free_array1d_int(oflag);

  free_array2d_fcomplex(diff);
  free_array2d_fcomplex(diff2);

  return 0;

}
