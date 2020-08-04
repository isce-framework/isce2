//////////////////////////////////////
// Cunren Liang, NASA JPL/Caltech
// Copyright 2015-2018...
//////////////////////////////////////


#include "resamp.h"
#include <fftw3.h>
#include <omp.h>

#define SWAP4(a) (*(unsigned int *)&(a) = (((*(unsigned int *)&(a) & 0x000000ff) << 24) | ((*(unsigned int *)&(a) & 0x0000ff00) << 8) | ((*(unsigned int *)&(a) >> 8) & 0x0000ff00) | ((*(unsigned int *)&(a) >> 24) & 0x000000ff)))

int rg_filter(char *inputfile, int nrg, int naz, int nout, char **outputfile, float *bw, float *bc, int nfilter, int nfft, float beta, int zero_cf, float offset, int byteorder, long imageoffset, long lineoffset){
  /*
  inputfile:  input file
  nrg         file width
  nout:       number of output files
  outputfile: (value_of_out_1, value_of_out_2, value_of_out_3...) output files
  bw:         (value_of_out_1, value_of_out_2, value_of_out_3...) filter bandwidth divided by sampling frequency [0, 1]
  bc:         (value_of_out_1, value_of_out_2, value_of_out_3...) filter center frequency divided by sampling frequency

  nfilter:    number samples of the filter (odd). Reference Value: 65
  nfft:       number of samples of the FFT. Reference Value: 1024
  beta:       kaiser window beta. Reference Value: 1.0
  zero_cf:    if bc != 0.0, move center frequency to zero? 0: Yes (Reference Value). 1: No.
  offset:     offset (in samples) of linear phase for moving center frequency. Reference Value: 0.0

  byteorder:      (0) LSB, little endian; (1) MSB, big endian of intput file
  imageoffset:    offset from start of the image of input file
  lineoffset:     length of each line of input file
  */

///////////////////////////////
  // int k;
  // printf("input parameters:");
  // printf("%s\n", inputfile);
  // printf("%d\n", nrg);
  // printf("%d\n", nout);

  // for(k =0; k<nout;k++){
  //   printf("%s\n", outputfile[k]);
  //   printf("%f\n", bw[k]);
  //   printf("%f\n", bc[k]);
  // }

  // printf("%d\n", nfilter);
  // printf("%d\n", nfft);
  // printf("%f\n", beta);
  // printf("%d\n", zero_cf);
  // printf("%f\n", offset);
///////////////////////////////


  FILE *infp;   //secondary image to be resampled
  FILE **outfp;  //resampled secondary image

  fcomplex **filter;
  fcomplex *in;
  fcomplex **out;
  fcomplex *tmp;
  fcomplex *tmp2;
  fcomplex *tmpf;
  int *zeroflag;

  fftwf_plan p_forward;
  fftwf_plan p_backward;
  fftwf_plan p_forward_filter;
  //fftwf_plan p_backward_filter;

  //int nout; //number of output files
  //int nrg; //file width
  //int naz; //file length

  //int nfft; //fft length
  //int nfilter; //filter length
  int hnfilter;

  //float *bw;
  //float *bc;
  //float beta; //kaiser window beta

  //int zero_cf;
  //float offset;
  int argc_mand;
  int nthreads;

  float sc; //constant to scale the data read in to avoid large values
            //during fft and ifft
  float cf_pha;
  float t;
  fcomplex cf;

  int nblock_in;
  int nblock_out;
  int num_block;
  int i_block;
  int nblock_in_last;
  int nblock_out_last;

  int i, j, i_out;



/*****************************************************************************/
  //nfilter = 65;
  //nfft = 1024;
  //beta = 1.0;
  //zero_cf = 0;
  //offset = 0.0;

  sc = 10000.0;
/*****************************************************************************/

  infp  = openfile(inputfile, "rb");
  //naz  = file_length(infp, nrg, sizeof(fcomplex));
  //fseeko(infp,0L,SEEK_END);
  //naz = (ftello(infp) - imageoffset) / (lineoffset + nrg*sizeof(fcomplex));
  //rewind(infp);
  printf("file width: %d, file length: %d\n\n", nrg, naz);
  if(nout < 1){
    fprintf(stderr, "there should be at least one output file!\n");
    exit(1);
  }
  outfp = array1d_FILE(nout);
  for(i = 0; i < nout; i++){
    outfp[i] = openfile(outputfile[i], "wb");
  }

  //check filter length
  if(nfilter < 3){
    fprintf(stderr, "filter length: %d too small!\n", nfilter);
    exit(1);
  }
  if(nfilter % 2 != 1){
    fprintf(stderr, "filter length must be odd!\n");
    exit(1);
  }

  if(byteorder == 0){
    printf("inputfile byte order: little endian\n");
  }
  else{
    printf("inputfile byte order: big endian\n");
  }
  printf("input file image offset [byte]: %ld\n", imageoffset);
  printf("input file line offset [byte]: %ld\n", lineoffset);
  if(imageoffset < 0){
    fprintf(stderr, "image offset must be >= 0\n");
    exit(1);
  }
  if(lineoffset < 0){
    fprintf(stderr, "lineoffset offset must be >= 0\n");
    exit(1);
  }

  //compute block processing parameters
  hnfilter = (nfilter - 1) / 2;
  nblock_in = nfft - nfilter + 1;
  nblock_in += hnfilter;
  if (nblock_in <= 0){
    fprintf(stderr, "fft length too small compared with filter length!\n");
    exit(1);
  }
  nblock_out = nblock_in - 2 * hnfilter;
  num_block = (nrg - 2 * hnfilter) / nblock_out;
  if((nrg - num_block * nblock_out - 2 * hnfilter) != 0){
    num_block += 1;
  }
  if((nrg - 2 * hnfilter) <= 0){
    num_block = 1;
  }
  if(num_block == 1){
    nblock_out_last = 0;
    nblock_in_last = nrg;
  }
  else{
    nblock_out_last = nrg - (num_block - 1) * nblock_out - 2 * hnfilter;
    nblock_in_last = nblock_out_last + 2 * hnfilter;
  }

  //allocate memory
  filter = array2d_fcomplex(nout, nfft);
  in     = array1d_fcomplex(nrg);
  out    = array2d_fcomplex(nout, nrg);
  tmp    = array1d_fcomplex(nfft);
  tmp2   = array1d_fcomplex(nfft);
  tmpf   = array1d_fcomplex(nfft);
  zeroflag = array1d_int(nrg);

  //as said in the FFTW document,
  //Typically, the problem will have to involve at least a few thousand data points before threads become beneficial.
  //so I choose not to use Multi-threaded FFTW, as our FFT size is mostly small.
  if(0){
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //Multi-threaded FFTW
    nthreads = fftwf_init_threads();
    if(nthreads == 0){
      fprintf(stderr, "WARNING: there is some error in using multi-threaded FFTW.\n");
      fprintf(stderr, "         therefore it is not used, and computation performance is reduced.\n");
      nthreads = 1;
    }
    else{
      //int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
      //nthreads = omp_get_num_threads();
      nthreads = omp_get_max_threads();
    }
    printf("FFTW is using %d threads\n", nthreads);

    //this works for all the following plans
    if(nthreads != 1)
      //actually it is OK to pass nthreads=1, in this case, threads are disabled.
      fftwf_plan_with_nthreads(nthreads);
    //////////////////////////////////////////////////////////////////////////////////////////////////
  }

  //create plans before initializing data, because FFTW_MEASURE overwrites the in/out arrays.
  p_forward = fftwf_plan_dft_1d(nfft, (fftwf_complex*)tmp, (fftwf_complex*)tmp, FFTW_FORWARD, FFTW_MEASURE);
  p_backward = fftwf_plan_dft_1d(nfft, (fftwf_complex*)tmp2, (fftwf_complex*)tmp2, FFTW_BACKWARD, FFTW_MEASURE);
  p_forward_filter = fftwf_plan_dft_1d(nfft, (fftwf_complex*)tmpf, (fftwf_complex*)tmpf, FFTW_FORWARD, FFTW_ESTIMATE);

  //computing filters
  for(i = 0; i < nout; i++){
    bandpass_filter(bw[i], bc[i], nfilter, nfft, (nfilter-1)/2, beta, tmpf);

    //relationship of nr and matlab fft
    //nr fft           matlab fft
    //  1      <==>     ifft()*nfft
    // -1      <==>     fft()

    //four1((float *)filter - 1, nfft, -1);
    fftwf_execute(p_forward_filter);
    for(j = 0; j < nfft; j++){
      filter[i][j].re = tmpf[j].re;
      filter[i][j].im = tmpf[j].im;
    }
  }
  fftwf_destroy_plan(p_forward_filter);  


  //skip image header
  if(imageoffset != 0)
    fseek(infp, imageoffset, SEEK_SET);

  //process data
  for(i = 0; i < naz; i++){
    //progress report
    if((i + 1) % 1000 == 0 || (i + 1) == naz)
      fprintf(stderr,"processing line: %6d of %6d\r", i+1, naz);
    if((i + 1) == naz)
      fprintf(stderr,"\n\n");
  
    //read data
    if(i != 0)
      fseek(infp, lineoffset-(size_t)nrg * sizeof(fcomplex), SEEK_CUR);
    readdata((fcomplex *)in, (size_t)nrg * sizeof(fcomplex), infp);
    //swap bytes
    if(byteorder!=0){
      for(j = 0; j < nrg; j++){
        SWAP4(in[j].re);
        SWAP4(in[j].im);
      }
    }

    #pragma omp parallel for private(j) shared(nrg,in, zeroflag, sc)
    for(j = 0; j < nrg; j++){
      if(in[j].re != 0.0 || in[j].im != 0.0){
        zeroflag[j] = 1;
        in[j].re *= 1.0 / sc;
        in[j].im *= 1.0 / sc;
      }
      else{
        zeroflag[j] = 0;
      }
    }

    //process each block
    for(i_block = 0; i_block < num_block; i_block++){
      //zero out
      //for(j = 0; j < nfft; j++){
      //  tmp[j].re = 0.0;
      //  tmp[j].im = 0.0;
      //}
      memset((void *)tmp, 0, (size_t)nfft*sizeof(fcomplex));

      //get data
      if(num_block == 1){
        for(j = 0; j < nrg; j++){
          tmp[j] = in[j];
        }
      }
      else{
        if(i_block == num_block - 1){
          for(j = 0; j < nblock_in_last; j++){
            tmp[j] = in[j+nblock_out*i_block];
          }
        }
        else{
          for(j = 0; j < nblock_in; j++){
            tmp[j] = in[j+nblock_out*i_block];
          }
        }
      }

      //four1((float *)tmp - 1, nfft, -1);
      //tested, the same as above
      fftwf_execute(p_forward);

      //process each output file
      for(i_out = 0; i_out < nout; i_out++){
        //looks like this makes it slower, so comment out
        //#pragma omp parallel for private(j) shared(nfft, tmp2, filter, i_out, tmp)
        for(j = 0; j < nfft; j++)
          tmp2[j] = cmul(filter[i_out][j], tmp[j]);      

        //four1((float *)tmp2 - 1, nfft, 1);
        //tested, the same as above
        fftwf_execute(p_backward);

        //get data
        if(num_block == 1){
          for(j = 0; j < nrg; j++){
            out[i_out][j] = tmp2[j];
          }
        }
        else{
          if(i_block == 0){
            for(j = 0; j < hnfilter + nblock_out; j++){
              out[i_out][j] = tmp2[j];
            }
          }
          else if(i_block == num_block - 1){
            for(j = 0; j < hnfilter + nblock_out_last; j++){
              out[i_out][nrg - 1 - j] = tmp2[nblock_in_last - 1 - j];
            }
          }
          else{
            for(j = 0; j < nblock_out; j++){
              out[i_out][j + hnfilter + i_block * nblock_out] = tmp2[j + hnfilter];
            }
          }
        }//end of getting data
      }//end of processing each output file
    }//end of processing each block

    //move center frequency
    if(zero_cf == 0){
      //process each output file
      //looks like this makes it slower, so comment out
      //#pragma omp parallel for private(i_out, j, t, cf_pha, cf) shared(nout, bc, nrg, offset, out)
      for(i_out = 0; i_out < nout; i_out++){
        if(bc[i_out] != 0){
          #pragma omp parallel for private(j, t, cf_pha, cf) shared(nrg, offset, bc, i_out, out)
          for(j = 0; j < nrg; j++){
            //t = j - (nrg - 1.0) / 2.0; //make 0 index exactly at range center
            t = j + offset; //make 0 index exactly at range center
            cf_pha = 2.0 * PI * (-bc[i_out]) * t;
            cf.re = cos(cf_pha);
            cf.im = sin(cf_pha);
            out[i_out][j] = cmul(out[i_out][j], cf);
          }
        }
      }
    }

    //scale back and write data
    //process each output file
    for(i_out = 0; i_out < nout; i_out++){
      //scale back
      #pragma omp parallel for private(j) shared(nrg, zeroflag, out, i_out, sc, nfft)
      for(j = 0; j < nrg; j++){
        if(zeroflag[j] == 0){
          out[i_out][j].re = 0.0;
          out[i_out][j].im = 0.0;
        }
        else{
          out[i_out][j].re *= sc / nfft;
          out[i_out][j].im *= sc / nfft;     
        }
      }
      //write data
      writedata((fcomplex *)out[i_out], nrg * sizeof(fcomplex), outfp[i_out]);
    }
  }//end of processing data

  fftwf_destroy_plan(p_forward);
  fftwf_destroy_plan(p_backward);

  free_array2d_fcomplex(filter);
  free_array1d_fcomplex(in);
  free_array2d_fcomplex(out);
  free_array1d_fcomplex(tmp);
  free_array1d_fcomplex(tmp2);
  free_array1d_fcomplex(tmpf);
  free_array1d_int(zeroflag);
  //free_array1d_float(bw);
  //free_array1d_float(bc);

  fclose(infp);
  for(i_out = 0; i_out < nout; i_out++)
    fclose(outfp[i_out]);
  //free_array1d_FILE(outfp);

  return 0;
}//end main()


