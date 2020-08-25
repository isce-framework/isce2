//////////////////////////////////////
// Cunren Liang, NASA JPL/Caltech
// Copyright 2017
//////////////////////////////////////


#include "resamp.h"
#include <fftw3.h>

#define SWAP4(a) (*(unsigned int *)&(a) = (((*(unsigned int *)&(a) & 0x000000ff) << 24) | ((*(unsigned int *)&(a) & 0x0000ff00) << 8) | ((*(unsigned int *)&(a) >> 8) & 0x0000ff00) | ((*(unsigned int *)&(a) >> 24) & 0x000000ff)))


int extract_burst(char *inputf, char *outputf, int nrg, int naz, float prf, float prf_frac, float nb, float nbg, float bsl, float *kacoeff, float *dopcoeff, float az_ratio, float min_line_offset, int byteorder, long imageoffset, long lineoffset){
  FILE *infp;
  FILE *outfp;
  FILE *infofp;

  char output_filename[512];
  char burst_num[512];

  fcomplex **in; //data read in
  fcomplex **out; //data written to output file
  fcomplex **filter; //multi-band bandpass filter
  fcomplex **deramp; //deramp signal

  fcomplex *data; //data to be filtered.

  //int nrg; //file width
  //int naz; //file length
  int naz_burst_in; //number of lines of the input burst
  int naz_burst_out; //number of lines of the output burst

  //float prf;
  float pri; // 1.0/prf
  //float prf_frac; // azimuth bandwidth used for burst extraction = prf_frac * prf
  //float kacoeff[3]; //FM rate along range (experessed by quadratic polynomial
                    //as a function of range sample number)
  //float dopcoeff[4]; //doppler centroid frequency along range (expressed by quadratic polynomial
                      //as a function of range sample number). this image

  //float nb; //burst length in terms of pri. number of lines
  //float nbg; //burst gap length in terms of pri. number of lines
  //float bsl; //burst start line, input float
  float bcl; //burst center line, float
  float bfw; //burst bandwidth
  //float az_ratio; //azimuth sampling interval of output burst: az_ratio * pri;

  float *ka; //azimuth fm rate
  float *dop; //doppler centroid frequency
  float *nfa;

  float *start_line; //burst imaged area start line number for each column
  float *end_line; //burst imaged area ending line number for each column

  float min_line; //minimum of start_line
  float max_line; //maximum of end_line
  int min_line_column; //column of minimum
  int max_line_column; //column of maximum

  int *output_start; //relative start line in the output burst for each column
  int *output_end; //relative end line in the output burst for each column

  int min_line_in; //first line of input burst
  float min_line_out; //first line of output burst
  float min_line_out_first; //first line of first output burst
  int offset_from_first_burst; //offset between first burst and this burst in az_ratio * pri
  int offset_from_first_burst0; //offset between first burst and last burst in az_ratio * pri

  //float min_line_offset; // the offset of first line of output burst in pri, compared with roundfi(min_line)
                         // this is mainly used to adjust the output burst location. so usually can be set to 0 

  int burst_index;
  int burst_index_tmp;
  int burst_index_first;

  int nfft; //fft length
  int nfilter; //filter length, MUST BE ODD
  int hnfilter; //half filter length
  float beta; //kaiser window beta of filter
  float sc; //constant to scale the data read in to avoid large values during fft and ifft

  //resampling parameters
  float beta_resamp; //beta of kaiser window of resampling kernal
  int n; //number of samples to be used in the resampling(odd)
  int m; //multiples of n, so that a lookup table can be generated(even)

  int hn;  //half of n
  int hnm; //half of n*m
  float *sincc; //sinc coefficents
  float *kaiserc; // kaiser window coefficents
  float *kernel; // sincc*kaiserc
  
  float tmpa, tmpb, tmpc; //temperal variables effective for a longer time
  float tmp1, tmp2, tmp3; //temperal variables effective for a shorter time
  fcomplex reramp;
  int index;

  int i, j, k;

  fftwf_plan p_forward;
  fftwf_plan p_backward;

/*****************************************************************************/
//I just put these parametes which can be set here. These can also be set via
//arguments before running the programs if modifying the code to accept these
//arguments.
  //min_line_offset = 0.0;

  //filtering parameters
  beta = 1.0;
  nfilter = 257; //MUST BE ODD
  sc = 10000.0;

  //resampling parameters
  beta_resamp = 2.5;
  n = 9; //MUST BE ODD
  m = 10000; //MUST BE EVEN

/*****************************************************************************/

  if(0){
  //if(argc != 18){
    fprintf(stderr, "\nusage: %s inputf outputf nrg naz prf prf_frac nb nbg bsl kacoeff[0-3] dopcoeff[0-3] az_ratio min_line_offset byteorder imageoffset lineoffset\n");
    fprintf(stderr, "\nmandatory:\n");
    fprintf(stderr, "  inputf:          input file\n");
    fprintf(stderr, "  outputf:         prefix of output files\n");
    fprintf(stderr, "  nrg:             file width\n");
    fprintf(stderr, "  naz:             file length\n");
    fprintf(stderr, "  prf:             PRF\n");
    fprintf(stderr, "  prf_frac:        fraction of PRF used for burst generation\n");
    fprintf(stderr, "                      (represents azimuth bandwidth)\n");
    fprintf(stderr, "  nb:              number of lines in a burst\n");
    fprintf(stderr, "                      (float, in terms of 1/PRF)\n");
    fprintf(stderr, "  nbg:             number of lines in a burst gap\n");
    fprintf(stderr, "                      (float, in terms of 1/PRF)\n");
    fprintf(stderr, "  bsl:             start line number of a burst\n");
    fprintf(stderr, "                      (float, the line number of the first line of the full-aperture SLC is zero)\n");
    fprintf(stderr, "                      (no need to be first burst, any one is OK)\n");

    fprintf(stderr, "  kacoeff[0-3]:    FM rate coefficients\n");
    fprintf(stderr, "                      (four coefficients of a third order polynomial with regard to)\n");
    fprintf(stderr, "                      (range sample number. range sample number starts with zero)\n");

    fprintf(stderr, "  dopcoeff[0-3]:   Doppler centroid frequency coefficients of this image\n");
    fprintf(stderr, "                      (four coefficients of a third order polynomial with regard to)\n");
    fprintf(stderr, "                      (range sample number. range sample number starts with zero)\n");
    fprintf(stderr, "  az_ratio:        line interval of output burst: az_ratio * (1/PRF)\n");
    fprintf(stderr, "  min_line_offset: adjust output line location by this offset\n");
    fprintf(stderr, "                      (in terms of 1/PRF, within [-50/PRF, 50/PRF])\n");
    fprintf(stderr, "                      (offset < 0, start earlier than original)\n");
    fprintf(stderr, "                      (offset = 0, original)\n");
    fprintf(stderr, "                      (offset > 0, start later than original)\n");

    fprintf(stderr, "byteorder:            (0) LSB, little endian; (1) MSB, big endian of intput file\n");
    fprintf(stderr, "imageoffset:          offset from start of the image of input file\n");
    fprintf(stderr, "lineoffset:           length of each line of input file\n\n");

    exit(1);
  }

  infofp  = openfile("extract_burst.txt", "w");

  //open files
  infp  = openfile(inputf, "rb");
  //nrg  = atoi(argv[3]);
  //prf = atof(argv[4]);
  //prf_frac = atof(argv[5]);
  //nb = atof(argv[6]);
  //nbg = atof(argv[7]);
  //bsl = atof(argv[8]);

  //kacoeff[0] = atof(argv[9]);
  //kacoeff[1] = atof(argv[10]);
  //kacoeff[2] = atof(argv[11]);

  //dopcoeff[0] = atof(argv[12]);
  //dopcoeff[1] = atof(argv[13]);
  //dopcoeff[2] = atof(argv[14]);
  //dopcoeff[3] = atof(argv[15]);

  //az_ratio = atof(argv[16]);
  //min_line_offset = atof(argv[17]);


  fprintf(infofp, "\n\ninput parameters:\n");
  fprintf(infofp, "input file: %s\n", inputf);
  fprintf(infofp, "prefix of output files: %s\n", outputf);
  fprintf(infofp, "nrg: %d\n", nrg);
  fprintf(infofp, "prf: %f\n", prf);
  fprintf(infofp, "prf_frac: %f\n", prf_frac);
  fprintf(infofp, "nb: %f\n", nb);
  fprintf(infofp, "nbg: %f\n", nbg);
  fprintf(infofp, "bsl: %f\n", bsl);

  fprintf(infofp, "kacoeff: %f, %f, %f, %f\n", kacoeff[0], kacoeff[1], kacoeff[2], kacoeff[3]);
  fprintf(infofp, "dopcoeff1: %f, %f, %f, %f\n", dopcoeff[0], dopcoeff[1], dopcoeff[2], dopcoeff[3]);

  fprintf(infofp, "az_ratio: %f\n", az_ratio);
  fprintf(infofp, "offset: %f\n\n", min_line_offset);

  if(fabs(min_line_offset) > 50.0){
    fprintf(stderr, "offset too big!\n");
    exit(1);
  }

  if(nfilter % 2 != 1){
    fprintf(stderr, "filter length must be odd!\n");
    exit(1);
  }

  if(n % 2 != 1){
    fprintf(stderr, "resample kernal length must be odd!\n");
    exit(1);
  }
  if(n < 7){
    fprintf(stderr, "resample kernal length too small!\n");
    exit(1);
  }

  if(m % 2 != 0){
    fprintf(stderr, "m must be even!\n");
    exit(1);
  }
  if(m < 1000){
    fprintf(stderr, "m too small!\n");
    exit(1);
  }

  pri = 1.0/prf;
  hnfilter = (nfilter - 1) / 2;

  hn = n / 2;
  hnm = n * m / 2;


  //naz  = file_length(infp, nrg, sizeof(fcomplex));
  fprintf(infofp, "file width: %d, file length: %d\n\n", nrg, naz);

  ka = array1d_float(nrg);
  dop = array1d_float(nrg);
  nfa = array1d_float(nrg);

  start_line = array1d_float(nrg);
  end_line = array1d_float(nrg);
  output_start = array1d_int(nrg);
  output_end = array1d_int(nrg);

  sincc = vector_float(-hnm, hnm);
  kaiserc = vector_float(-hnm, hnm);
  kernel = vector_float(-hnm, hnm);

  //initialize sinc coefficents
  sinc(n, m, sincc);
  kaiser(n, m, kaiserc, beta_resamp);
  for(i = -hnm; i <= hnm; i++)
    kernel[i] = kaiserc[i] * sincc[i];

  //calculate some range variant variables
  for(i = 0; i < nrg; i++){
    //azimuth FM rate. we follow the convention ka > 0
    ka[i] = kacoeff[3] * i * i * i + kacoeff[2] * i * i + kacoeff[1] * i + kacoeff[0];
    ka[i] = -ka[i];
    
    //doppler centroid frequency
    dop[i] = dopcoeff[0] + dopcoeff[1] * i + dopcoeff[2] * i * i + dopcoeff[3] * i * i * i;
    //dop[i] *= prf;

    //full-aperture length
    nfa[i] = prf * prf_frac / ka[i] / pri;
  }

  tmp1 = -1.0; //maximum oversampling ratio
  tmp2 = 10000000000.0; //minimum oversampling ratio
  for(i = 0; i < nrg; i++){
    tmp3 = 1.0 / (az_ratio * pri) / (nb * pri * ka[i]);
    if(tmp3 > tmp1)
      tmp1 = tmp3;
    if(tmp3 < tmp2)
      tmp2 = tmp3;
  }

  fprintf(infofp, "azimuth oversampling ratio of output burst, minimum: %6.2f, maximum: %6.2f\n\n", tmp2, tmp1);


  //find burst starting line closest to first line and after first line
  //to make sure the bsl used in the following is not too big to avoid overflow
  //bsl is defined by 0 = first line of input SLC, this defines the absolute line numbers used in the following
  //here we stop at burst_index_tmp
  for(i = -100000; i < 100000; i++){
    tmp1 = bsl + (nb + nbg) * i;
    if(tmp1 >= 0){
      bsl = tmp1;
      burst_index_tmp = i;
      break;
    }
  }

  //starting and ending lines for each column
  for(i = 0; i < nrg; i++){
    //starting index
    start_line[i] = bsl + (nb - 1.0) / 2.0 + dop[i] / ka[i] / pri - (nfa[i] - nb - 1.0) / 2.0;
    //ending index
    end_line[i] = bsl + (nb - 1.0) / 2.0 + dop[i] / ka[i] / pri + (nfa[i] - nb - 1.0) / 2.0;
  }

  //starting and ending lines for the whole block
  min_line = start_line[0];
  max_line = end_line[0];
  for(i = 0; i < nrg; i++){
  	if(start_line[i] <= min_line){
  	  min_line = start_line[i];
  	  min_line_column = i;
  	}
  	if(end_line[i] >= max_line){
  	  max_line = end_line[i];
  	  max_line_column = i;
  	}
  }

  //number of lines of the input burst
  naz_burst_in = roundfi((max_line - min_line) + 2 * hnfilter);
  //number of lines of the output burst
  naz_burst_out = roundfi((max_line - min_line) / az_ratio);
  //to make circular convolution equivalent to linear convolution
  nfft = next_pow2(naz_burst_in + nfilter - 1);

  fprintf(infofp, "for all the output bursts:\n");
  fprintf(infofp, "input data block length: %d\n", naz_burst_in);
  fprintf(infofp, "output burst length: %d\n", naz_burst_out);
  fprintf(infofp, "fft length: %d\n\n", nfft);

  //calculate relative start and end lines in the output burst
  for(i = 0; i < nrg; i++){
    output_start[i] = roundfi((start_line[i] - min_line) / az_ratio); //made sure: first line has output. Include this start line
    output_end[i] = naz_burst_out - 1 + roundfi((end_line[i] - max_line) / az_ratio); //made sure: last line has output. Include this end line
  }

  in  = array2d_fcomplex(naz_burst_in, nrg);
  out = array2d_fcomplex(naz_burst_out, nrg);
  deramp  = array2d_fcomplex(naz_burst_in, nrg);
  filter = array2d_fcomplex(nrg, nfft);
  data = array1d_fcomplex(nfft);

  fprintf(infofp, "calculating filter...\n\n");

  //create plans before initializing data, because FFTW_MEASURE overwrites the in/out arrays.
  p_forward = fftwf_plan_dft_1d(nfft, (fftwf_complex*)data, (fftwf_complex*)data, FFTW_FORWARD, FFTW_ESTIMATE);
  p_backward = fftwf_plan_dft_1d(nfft, (fftwf_complex*)data, (fftwf_complex*)data, FFTW_BACKWARD, FFTW_ESTIMATE);

  //create filter, ZERO center frequency for all columns
  for(i = 0; i < nrg; i++){
    bfw = nb * pri * ka[i];
    //create filter: first sample corresponding to first fully convolution sample
    bandpass_filter(bfw/prf, 0.0/prf, nfilter, nfft, (nfilter-1)/2, beta, filter[i]);
    //forward fft
    //four1((float *)filter[i] - 1, nfft, -1);
    //data = filter[i];
    memcpy((void *) data, (const void *) filter[i], (size_t) nfft * sizeof(fcomplex));
    fftwf_execute(p_forward);
    //filter[i] = data;
    memcpy((void *) filter[i], (const void *) data, (size_t) nfft * sizeof(fcomplex));
  }


  //let's extract burst now, start from burst_index_tmp where we stop last time
  burst_index_first = burst_index_tmp - 1;
  tmpa = min_line; //save min_line caculated last time
  offset_from_first_burst0 = 0;
  for(burst_index = burst_index_tmp; burst_index < 100000; burst_index++){
    
    //burst first line number
    tmpb = bsl + (burst_index - burst_index_tmp) * (nb + nbg);  
    //burst center line number
    bcl = tmpb + (nb - 1.0) / 2.0;
    //minimum line of imaged area of the burst
  	min_line = tmpa + (burst_index - burst_index_tmp) * (nb + nbg);
    //minimum line of input burst
  	min_line_in = roundfi(min_line) - hnfilter;

    //skip bursts that are not or partly in the image
    if(min_line_in < 0)
      continue;
    //stop at last burst that is fully in the image
    if(min_line_in + naz_burst_in - 1 > naz - 1)
      break;
    

/*********************************************************
                   (int)
                 min_line_in
                   ------
    (float)        |    |          (float)
   min_line        |    |         min_line_out
    ------         |    |           ------           
    |    |         |    |           |    |
    |    |  ====>  |    |   ====>   |    |
    |    |         |    |           |    |
    ------         |    |           ------  
 burst imaged      |    |         output burst
     area          ------
                burst read in
*********************************************************/

    //first burst
    if(burst_index_first == burst_index_tmp - 1){
      burst_index_first = burst_index;

      min_line_out = roundfi(min_line) + min_line_offset;
      
      min_line_out_first = min_line_out;
      offset_from_first_burst = 0;

      fprintf(infofp, "line number of first line of original SLC is 0.\n");
      fprintf(infofp, "line number of first line of first output burst in original SLC (1.0/prf): %f\n", min_line_out);
      fprintf(infofp, "bsl of first output burst: %f\n\n", tmpb);
    }
    //adjust starting line of following bursts
    else{
      min_line_out = min_line + min_line_offset;
      offset_from_first_burst = roundfi((min_line_out - min_line_out_first) / az_ratio);
      tmp1 = offset_from_first_burst - (min_line_out - min_line_out_first) / az_ratio;
      min_line_out = min_line_out + tmp1 * az_ratio;
    }
   
    fprintf(infofp, "extracting burst %3d\n", burst_index - burst_index_first + 1);
    fprintf(infofp, "offset from first burst: %5d, offset from last burst: %5d (az_ratio/prf)\n\n", offset_from_first_burst, offset_from_first_burst - offset_from_first_burst0);
    offset_from_first_burst0 = offset_from_first_burst;

    //read data block
    //fseeko(infp, (size_t)min_line_in * (size_t)nrg * sizeof(fcomplex), SEEK_SET);
    //readdata((fcomplex *)in[0], (size_t)naz_burst_in * (size_t)nrg * sizeof(fcomplex), infp);

    fseeko(infp, (size_t)imageoffset + (size_t)min_line_in * (size_t)lineoffset, SEEK_SET);
    for(i = 0; i < naz_burst_in; i++){
      if(i!=0)
        fseek(infp, lineoffset-(size_t)nrg*sizeof(fcomplex), SEEK_CUR);
      readdata((fcomplex *)in[i], (size_t)nrg * sizeof(fcomplex), infp);
    }

    if(byteorder!=0){
      //printf("swapping bytes...\n");
      for(i = 0; i < naz_burst_in; i++)
        for(j = 0; j < nrg; j++){
          SWAP4(in[i][j].re);
          SWAP4(in[i][j].im);
        }
    }

    //create deramping signal: make center of azimuth spectrum ZERO
    for(i = 0; i < nrg; i++){
      for(j = 0; j < naz_burst_in; j++){
        //distance from raw burst center in number of lines
        tmp1 = j + min_line_in - bcl;
        tmp2 = - PI * ka[i] * (tmp1 * pri) * (tmp1 * pri);
        deramp[j][i].re = cos(tmp2);
        deramp[j][i].im = sin(tmp2);
      }
    }

    //do the filtering column by column
    for(i = 0; i < nrg; i++){
      //prepare data
      for(j = 0; j < nfft; j++){
        if(j < naz_burst_in){
          data[j].re = in[j][i].re / sc;
          data[j].im = in[j][i].im / sc;
        }
        else{
          data[j].re = 0.0;
          data[j].im = 0.0;
        }
      }
      
      //deramp the data
      for(j = 0; j < naz_burst_in; j++){
        data[j] = cmul(data[j], deramp[j][i]);
      }

      //forward fft
      //four1((float *)data - 1, nfft, -1);
      fftwf_execute(p_forward);

      //multiplication in the frequency domain
      for(j = 0; j < nfft; j++)
        data[j] = cmul(data[j], filter[i][j]);

      //backward fft
      //four1((float *)data - 1, nfft, 1);
      fftwf_execute(p_backward);

      //put filtered data back
      for(j = 0; j < naz_burst_in; j++){
        in[j][i].re = data[j].re * sc / nfft;
        in[j][i].im = data[j].im * sc / nfft;
      }
    }

    //zero output
    for(i = 0; i < naz_burst_out; i++){
      for(j = 0; j < nrg; j++){
        out[i][j].re = 0.0;
        out[i][j].im = 0.0;
      }
    }
    
    //do the resampling column by column
    for(i = 0; i < nrg; i++){
      //resampling to output grid
      for(j = 0; j < naz_burst_out; j++){
        if((j < output_start[i]) || (j > output_end[i]))
          continue;
        
        //location of output line in the data block read in
        tmp1 = min_line_out + j * az_ratio - min_line_in;

        //interpolation
        for(k = -hn; k <= hn; k++){
          index = roundfi(tmp1) + k;
          tmp2 = index - tmp1;

          if( (index < 0) || (index > naz_burst_in - 1) )
            continue;
          //verified: roundfi(tmp2*m) won't be out of [-hnm, hnm], if no computation error of floating point
          out[j][i].re += in[index][i].re * kernel[roundfi(tmp2*m)];
          out[j][i].im += in[index][i].im * kernel[roundfi(tmp2*m)];
        }
        
        //reramp
        tmp1 = j * az_ratio + min_line_out - bcl;
        tmp2 = PI * ka[i] * (tmp1 * pri) * (tmp1 * pri);
        reramp.re = cos(tmp2);
        reramp.im = sin(tmp2);
        
        out[j][i] = cmul(out[j][i], reramp);

      }
    }

    //write to file
    strcpy(output_filename, outputf);
    sprintf(burst_num, "_%02d.slc", burst_index - burst_index_first + 1);
    strcat(output_filename, burst_num);
    
    outfp = openfile(output_filename, "wb");
    writedata((fcomplex *)out[0], (size_t)naz_burst_out * (size_t)nrg * sizeof(fcomplex), outfp);
    fclose(outfp);
  }

  fprintf(infofp, "total number of bursts extracted: %3d\n\n", burst_index - burst_index_first);

  fftwf_destroy_plan(p_forward);
  fftwf_destroy_plan(p_backward);

  free_array1d_float(ka);
  free_array1d_float(dop);
  free_array1d_float(nfa);

  free_array1d_float(start_line);
  free_array1d_float(end_line);
  free_array1d_int(output_start);
  free_array1d_int(output_end);

  free_vector_float(sincc, -hnm, hnm);
  free_vector_float(kaiserc, -hnm, hnm);
  free_vector_float(kernel, -hnm, hnm);

  free_array2d_fcomplex(in);
  free_array2d_fcomplex(out);
  free_array2d_fcomplex(deramp);
  free_array2d_fcomplex(filter);
  free_array1d_fcomplex(data);

  //close files
  fclose(infp);
  fclose(infofp);
}
