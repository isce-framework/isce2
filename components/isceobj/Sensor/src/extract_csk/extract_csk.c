/*CSK raw data extractor.
 * Original Author: Walter Szeliga
 * Optimized version: Piyush Agram
 * Changes for optimization:
 *  - No more hyperslab selection. 
 *  - Direct write of dataset to memory mapped raw file.
 *  - OpenMP loop on the memory mapped file to adjust the values. */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <sys/types.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <omp.h>
#include "hdf5.h"



double getExtrema(double *lut_data);
double* extractLUT(hid_t file);
int extractImage(hid_t file,char *outFile,double *lut_data);

int
extract_csk(char *filename, char *outFile)
{
  double *lut_data;
  hid_t file;
  herr_t status;
  
  /* Open the file and get the dataset */ 
  file = H5Fopen(filename,H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
    {
      fprintf(stderr,"Unable to open file: %s\n",filename);
      return EXIT_FAILURE;
    }
  
  lut_data = extractLUT(file);
  extractImage(file,outFile,lut_data);
  
  status = H5Fclose(file);

  free(lut_data);
  return 0;
}

double *
extractLUT(hid_t file)
{
  int i, ndims;
  double val;
  double *double_lut;
  hid_t lut_attr,lut_space;
  hsize_t dims[1];
  herr_t status;
  
  lut_attr = H5Aopen_name(file,"Analog Signal Reconstruction Levels");
  lut_space = H5Aget_space(lut_attr);
  ndims = H5Sget_simple_extent_dims(lut_space,dims,NULL);
  double_lut = (double *)malloc(dims[0]*sizeof(double));
  status = H5Aread(lut_attr, H5T_NATIVE_DOUBLE, double_lut);

  for(i=0;i<dims[0];i++)
    {
      if(isnan(double_lut[i]))
	{
	  double_lut[i] = 0.0;
        }
    }

  H5Aclose(lut_attr);
  H5Sclose(lut_space);
  
  return double_lut;
}

double 
getExtrema(double *double_lut)
{
  int i;
  double min,max;
  
  max = DBL_MIN;
  min = DBL_MAX;
  for(i=0;i<256;i++)
    {
      if (double_lut[i] > max)
	{
	  max = double_lut[i];
	}
      if (double_lut[i] < min)
	{
	  min = double_lut[i];
	}
    }

  printf("Max: %lf\n",max);
  return max;
}

int
extractImage(hid_t file, char* outFile, double *lut_data)
{
  unsigned char *IQ,*data;
  int i,j,k;
  hid_t type,native_type;
  hid_t dataset,dataspace, cparms;
  hsize_t dims[3],chunk[3];
  hsize_t count_out;
  herr_t status;
  int out;
  long index;
  unsigned char I;

  double max = getExtrema(lut_data);
  max = hypot(max,max);

  #if H5Dopen_vers == 2
    dataset = H5Dopen2(file,"/S01/B001",H5P_DEFAULT);
  #else
    dataset = H5Dopen(file,"/S01/B001");
  #endif
  type = H5Dget_type(dataset);
  native_type = H5Tget_native_type(type,H5T_DIR_ASCEND);

  dataspace = H5Dget_space(dataset);
  status = H5Sget_simple_extent_dims(dataspace, dims, NULL);
  
  printf("Dimensions %lu x %lu x %lu\n",(unsigned long)dims[0],(unsigned long)dims[1],(unsigned long)dims[2]);


  /* Memory map output file */
  out = open(outFile, O_RDWR | O_CREAT, (mode_t)0600);
  if(ftruncate(out,(dims[0]*dims[1]*dims[2]*sizeof(unsigned char))) == -1 )
  {
      fprintf(stderr,"Unable to create file %s\n",outFile);
      close(out);
      return 1;
  }
  data = (char *)mmap(0,dims[0]*dims[1]*dims[2]*sizeof(unsigned char), PROT_READ | PROT_WRITE, MAP_SHARED, out, 0);

  /* Check if the dataset is chunked */
  cparms = H5Dget_create_plist(dataset);
  
  if (H5D_CHUNKED == H5Pget_layout(cparms))
  {
      status = H5Pget_chunk(cparms,3,chunk);
      printf("The dataset is chunked. \n");
      printf("Chunk size: %lu x  %lu x %lu \n", (unsigned long) chunk[0], (unsigned long) chunk[1], (unsigned long) chunk[2]);
  }

  IQ = (unsigned char*)malloc(2*dims[1]*sizeof(unsigned char));
      
  //Lets do the whole thing in one go
  //Super fast but we need type conversion
  status = H5Dread(dataset, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  count_out = dims[1]*2;

  for(k=0; k<dims[0]; k++)
  {
        index = 2*dims[1]*k;
        memcpy(IQ,data+index, count_out);

        #pragma omp parallel for\
        shared(IQ, lut_data, max, count_out) \
        private(i)
        for(i=0; i<count_out; i++)
            IQ[i] = (unsigned char) (127.0*lut_data[IQ[i]]/max + 127.0);
         
        memcpy(data+index, IQ, count_out); 
  }

  munmap(data,(2*dims[0]*dims[1]*sizeof(unsigned char)));
  close(out);
  free(IQ);

  /*Cleanup*/
  status = H5Pclose(cparms);
  status = H5Sclose(dataspace);
  status = H5Dclose(dataset);

  return 0;
}

