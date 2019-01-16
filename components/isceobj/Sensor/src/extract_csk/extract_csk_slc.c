/*CSK slc data extractor.
 * Author: Piyush Agram
*/

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

int extractImageSlc(hid_t file,char *outFile);

int
extract_csk_slc(char *filename, char *outFile)
{
  hid_t file;
  herr_t status;
  
  /* Open the file and get the dataset */ 
  file = H5Fopen(filename,H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
    {
      fprintf(stderr,"Unable to open file: %s\n",filename);
      return EXIT_FAILURE;
    }
  
  extractImageSlc(file,outFile);
  
  status = H5Fclose(file);

  return 0;
}

int
extractImageSlc(hid_t file, char* outFile)
{
  char *data;
  int i,j,k;
  hid_t type,native_type;
  hid_t dataset, cparms;
  hid_t dataspace;
  hsize_t dims[3],chunk[3];
  hsize_t count_out;
  herr_t status;
  int out;
  long index;
  unsigned char I;

  #if H5Dopen_vers == 2
    dataset = H5Dopen2(file,"/S01/SBI",H5P_DEFAULT);
  #else
    dataset = H5Dopen(file,"/S01/SBI");
  #endif
  type = H5Dget_type(dataset);
  native_type = H5Tget_native_type(type,H5T_DIR_ASCEND);

  dataspace = H5Dget_space(dataset);
  status = H5Sget_simple_extent_dims(dataspace, dims, NULL);
  
  printf("Dimensions %lu x %lu x %lu\n",(unsigned long)dims[0],(unsigned long)dims[1],(unsigned long)dims[2]);


  /* Memory map output file */
  out = open(outFile, O_RDWR | O_CREAT, (mode_t)0600);
  if(ftruncate(out,(dims[0]*dims[1]*dims[2]*sizeof(float))) == -1 )
  {
      fprintf(stderr,"Unable to create file %s\n",outFile);
      close(out);
      return 1;
  }
  data = (char *)mmap(0,dims[0]*dims[1]*dims[2]*sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED, out, 0);

  /* Check if the dataset is chunked */
  cparms = H5Dget_create_plist(dataset);
  
  if (H5D_CHUNKED == H5Pget_layout(cparms))
  {
      status = H5Pget_chunk(cparms,3,chunk);
      printf("The dataset is chunked. \n");
      printf("Chunk size: %lu x  %lu x %lu \n", (unsigned long) chunk[0], (unsigned long) chunk[1], (unsigned long) chunk[2]);
  }

      
  //Lets do the whole thing in one go
  //Super fast but we need type conversion
  status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  munmap(data,(dims[0]*dims[1]*dims[2]*sizeof(float)));
  close(out);

  /*Cleanup*/
  status = H5Pclose(cparms);
  status = H5Sclose(dataspace);
  status = H5Dclose(dataset);

  return 0;
}

