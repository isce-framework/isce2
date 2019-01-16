/* SccsId[ ]= @(#)io.c	1.1 2/5/92 */
#include <stdio.h> 
#include <fcntl.h>

/* modified to add iolen function EJF 96/8/29 */

#include <sys/types.h>
#include <unistd.h>

#define PERMS 0666
/* IO library:
 * done by quyen dinh nguyen
 * 11/12/91:
 */

/* To open a file and assign a channel to it. This must be
   done before any attempt is made to access the file. The 
   return value (initdk) is the file descriptor. The file can
   be closed with the closedk subroutine.
   
   Remember, always open files before you need to access to them
   and close them after you don't need them any more. In UNIX,
   there is a limit (20) of number files can be opened at once.

   Note that, if the file is not existing, the routine will create
   the new file  with PERM=0666.

   Calling sequence(from FORTRAN):
         fd = initdk(lun,filename)
   where:
         fd is the long int for file descriptor.

         lun is the dummy variable to be compatible with VMS calls.

         filename is the name of the file. Include directory paths 
         if necessary.
 */

#ifndef UL
int initdk(lun, filename)
#else
int initdk_(lun, filename)
#endif
int *lun; char *filename;
{  int i;
   int fd;
   for(i=0; i < strlen(filename); i++)
     if( *(filename+i) == ' ') *(filename+i) = '\0' ;
   if((fd=open(filename,O_RDWR)) < 0){
       if( (fd = open(filename,O_RDONLY)) > 0)
           printf(" Open filename %s as READ ONLY\n",filename);
   }
   if( fd < 0 ) fd = open(filename,O_CREAT|O_RDWR,0666);
   if(fd == -1)printf(" Cannot open the filename: %s\n",filename);
   return(fd);
}

/* To write data into a previous opened file. This routine
   will wait until the write operations are completed.
  
   Calling sequence (from FORTRAN):
         nbytes = iowrit( chan, buff, bytes)
	 call iowrit(chan,buff,bytes)
   where:
         nbytes is the number bytes that transfered.
   
         chan is the file descriptor.

         buff is the buffer or array containing the data you
         wish to write.

         bytes is the number of bytes you wish to write.
*/ 

#ifndef UL
int iowrit(chan, buff, bytes)
#else
int iowrit_(chan, buff, bytes)
#endif
int *chan, *bytes;
char *buff;
{  
   int nbytes;
   nbytes = write(*chan, buff, *bytes);
   if(nbytes != *bytes) fprintf(stderr,
       " ** ERROR **: only %d bytes transfered out of %d bytes\n",
       nbytes, *bytes);
   return(nbytes);
}

/* To read data from a previously opened file. This routine will
   wait until after its operations are completed.

   Calling sequence (from FORTRAN):
       nbytes = ioread( chan, buff, bytes)
       call ioread( chan, buff, bytes)
   where:
       nbytes is the number bytes that transfered.
  
       chan is the file descriptor.
 
       buff is the buffer or array containning the data you wish
       to read.

       bytes is the number of bytes you wish to read.

 */

#ifndef UL
int ioread(chan, buff, bytes)
#else
int ioread_(chan, buff, bytes)
#endif

int *chan, *bytes ;
char *buff;
{  
   int nbytes;
   nbytes = read(*chan, buff, *bytes);
   if(nbytes != *bytes) fprintf(stderr,
     " ** ERROR **: only %d bytes are read out of %d requested\n",
     nbytes, *bytes);
   return(nbytes);
}


/* To position the file pointer. This routine will call the lseek 
   to update the file pointer.

   Calling sequence (from FORTRAN):
      file_loc = ioseek(chan,loc_byte)
      call ioseek(chan,loc_byte)
   where:
        file_loc is the returned file location.

        chan is the file descriptor.

        loc_byte is byte location that requested to be set. This value
        must be greater or equal to zero for positioning the file at
        that location. If loc_byte is negative, the file pointer will
        move abs(loc_byte) from the current location.
        
*/

#ifdef C32_IO
#ifndef UL
int ioseek(chan, loc_byte)
#else
int ioseek_(chan, loc_byte)
#endif

int *chan, *loc_byte;
{  
   int ibytes,nloc;
   ibytes = *loc_byte ;
   if(ibytes >= 0) nloc = lseek(*chan, ibytes, 0);
   else {
      ibytes = - ibytes;
      nloc = lseek(*chan, ibytes, 1);
   }
   return(nloc);
}
#endif

#ifdef C64_IO
#ifndef UL
off64_t ioseek(chan, loc_byte)
#else
off64_t ioseek_(chan, loc_byte)
#endif

int *chan;
off64_t *loc_byte;
{  
   off64_t ibytes,nloc;
   ibytes = *loc_byte ;
   if(ibytes >= 0) nloc = lseek64(*chan, ibytes, 0);
   else {
      ibytes = - ibytes;
      nloc = lseek64(*chan, ibytes, 1);
   }
   return(nloc);
}
#endif

/* To close the file previously opened by initdk.

   Calling sequence (from FORTRAN):
      istatus = closedk( lun, chan)
      call closedk( lun, chan)
   where:
      istatus is the return value (0 is success, -1 is error)
 
      lun is the dummy variable to be compatible the VAX VMS call.

      chan is the file descriptor that you want to close.
 */

#ifndef UL
int closedk(lun,chan)
#else
int closedk_(lun,chan)
#endif

int *lun, *chan;
{
   return(close(*chan));
}



/* To determine the file length. This routine will call lseek 
   to find the end of the file, and return the length in bytes.
   The file pointer is then set back to the beginning.

   written 96/8/29 EJF

   Calling sequence (from FORTRAN):
      length = iolen(chan)

   where:
        length is the returned file length (bytes).

        chan is the file descriptor.
*/

#ifndef C64_IO

#ifndef UL
int iolen(chan)
#else
int iolen_(chan)
#endif
int *chan;
{  
   off_t nloc, junk;
   nloc = lseek(*chan, (off_t)0, SEEK_END); /* go to end, get length */
   printf("length 32bits=%d\n",(int)nloc);
   junk = lseek(*chan, (off_t)0, SEEK_SET); /* rewind back to beginning */
   return((int)nloc);
}

#else

#ifndef UL
int iolen(chan)
#else
int iolen_(chan)
#endif
int *chan;
{  
   off64_t nloc, junk;
   nloc = lseek64(*chan, (off64_t)0, SEEK_END); /* go to end, get length */
   printf("length 64bits=%d\n",(int)nloc);
   junk = lseek64(*chan, (off64_t)0, SEEK_SET); /* rewind back to beginning */
   return((int)nloc);
}

#endif
