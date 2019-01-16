/* SccsId[ ]= @(#)io.c	1.1 2/5/92 */
#include <stdio.h> 
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

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

// Function declaration
int initdk_(lun, file)
int *lun; char *file;
{  int i;
   int fd;
   char filename[100];

   i=0;
   while(file[i]!=' ' && i<strlen(file) && i<99){
     filename[i]=file[i];
     i++;
     filename[i]='\0';
   }
   
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
int iowrit_(chan, buff, bytes)
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
int ioread_(chan, buff, bytes)
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

int ioseek_(chan, loc_byte)
int *chan, *loc_byte;

{  
   int nloc;
   off_t ibytes;
   ibytes = (off_t) *loc_byte ;

   if(ibytes >= 0) nloc = lseek(*chan, ibytes, SEEK_SET);
   else {
      ibytes = - ibytes;
      nloc = lseek(*chan, ibytes, SEEK_CUR);
   }
   /*   printf("nloc= %d\n",nloc);  */
   return(nloc);
}



/* To close the file previously opened by initdk.

   Calling sequence (from FORTRAN):
      istatus = closedk( lun, chan)
      call closedk( lun, chan)
   where:
      istatus is the return value (0 is success, -1 is error)
 
      lun is the dummy variable to be compatible the VAX VMS call.

      chan is the file descriptor that you want to close.
 */

int closedk_(lun,chan)
int *lun, *chan;
{
   return(close(*chan));
}



