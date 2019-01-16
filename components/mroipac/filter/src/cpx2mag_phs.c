#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>  /* off_t is hidden here under Mac OS 10.4 */

#define MaxWidth  80000
 
void cpx2mag_phs(char *InFile, char *OutFile1, char *OutFile2, int Width)
{
   FILE   *InFP,              *OutFP1,              *OutFP2;
   float  InLine[MaxWidth*2], OutLine1[MaxWidth],   OutLine2[MaxWidth];
   double Pi;
   long   Length;
   off_t  LenBytes;
   int    i,j;
   
   Pi=4*atan2(1,1);

/********************************************************************************/
/****************************                           *************************/
/********************************************************************************/
   if((InFP=fopen(InFile,"r"))==NULL){
      fprintf(stderr,"file %s not open\n",InFile);
      exit(0);
      }
   if((OutFP1=fopen(OutFile1,"w"))==NULL){
      fprintf(stderr,"file %s not open\n",OutFile1);
      exit(0);
      }
   if((OutFP2=fopen(OutFile2,"w"))==NULL){
      fprintf(stderr,"file %s not open\n",OutFile2);
      exit(0);
      }
      
   fseeko(InFP,0L,SEEK_END);  /* need to use fseeko and ftello to read large files EJF 07/2/5 */
   LenBytes = ftello(InFP);
   Length=(long)(LenBytes/(2*sizeof(float)*Width));
   printf("length %ld lines\n",Length);
   rewind(InFP);
   
   for(i=0;i<Length;i++){
      if(i%100==0)fprintf(stderr,"\rline %d",i);
      fread(InLine,sizeof(InLine[0]),Width*2,InFP);
      for(j=0;j<Width;j++){
         OutLine1[j]  =sqrt(InLine[2*j]*InLine[2*j]+InLine[2*j+1]*InLine[2*j+1]);
         if(OutLine1[j] != 0.){
            OutLine2[j] =atan2(InLine[2*j+1],InLine[2*j]);
	    }else{
            OutLine2[j]=0.;
	  }
         }
      fwrite(OutLine1,sizeof(OutLine1[0]),Width,OutFP1);
      fwrite(OutLine2,sizeof(OutLine2[0]),Width,OutFP2);
      }
   fprintf(stderr,"\n");
   close(InFP);
   close(OutFP1);
   close(OutFP2);
   }
//POD=pod
//POD
//POD=head1 USAGE
//POD
//POD  Usage:cpx2mag_phs Infile Outfile1 Outfile2 Width
//POD  Infile: complex image
//POD  Outfile1: amplitude image
//POD  Outfile2: phase image
//POD  Width: number of complex pixels
//POD
//POD=head1 FUNCTION
//POD
//POD FUNCTIONAL DESCRIPTION:  "cpx2mag_phs" collects amplitude and phase (r*4/float) from
//POD an input c*8 file (Infile, c*8/complex) an writes them in two sepate files: Infile1 and Infile2.
//POD Each of these files contains either the amplitude (Infile1) or the phase (Infile2) of the input data.
//POD In all these files the record length is "width".
//POD  
//POD=head1 ROUTINES CALLED
//POD
//POD none
//POD
//POD=head1 CALLED BY
//POD
//POD
//POD=head1 FILES USED
//POD
//POD Input data from a binary file whose records consist of "Width" c*8/complex samples.
//POD
//POD=head1 FILES CREATED
//POD
//POD The two output files (r*4/float) "Infile1" & "Infile2" contains the amplitudes and phases (radians) 
//POD of the input c*8/complex file.
//POD
//POD=head1 DIAGNOSTIC FILES
//POD
//POD
//POD=head1 HISTORY
//POD
//POD Routines written by Francois Rogez
//POD
//POD=head1 LAST UPDATE
//POD  Date Changed        Reason Changed 
//POD  ------------       ----------------
//POD
//POD POD comments trm Jan 29th '04
//POD changed to use 64-bit file operations to work with large files EJF 07/2/5
//POD
//POD=cut
