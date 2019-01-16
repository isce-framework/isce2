#include <stdio.h>
#include <math.h>
#include <string.h>

#define MaxWidth  80000
#define MaxLength 80000
 
void mag_phs2cpx(char *InFile1, char *InFile2, char *OutFile, int Width)
  {

   FILE   *InFP1,              *InFP2,              *OutFP;
   float  InLine1[MaxWidth],   InLine2[MaxWidth],   OutLine[2*MaxWidth];
   double Pi;
   int    Length,Length1,Length2;
   int    i,j;

   Pi=4*atan2(1,1);

/********************************************************************************/
/****************************                           *************************/
/********************************************************************************/
   if((InFP1=fopen(InFile1,"r"))==NULL){
      fprintf(stderr,"file %s not open\n",InFile1);
      exit(0);
      }
   if((InFP2=fopen(InFile2,"r"))==NULL){
      fprintf(stderr,"file %s not open\n",InFile2);
      exit(0);
      }
   if((OutFP=fopen(OutFile,"w"))==NULL){
      fprintf(stderr,"file %s not open\n",OutFile);
      exit(0);
      }
      
   fseeko(InFP1,0L,SEEK_END);
   Length1=ftello(InFP1)/(sizeof(InLine1[0])*Width);
   rewind(InFP1);
   fseeko(InFP2,0L,SEEK_END);
   Length2=ftello(InFP2)/(sizeof(InLine2[0])*Width);
   rewind(InFP2);
   Length=Length1;
   if(Length>Length2)Length=Length2;
   
   for(i=0;i<Length;i++){
      if(i%100==0)fprintf(stderr,"\rline %d",i);
      fread(InLine1,sizeof(InLine1[0]),Width,InFP1);
      fread(InLine2,sizeof(InLine2[0]),Width,InFP2);
      for(j=0;j<Width;j++){
         OutLine[2*j]        = InLine1[j]*cos(InLine2[j]);
         OutLine[2*j+1]      = InLine1[j]*sin(InLine2[j]);
         }
      fwrite(OutLine,sizeof(OutLine[0]),Width*2,OutFP);
      }
   fprintf(stderr,"\n");
   close(InFP1);
   close(InFP2);
   close(OutFP);
   }
//POD=pod
//POD
//POD=head1 USAGE
//POD
//POD Usage: mag_phs2cpx: Infile1 Infile2 Outfile width    
//POD Infile1: amplitude image                             
//POD Infile2: phase image                                 
//POD Outfile: rmg image;                                  
//POD width: number of pixels                      
//POD
//POD=head1 FUNCTION
//POD
//POD FUNCTIONAL DESCRIPTION:  "mag_phs2cpx" collects amplitude and phase (r*4/float) values from two sepate files,
//POD Infile1 and Infile2 respectively, and writes them to an output c*8 file (Outfile, c*8/complex)
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
//POD Each of the input files "Infile1" & "Infile2" contains one of the two r*4/float arrays to store in the c*8/complex file.
//POD
//POD=head1 FILES CREATED
//POD
//POD Amplitude and phase stored as c*8/complex samples.
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
//POD=cut
