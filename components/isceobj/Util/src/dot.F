c****************************************************************

	real*8 function dot(r_v,r_w)

c****************************************************************
c**
c**	FILE NAME: dot.f
c**
c**     DATE WRITTEN:7/15/90 
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: This routine computes the dot product of
c**     two 3 vectors as a function.
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       	implicit none

c	INPUT VARIABLES:
        real*8 r_v(3),r_w(3)          !3x1 vectors 
   
c   	OUTPUT VARIABLES: dot is the output

c	LOCAL VARIABLES:none

c	DATA STATEMENTS:none

C	FUNCTION STATEMENTS:none

c  	PROCESSING STEPS:

c   compute dot product of two 3-vectors

	dot = r_v(1)*r_w(1) + r_v(2)*r_w(2) + r_v(3)*r_w(3) 

	end
