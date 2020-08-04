c****************************************************************

	subroutine norm(r_v,r_n)

c****************************************************************
c**
c**	FILE NAME: norm.f
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: The subroutine takes vector and returns 
c**     its norm.
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
        real*8 r_v(3)                              !3x1 vector
   
c   	OUTPUT VARIABLES:see input

c	LOCAL VARIABLES:
	real*8 r_n

c  	PROCESSING STEPS:

c       compute vector norm

        r_n = sqrt(r_v(1)**2 + r_v(2)**2 + r_v(3)**2)
      
        end  
