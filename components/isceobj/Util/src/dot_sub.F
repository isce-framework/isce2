c****************************************************************

	subroutine dot(r_u,r_v,r_d)

c****************************************************************
c**
c**	FILE NAME: dot.f
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: The subroutine takes two vectors and returns 
c**     their dot product.
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
        real*8 r_v(3),r_u(3)            !3x1 vectors
   
c   	OUTPUT VARIABLES:
        real*8 r_d

c	LOCAL VARIABLES:
	
c  	PROCESSING STEPS:

c       compute vector dot product

        r_d = r_v(1)*r_u(1) + r_v(2)*r_u(2) + r_v(3)*r_u(3)

        end 
