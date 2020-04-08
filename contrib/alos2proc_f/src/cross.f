
c****************************************************************

	subroutine cross(r_u,r_v,r_w)

c****************************************************************
c**
c**	FILE NAME: cross.f
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: The subroutine takes two vectors and returns 
c**     their cross product.
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
        real*8 r_w(3)

c	LOCAL VARIABLES:
	
c  	PROCESSING STEPS:

c       compute vector norm

        r_w(1) = r_u(2)*r_v(3) - r_u(3)*r_v(2)  
        r_w(2) = r_u(3)*r_v(1) - r_u(1)*r_v(3)  
        r_w(3) = r_u(1)*r_v(2) - r_u(2)*r_v(1)  

        end  
