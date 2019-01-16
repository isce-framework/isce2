c****************************************************************

	subroutine unitvec(r_v,r_u)

c****************************************************************
c**
c**	FILE NAME: unitvec.f
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: The subroutine takes vector and returns 
c**     a unit vector.
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
   
c   	OUTPUT VARIABLES:
        real*8 r_u(3)                              !3x1 vector

c	LOCAL VARIABLES:
	real*8 r_n

c  	PROCESSING STEPS:

c       compute vector norm

        r_n = sqrt(r_v(1)**2 + r_v(2)**2 + r_v(3)**2)
      
        if(r_n .ne. 0)then  
          r_u(1) = r_v(1)/r_n
          r_u(2) = r_v(2)/r_n
          r_u(3) = r_v(3)/r_n
        endif
 
        end  

