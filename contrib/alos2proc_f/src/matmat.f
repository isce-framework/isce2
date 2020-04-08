c****************************************************************

	subroutine matmat(r_a,r_b,r_c)

c****************************************************************
c**
c**	FILE NAME: matmat.for
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: The subroutine takes two 3x3 matrices
c**     and multiplies them to return another 3x3 matrix.
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
 	real*8 r_a(3,3),r_b(3,3)              !3x3 matrix
   
c   	OUTPUT VARIABLES:
        real*8 r_c(3,3)                       !3x3 matrix

c	LOCAL VARIABLES:
	integer i         

c  	PROCESSING STEPS:

c       compute matrix product

        do i=1,3
       	  r_c(i,1) = r_a(i,1)*r_b(1,1) + r_a(i,2)*r_b(2,1) + 
     +               r_a(i,3)*r_b(3,1)
	  r_c(i,2) = r_a(i,1)*r_b(1,2) + r_a(i,2)*r_b(2,2) + 
     +               r_a(i,3)*r_b(3,2)
	  r_c(i,3) = r_a(i,1)*r_b(1,3) + r_a(i,2)*r_b(2,3) + 
     +               r_a(i,3)*r_b(3,3)
        enddo 
          
        end 
