
c****************************************************************

	subroutine tranmat(r_a,r_b)

c****************************************************************
c**
c**	FILE NAME: matmat.for
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: The subroutine takes a 3x3 matrix
c**     and computes its transpose.
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
 	real*8 r_a(3,3)                      !3x3 matrix
   
c   	OUTPUT VARIABLES:
        real*8 r_b(3,3)                      !3x3 matrix

c	LOCAL VARIABLES:
	integer i,j         

c  	PROCESSING STEPS:

c       compute matrix product

        do i=1,3
           do j=1,3
             r_b(i,j) = r_a(j,i)
           enddo 
        enddo
          
        end  
