!c****************************************************************

	subroutine matvec(r_t,r_v,r_w)

!c****************************************************************
!c**
!c**	FILE NAME: matvec.for
!c**
!c**     DATE WRITTEN: 7/20/90
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c** 	FUNCTIONAL DESCRIPTION: The subroutine takes a 3x3 matrix 
!c**     and a 3x1 vector a multiplies them to return another 3x1
!c**	vector.
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c****************************************************************

       	implicit none

!c	INPUT VARIABLES:
 	real*8 r_t(3,3)                            !3x3 matrix
        real*8 r_v(3)                              !3x1 vector
   
!c   	OUTPUT VARIABLES:
        real*8 r_w(3)                              !3x1 vector

!c	LOCAL VARIABLES:none

!c  	PROCESSING STEPS:

!c       compute matrix product

	r_w(1) = r_t(1,1)*r_v(1) + r_t(1,2)*r_v(2) + r_t(1,3)*r_v(3)
	r_w(2) = r_t(2,1)*r_v(1) + r_t(2,2)*r_v(2) + r_t(2,3)*r_v(3)
	r_w(3) = r_t(3,1)*r_v(1) + r_t(3,2)*r_v(2) + r_t(3,3)*r_v(3)
          
        end  
