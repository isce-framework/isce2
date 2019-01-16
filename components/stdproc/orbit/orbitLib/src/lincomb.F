!c****************************************************************

	subroutine lincomb(r_k1,r_u,r_k2,r_v,r_w)

!c****************************************************************
!c**
!c**	FILE NAME: lincomb.for
!c**
!c**     DATE WRITTEN: 8/3/90
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c** 	FUNCTIONAL DESCRIPTION: The subroutine forms the linear combination
!c**	of two vectors.
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************

       	implicit none

!c	INPUT VARIABLES:
        real*8 r_u(3)                              !3x1 vector
        real*8 r_v(3)                              !3x1 vector
        real*8 r_k1				 !scalar
        real*8 r_k2				 !scalar
   
!c   	OUTPUT VARIABLES:
        real*8 r_w(3)                              !3x1 vector

!c	LOCAL VARIABLES:none

!c  	PROCESSING STEPS:

!c       compute linear combination

	r_w(1) = r_k1*r_u(1) + r_k2*r_v(1)
	r_w(2) = r_k1*r_u(2) + r_k2*r_v(2)
	r_w(3) = r_k1*r_u(3) + r_k2*r_v(3)
      
        end  


