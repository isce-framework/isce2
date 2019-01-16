module linalg
  !!********************************************************
  !*
  !*    DESCRIPTION: collection of matrix/vector linear algebra functions
  !*
  !*    FUNCTION LIST: dot, matvec, lincomb, unitvec
  !*
  !!*********************************************************
  implicit none
contains
  
  real*8 function dot(r_v,r_w)
    !c****************************************************************
    !c**
    !c** FILE NAME: dot.f
    !c**
    !c** DATE WRITTEN:7/15/90
    !c**
    !c** PROGRAMMER:Scott Hensley
    !c**
    !c** FUNCTIONAL DESCRIPTION: This routine computes the dot product of
    !c** two 3 vectors as a function.
    !c**
    !c** ROUTINES CALLED:none
    !c**
    !c** NOTES: none
    !c**
    !c** UPDATE LOG:
    !c**
    !c*****************************************************************
    
    !c INPUT VARIABLES:
    real*8, intent(in) :: r_v(3),r_w(3) !3x1 vectors
    
    !c compute dot product of two 3-vectors
    dot = r_v(1)*r_w(1) + r_v(2)*r_w(2) + r_v(3)*r_w(3)
  end function dot
  
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
    
    
    !c	INPUT VARIABLES:
    real*8, intent(in) :: r_t(3,3)                            !3x3 matrix
    real*8, intent(in) :: r_v(3)                              !3x1 vector
    
    !c   	OUTPUT VARIABLES:
    real*8, intent(out) :: r_w(3)                              !3x1 vector
    
    
    !c  	PROCESSING STEPS:
    
    !c       compute matrix product
    r_w(1) = r_t(1,1)*r_v(1) + r_t(1,2)*r_v(2) + r_t(1,3)*r_v(3)
    r_w(2) = r_t(2,1)*r_v(1) + r_t(2,2)*r_v(2) + r_t(2,3)*r_v(3)
    r_w(3) = r_t(3,1)*r_v(1) + r_t(3,2)*r_v(2) + r_t(3,3)*r_v(3)
    
  end subroutine matvec
  
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
    
    
    !c	INPUT VARIABLES:
    real*8, intent(in), dimension(3) :: r_u                              !3x1 vector
    real*8, intent(in), dimension(3) ::  r_v                              !3x1 vector
    real*8, intent(in) ::  r_k1				 !scalar
    real*8, intent(in) ::  r_k2				 !scalar
    
    !c   	OUTPUT VARIABLES:
    real*8, intent(out) ::  r_w(3)                              !3x1 vector
    
    !c  	PROCESSING STEPS:
    
    !c       compute linear combination
    
    r_w(1) = r_k1*r_u(1) + r_k2*r_v(1)
    r_w(2) = r_k1*r_u(2) + r_k2*r_v(2)
    r_w(3) = r_k1*r_u(3) + r_k2*r_v(3)
    
  end subroutine lincomb
  
  !c*****************************************************************
  
  subroutine unitvec(r_v,r_u)
    
    !c****************************************************************
    !c**
    !c**	FILE NAME: unitvec.for
    !c**
    !c**     DATE WRITTEN: 8/3/90
    !c**
    !c**     PROGRAMMER:Scott Hensley
    !c**
    !c** 	FUNCTIONAL DESCRIPTION: The subroutine takes vector and returns 
    !c**     a unit vector.
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
    real*8, intent(in), dimension(3) ::  r_v                              !3x1 vector
    
    !c   	OUTPUT VARIABLES:
    real*8, intent(out), dimension(3) ::  r_u                              !3x1 vector
    
    !c	LOCAL VARIABLES:
    real*8 r_n
    
    !c  	PROCESSING STEPS:
    
    !c       compute vector norm
    
    r_n = sqrt(r_v(1)**2 + r_v(2)**2 + r_v(3)**2)
    
    if(r_n .ne. 0)then  
       r_u(1) = r_v(1)/r_n
       r_u(2) = r_v(2)/r_n
       r_u(3) = r_v(3)/r_n
    endif
    
  end  subroutine unitvec

!c****************************************************************

        function norm(r_v)

!c****************************************************************
!c**
!c**    FILE NAME: norm.for
!c**
!c**     DATE WRITTEN: 8/3/90
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: The subroutine takes vector and returns
!c**     its norm.
!c**
!c**     ROUTINES CALLED:none
!c**
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************

 

!c      INPUT VARIABLES:
        real*8 r_v(3)                              !3x1 vector

!c      OUTPUT VARIABLES:see input

!c      LOCAL VARIABLES:
        real*8 norm

!c      PROCESSING STEPS:

!c       compute vector norm

        norm = sqrt(r_v(1)**2 + r_v(2)**2 + r_v(3)**2)

        end function norm

  
  
end module linalg
