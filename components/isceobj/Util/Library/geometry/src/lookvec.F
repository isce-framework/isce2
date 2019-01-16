 
!c****************************************************************

        subroutine lookvec(pos,r_look,r_az,r_v)BIND(C,NAME='lookvec_C')

!c****************************************************************
!c**
!c**    FILE NAME: lookvec.f
!c**
!c**     DATE WRITTEN: 1/25/92
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: This subroutine will compute the look
!c**    vector given the look angle,azimuth angle, and the position
!c**    vector. 
!c**
!c**     ROUTINES CALLED:cross,unitvec,lincomb
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************

        use, intrinsic :: iso_c_binding
        use linalg3module
        implicit none

!c      INPUT VARIABLES:

        type (positionType) pos

        real(C_DOUBLE), value :: r_look                 !r_look angle
        real(C_DOUBLE), value :: r_az                   !azimuth angle
   
!c       OUTPUT VARIABLES:
        real(C_DOUBLE), dimension(3) :: r_v                   !look vector

!c       LOCAL VARIABLES:
        real*8 r_temp(3),r_t(3),r_c(3),r_n(3),r_w(3)
        integer i


!c       PROCESSING STEPS:

!c       compute a TCN basis vector set

        call unitvec(pos%r_j,r_n)
        do i=1,3
          r_n(i) = -r_n(i)
        enddo

        call cross(r_n,pos%r_jdot,r_temp)
        call unitvec(r_temp,r_c)

        call cross(r_c,r_n,r_temp)
        call unitvec(r_temp,r_t)

!c       compute the look vector

        call lincomb(cos(r_az),r_t,sin(r_az),r_c,r_temp)
        call lincomb(cos(r_look),r_n,sin(r_look),r_temp,r_w)    
        call unitvec(r_w,r_v)
      
        end subroutine lookvec 
