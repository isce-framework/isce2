!c****************************************************************

        subroutine getTCN_TCvec(pos,vel,vec,elp,TCvec)BIND(C,NAME='getTCN_TCvec_C')

!c****************************************************************
!c**
!c**    FILE NAME: getTCN_TCvec
!c**
!c**     DATE WRITTEN: 3-97
!c**
!c**     PROGRAMMER:par
!c**
!c**    FUNCTIONAL DESCRIPTION: This subroutine will compute the 
!c**    projection of an xyz vector on the TC plane, in xyz
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
        real(C_DOUBLE), dimension(3) :: pos
        real(C_DOUBLE), dimension(3) :: vel
        real(C_DOUBLE), dimension(3) :: vec
        
        type (ellipsoidType) elp

!c      OUTPUT VARIABLES:
        real(C_DOUBLE), dimension(3) :: TCVec

!c      LOCAL VARIABLES:
        real*8 tvt, tvc,tvn, r_a, r_e2, lat, lon, rad
        real*8 r_temp(3),r_t(3),r_c(3),r_n(3), r_llh(3)
        integer i


!C      FUNCTION STATEMENTS:

!        real(C_DOUBLE) :: dot


!c      compute a TCN basis vector set

        call latlon(elp,pos,r_llh,XYZ_2_LLH)
        lat = r_llh(1) 
        lon = r_llh(2) 
        rad = r_llh(3) 

        r_n(1) = -cos(lat)*cos(lon)
        r_n(2) = -cos(lat)*sin(lon)
        r_n(3) = -sin(lat)

        call cross(r_n,vel,r_temp)
        call unitvec(r_temp,r_c)

        call cross(r_c,r_n,r_temp)
        call unitvec(r_temp,r_t)

!c      compute the angles

        tvt = dot(r_t,vec)
        tvc = dot(r_c,vec)
        tvn = dot(r_n,vec)
        do i = 1 , 3
           TCvec(i) = tvt * r_t(i) + tvc * r_c(i)
        end do

        end subroutine gettcn_tcvec

