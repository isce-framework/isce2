!c****************************************************************

        subroutine getangs(pos,vel,vec,elp,r_az,r_lk)BIND(C, NAME='getangs_C')

!c****************************************************************
!c**
!c**    FILE NAME: getangs.f
!c**
!c**     DATE WRITTEN: 4-94
!c**
!c**     PROGRAMMER:par
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
        real(C_DOUBLE), dimension(3) :: pos
        real(C_DOUBLE), dimension(3) :: vel
        real(C_DOUBLE), dimension(3) :: vec

!c      OUTPUT VARIABLES:
        real(C_DOUBLE) :: r_az
        real(C_DOUBLE) :: r_lk

!c      LOCAL VARIABLES:
        real*8 tvt, tvc,tvn, r_a, r_e2, lat, lon, rad
        real*8 r_temp(3),r_t(3),r_c(3),r_n(3), r_dd
        real*8 r_vecnorm, r_llh(3)

        type (ellipsoidType) elp

!C      FUNCTION STATEMENTS:
        
!        real(C_DOUBLE) dot

!c      PROCESSING STEPS:

!c      compute a TCN basis vector set

        call latlon(elp,pos,r_llh,XYZ_2_LLH)
        lat = r_llh(1)
        lon = r_llh(2)
        rad = r_llh(3)

        r_n(1) = -cos(lat)*cos(lon)
        r_n(2) = -cos(lat)*sin(lon)
        r_n(3) = -sin(lat)


        r_dd = dot(r_n,vec)
        r_vecnorm = norm(vec)
        r_lk = acos(r_dd/r_vecnorm)

        call cross(r_n,vel,r_temp)
        call unitvec(r_temp,r_c)

        call cross(r_c,r_n,r_temp)
        call unitvec(r_temp,r_t)

!c       compute the angles
        tvt = dot(r_t,vec)
        tvc = dot(r_c,vec)
        tvn = dot(r_n,vec)

        r_az = atan2(tvc,tvt)
        end subroutine getangs 

