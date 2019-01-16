!c****************************************************************

        subroutine tcnbasis(pos,vel,elp,r_t,r_c,r_n)BIND(C,NAME='tcnbasis_C')

        use linalg3module
        implicit none

!c      INPUT VARIABLES:
        real(C_DOUBLE), dimension(3) :: pos
        real(C_DOUBLE), dimension(3) :: vel
        
        type (ellipsoidType) elp

!c      OUTPUT VARIABLES:
        real(C_DOUBLE), dimension(3) :: r_c, r_t, r_n

!c      LOCAL VARIABLES:
        real*8 tvt, tvc,tvn, r_a, r_e2, lat, lon, rad
        real*8 r_temp(3),r_llh(3)
        integer i



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

        end subroutine tcnbasis

