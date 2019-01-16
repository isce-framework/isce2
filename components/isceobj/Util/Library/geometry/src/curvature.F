!c****************************************************************
!c
!c       Various curvature functions
!c 
!c
!c****************************************************************
!c**
!c**    FILE NAME: curvature.f
!c**
!c**     DATE WRITTEN: 12/02/93
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: This routine computes the curvature for 
!c**     of various types required for ellipsoidal or spherical earth 
!c**     calculations.  
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************

        function  reast(elp,r_lat)BIND(C,NAME='reast_C')

        use, intrinsic :: iso_c_binding
        implicit none
        type(ellipsoidType) :: elp
        real(C_DOUBLE), value:: r_lat
        real(C_DOUBLE) :: reast
        
        reast = elp%r_a/sqrt(1.d0 - elp%r_e2*sin(r_lat)**2) 
      
        end function reast 

        function  rnorth(elp,r_lat)BIND(C,NAME='rnorth_C')
        use, intrinsic :: iso_c_binding
        implicit none
        type(ellipsoidType) :: elp
        real(C_DOUBLE), value :: r_lat
        real(C_DOUBLE)  :: rnorth
        
        rnorth = (elp%r_a*(1.d0 - elp%r_e2))/(1.d0 - elp%r_e2*sin(r_lat)**2)**(1.5d0) 

        end function rnorth

        function  rdir(elp,r_hdg,r_lat)BIND(C,NAME='rdir_C')

        use, intrinsic :: iso_c_binding
        implicit none
        type(ellipsoidType) :: elp
        real(C_DOUBLE) :: rdir
        real(C_DOUBLE), value :: r_hdg
        real(C_DOUBLE), value :: r_lat
        real*8 r_re,r_rn
        
        r_re = reast(elp,r_lat)
        r_rn = rnorth(elp,r_lat)

        rdir = (r_re*r_rn)/(r_re*cos(r_hdg)**2 + r_rn*sin(r_hdg)**2) 

        end function rdir 

