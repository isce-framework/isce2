!c****************************************************************

        subroutine convert_sch_to_xyz(ptm,r_schv,r_xyzv,i_type)BIND(C,NAME='convert_sch_to_xyz_C')

!c****************************************************************
!c**
!c**    FILE NAME: convert_sch_to_xyz.for
!c**
!c**     DATE WRITTEN:1/15/93 
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: This routine applies the affine matrix 
!c**     provided to convert the sch coordinates xyz WGS-84 coordintes or
!c**     the inverse transformation.
!c**
!c**     ROUTINES CALLED:latlon,matvec
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************
        use, intrinsic :: iso_c_binding
        use linalg3module
        implicit none
        

        type(pegtransType) :: ptm
        
        
        real(C_DOUBLE), dimension(3) ::  r_schv  !sch coordinates of a point
        real(C_DOUBLE), dimension(3) ::  r_xyzv   !WGS-84 coordinates of a point
        integer(C_INT), value ::  i_type     !i_type = 0 sch => xyz ; 
                                      !i_type = 1 xyz => sch
   

!c      LOCAL VARIABLES:
        integer i_t
        real*8 r_schvt(3),r_llh(3)
        type (ellipsoidType) sph



!c      PROCESSING STEPS:

!c       compute the linear portion of the transformation 

        sph%r_a = ptm%r_radcur
        sph%r_e2 = 0.0d0

        if(i_type .eq. SCH_2_XYZ)then

           r_llh(1) = r_schv(2)/ptm%r_radcur
           r_llh(2) = r_schv(1)/ptm%r_radcur
           r_llh(3) = r_schv(3)

           i_t = LLH_2_XYZ
           call latlon(sph,r_schvt,r_llh,i_t)
           call matvec(ptm%r_mat,r_schvt,r_xyzv)
           call lincomb(1.d0,r_xyzv,1.d0,ptm%r_ov,r_xyzv)           

        elseif(i_type .eq. XYZ_2_SCH)then

           call lincomb(1.d0,r_xyzv,-1.d0,ptm%r_ov,r_schvt)
           call matvec(ptm%r_matinv,r_schvt,r_schv)
           i_t = XYZ_2_LLH
           call latlon(sph,r_schv,r_llh,i_t)
 
           r_schv(1) = ptm%r_radcur*r_llh(2)
           r_schv(2) = ptm%r_radcur*r_llh(1)
           r_schv(3) = r_llh(3)

        endif

        end subroutine convert_sch_to_xyz




