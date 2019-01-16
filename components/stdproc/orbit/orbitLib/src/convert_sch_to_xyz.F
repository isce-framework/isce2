!c****************************************************************

        subroutine convert_sch_to_xyz(ptm,r_schv,r_xyzv,i_type)

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
!c**     ROUTINES CALLED: latlon,matvec,lincomb
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************

        implicit none

!c      INPUT VARIABLES:

        type :: pegtrans          !transformation parameters
           real*8 r_mat(3,3)          !Transformation matrix
           real*8 r_matinv(3,3)       !Inverse Transformation matrix
           real*8 r_ov(3)             !Offset vector
           real*8 r_radcur            !radius of curvature
        end type pegtrans
        type(pegtrans) :: ptm

        real*8 r_schv(3)              !sch coordinates of a point
        real*8 r_xyzv(3)              !WGS-84 coordinates of a point
        integer i_type                !i_type = 0 sch => xyz ; 
                                      !i_type = 1 xyz => sch
   
!c      OUTPUT VARIABLES: see input

!c      LOCAL VARIABLES:

        integer i_t
        real*8 r_schvt(3),r_llh(3)
        type :: ellipsoid              !Ellipsoid parameters
           real*8 r_a                      !semi-major axis
           real*8 r_e2                     !eccentricity squared
        end type ellipsoid
        type(ellipsoid) :: sph

!c              DATA STATEMENTS:

!C      FUNCTION STATEMENTS:none

!c      PROCESSING STEPS:

!c       compute the linear portion of the transformation 

        sph%r_a = ptm%r_radcur
        sph%r_e2 = 0.0d0

        if(i_type .eq. 0)then

           r_llh(1) = r_schv(2)/ptm%r_radcur
           r_llh(2) = r_schv(1)/ptm%r_radcur
           r_llh(3) = r_schv(3)

           i_t = 1
           call latlon(sph,r_schvt,r_llh,i_t)
           call matvec(ptm%r_mat,r_schvt,r_xyzv)
           call lincomb(1.d0,r_xyzv,1.d0,ptm%r_ov,r_xyzv)           

        elseif(i_type .eq. 1)then

           call lincomb(1.d0,r_xyzv,-1.d0,ptm%r_ov,r_schvt)
           call matvec(ptm%r_matinv,r_schvt,r_schv)
           i_t = 2
           call latlon(sph,r_schv,r_llh,i_t)
 
           r_schv(1) = ptm%r_radcur*r_llh(2)
           r_schv(2) = ptm%r_radcur*r_llh(1)
           r_schv(3) = r_llh(3)

        endif

        end




