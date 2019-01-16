!c****************************************************************

      subroutine get_tpsch(ptm1,r_sch1,ptm2,r_sch2,r_tpsch)

!c****************************************************************
!c**     
!c**   FILE NAME: get_tpsch.f
!c**     
!c**   DATE WRITTEN: 11/02/98
!c**     
!c**   PROGRAMMER: Scott Hensley
!c** 
!c**   FUNCTIONAL DESCRIPTION: The routine will take two sch positions,
!c**   possibibly in different SCH frames and generate the local SCH vector
!c**   pointing from the first position to the second position in the local
!c**   SCH frame of the first point.
!c**     
!c**   ROUTINES CALLED: convert_sch_to_xyz,lincomb,schbasis,matvec
!c**     
!c**   NOTES: 
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed                  CR # and Version #
!c**   ------------       ----------------                 -----------------
!c**     
!c*****************************************************************

      implicit none

!c     INCLUDE FILES:

!c     PARAMETER STATEMENTS:

      integer i_schtoxyz,i_xyztosch
      parameter(i_schtoxyz=0,i_xyztosch=1)

!c     INPUT VARIABLES:

      type :: pegtrans      !transformation parameters
         real*8 r_mat(3,3)      !Transformation matrix
         real*8 r_matinv(3,3)   !Inverse Transformation matrix
         real*8 r_ov(3)         !Offset vector
         real*8 r_radcur        !radius of curvature
      end type pegtrans
      type(pegtrans) :: ptm1,ptm2            !peg transformation parameters

      real*8 r_sch1(3)                       !SCH position first vector
      real*8 r_sch2(3)                       !SCH position second vector
	
!c     OUTPUT VARIABLES:

      real*8 r_tpsch(3)                     !local SCH resultant vector

!c     LOCAL VARIABLES:

      real*8 r_xyz1(3),r_xyz2(3),r_xyzout(3),r_schxyzmat(3,3),r_xyzschmat(3,3)

!c     COMMON BLOCKS:

!c     EQUIVALENCE STATEMENTS:

!c     DATA STATEMENTS:

!c     FUNCTION STATEMENTS:

!c     SAVE STATEMENTS:

!c     PROCESSING STEPS:

!c     convert both SCH positions to XYZ

      call convert_sch_to_xyz(ptm1,r_sch1,r_xyz1,i_schtoxyz)
      call convert_sch_to_xyz(ptm2,r_sch2,r_xyz2,i_schtoxyz)

!c     add vectors in XYZ

      call lincomb(1.d0,r_xyz2,-1.d0,r_xyz1,r_xyzout)

!c     convert resultant to SCH frame of first vector
      
      call schbasis(ptm1,r_sch1,r_xyzschmat,r_schxyzmat)
      call matvec(r_xyzschmat,r_xyzout,r_tpsch)
	
      end  




