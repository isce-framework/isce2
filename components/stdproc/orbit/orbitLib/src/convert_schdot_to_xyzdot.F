!c****************************************************************

        subroutine convert_schdot_to_xyzdot(ptm,r_sch,r_schdot,r_xyzdot,i_type)

!c****************************************************************
!c**
!c**    FILE NAME: convert_schdot_to_xyzdot.f
!c**
!c**     DATE WRITTEN:1/15/93 
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: This routine applies the affine matrix 
!c**     provided to convert the sch velocity to xyz WGS-84 velocity or
!c**     the inverse transformation.
!c**
!c**     ROUTINES CALLED: schbasis,matvec,dot
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************

        implicit none

!c      INPUT VARIABLES:

        type :: pegtrans           !transformation parameters        
           real*8 r_mat(3,3)           !Transformation matrix            
           real*8 r_matinv(3,3)        !Inverse Transformation matrix    
           real*8 r_ov(3)              !Offset vector                    
           real*8 r_radcur             !radius of curvature              
        end type pegtrans
        type(pegtrans) :: ptm

        real*8 r_sch(3)                 !sch coordinates of a point
        real*8 r_schdot(3)              !sch velocity
        real*8 r_xyzdot(3)              !WGS-84 velocity
        integer i_type                  !i_type = 0 sch => xyz 
                                        !i_type = 1 xyz => sch
   
!c      OUTPUT VARIABLES: see input

!c      LOCAL VARIABLES:

        real*8 r_schxyzmat(3,3),r_xyzschmat(3,3)

!c      DATA STATEMENTS:

!C      FUNCTION STATEMENTS:none

!c      PROCESSING STEPS:

!c       get the change of basis to the local tangent plane

        call schbasis(ptm,r_sch,r_xyzschmat,r_schxyzmat)

        if(i_type .eq. 0)then   !convert from sch velocity to xyz velocity
          
           call matvec(r_schxyzmat,r_schdot,r_xyzdot)

        elseif(i_type .eq. 1)then   !convert from xyz velocity to sch velocity

           call matvec(r_xyzschmat,r_xyzdot,r_schdot)

        endif
           
        end




