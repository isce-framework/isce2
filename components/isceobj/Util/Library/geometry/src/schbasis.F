!c****************************************************************
        
        subroutine schbasis(ptm,r_sch,r_xyzschmat,r_schxyzmat)BIND(C,NAME='schbasis_C')
        
!c****************************************************************
!c**     
!c**    FILE NAME: schbasis.f
!c**     
!c**     DATE WRITTEN: 10/01/97 
!c**     
!c**     PROGRAMMER: Scott Hensley
!c**     
!c**    FUNCTIONAL DESCRIPTION: This routine computes the transformation
!c**     matrix from xyz to a local sch frame. 
!c**     
!c**     ROUTINES CALLED:
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

        type (pegtransType) ptm

        real(C_DOUBLE), dimension(3) :: r_sch   !SCH position
   
!c      OUTPUT VARIABLES:

        real(C_DOUBLE), dimension(3,3) :: r_xyzschmat
        real(C_DOUBLE), dimension(3,3) :: r_schxyzmat

!c      LOCAL VARIABLES:

        real*8  r_coss,r_cosc,r_sins,r_sinc
        real*8 r_matschxyzp(3,3)



!c      PROCESSING STEPS:

!c       compute transformation from a sch local basis to X'Y'Z' basis

        r_coss = cos(r_sch(1)/ptm%r_radcur)
        r_sins = sin(r_sch(1)/ptm%r_radcur)

        r_cosc = cos(r_sch(2)/ptm%r_radcur)
        r_sinc = sin(r_sch(2)/ptm%r_radcur)

        r_matschxyzp(1,1) = -r_sins 
        r_matschxyzp(1,2) = -r_sinc*r_coss
        r_matschxyzp(1,3) = r_coss*r_cosc
        r_matschxyzp(2,1) = r_coss
        r_matschxyzp(2,2) = -r_sinc*r_sins
        r_matschxyzp(2,3) = r_sins*r_cosc
        r_matschxyzp(3,1) = 0.0
        r_matschxyzp(3,2) = r_cosc
        r_matschxyzp(3,3) = r_sinc
        
!c       compute sch to xyz matrix

        call matmat(ptm%r_mat,r_matschxyzp,r_schxyzmat)

!c       get the inverse

        call tranmat(r_schxyzmat,r_xyzschmat)

        end subroutine schbasis 




