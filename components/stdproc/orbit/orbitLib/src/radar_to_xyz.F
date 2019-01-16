!c****************************************************************

	subroutine radar_to_xyz(elp,peg,ptm)

!c****************************************************************
!c**
!c**	FILE NAME: radar_to_xyz.f
!c**
!c**     DATE WRITTEN:1/15/93 
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c** 	FUNCTIONAL DESCRIPTION: This routine computes the transformation
!c**     matrix and translation vector needed to get between radar (s,c,h)
!c**     coordinates and (x,y,z) WGS-84 coordinates.
!c**
!c**     ROUTINES CALLED: latlon,rdir
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************

       	implicit none

!c	INPUT VARIABLES:

	type :: ellipsoid 
	   real*8 r_a		! semi-major axis
	   real*8 r_e2		! eccentricity-squared of earth ellisoid
	end type ellipsoid
	type(ellipsoid) :: elp

	type :: pegpoint 
	   real*8 r_lat		! peg latitude
	   real*8 r_lon		! peg longitude
	   real*8 r_hdg		! peg heading
        end type pegpoint
	type(pegpoint) :: peg
   
!c   	OUTPUT VARIABLES:

	type :: pegtrans 
	   real*8 r_mat(3,3)	!transformation matrix SCH->XYZ
	   real*8 r_matinv(3,3)	!transformation matrix XYZ->SCH
	   real*8 r_ov(3)	!Offset vector SCH->XYZ
	   real*8 r_radcur	!peg radius of curvature
	end type pegtrans
	type(pegtrans) :: ptm

!c	LOCAL VARIABLES:

        integer i,j,i_type
        real*8 pi,r_radcur,r_llh(3),r_p(3),r_slt,r_clt,r_clo,r_slo,r_up(3)
        real*8 r_chg,r_shg,rdir

!c	DATA STATEMENTS: none

!C	FUNCTION STATEMENTS:

        external rdir

!c  	PROCESSING STEPS:

!c       first determine the rotation matrix

        r_clt = cos(peg%r_lat)
        r_slt = sin(peg%r_lat)
        r_clo = cos(peg%r_lon)
        r_slo = sin(peg%r_lon)
        r_chg = cos(peg%r_hdg)
        r_shg = sin(peg%r_hdg)

	ptm%r_mat(1,1) = r_clt*r_clo
	ptm%r_mat(1,2) = -r_shg*r_slo - r_slt*r_clo*r_chg
	ptm%r_mat(1,3) = r_slo*r_chg - r_slt*r_clo*r_shg
	ptm%r_mat(2,1) = r_clt*r_slo 
	ptm%r_mat(2,2) = r_clo*r_shg - r_slt*r_slo*r_chg 
	ptm%r_mat(2,3) = -r_clo*r_chg - r_slt*r_slo*r_shg
	ptm%r_mat(3,1) = r_slt
	ptm%r_mat(3,2) = r_clt*r_chg
	ptm%r_mat(3,3) = r_clt*r_shg

	do i=1,3
	   do j=1,3
	      ptm%r_matinv(i,j) = ptm%r_mat(j,i)
	   enddo
	enddo

!c       find the translation vector

        ptm%r_radcur = rdir(elp%r_a,elp%r_e2,peg%r_hdg,peg%r_lat)

        i_type = 1
	r_llh(1) = peg%r_lat
	r_llh(2) = peg%r_lon
	r_llh(3) = 0.0d0
        call latlon(elp,r_p,r_llh,i_type)

        r_clt = cos(peg%r_lat)
        r_slt = sin(peg%r_lat)
        r_clo = cos(peg%r_lon)
        r_slo = sin(peg%r_lon)
        r_up(1) = r_clt*r_clo        
        r_up(2) = r_clt*r_slo
        r_up(3) = r_slt        

        do i=1,3
	   ptm%r_ov(i) = r_p(i) - ptm%r_radcur*r_up(i)
	enddo

        end  


