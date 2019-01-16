c****************************************************************

	subroutine radar_to_xyz(r_a,r_e2,r_lat0,r_lon0,r_hdg0,r_mat,r_ov)

c****************************************************************
c**
c**	FILE NAME: radar_to_xyz.for
c**
c**     DATE WRITTEN:1/15/93 
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION:This routine computes the transformation
c**     matrix and translation vector needed to get between radar (s,c,h)
c**     coordinates and (x,y,z) WGS-84 coordinates.
c**
c**     ROUTINES CALLED:euler,
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       	implicit none

c	INPUT VARIABLES:
        real*8 r_a                    !semimajor axis
        real*8 r_e2                   !eccentricity squared
	real*8 r_lat0                 !peg latitude
	real*8 r_lon0                 !peg longitude
	real*8 r_hdg0                 !peg heading
   
c   	OUTPUT VARIABLES:
        real*8 r_mat(3,3)             !rotation matrix
        real*8 r_ov(3)                !translation vector

c	LOCAL VARIABLES:
        integer i,i_type
        real*8 r_radcur,r_h,r_p(3),r_slt,r_clt,r_clo,r_slo,r_up(3)
        real*8 r_chg,r_shg,rdir

c	DATA STATEMENTS:none

C	FUNCTION STATEMENTS:
        external rdir

c  	PROCESSING STEPS:

c       first determine the rotation matrix

        r_clt = cos(r_lat0)
        r_slt = sin(r_lat0)
        r_clo = cos(r_lon0)
        r_slo = sin(r_lon0)
        r_chg = cos(r_hdg0)
        r_shg = sin(r_hdg0)

	r_mat(1,1) = r_clt*r_clo
	r_mat(1,2) = -r_shg*r_slo - r_slt*r_clo*r_chg
	r_mat(1,3) = r_slo*r_chg - r_slt*r_clo*r_shg
	r_mat(2,1) = r_clt*r_slo 
	r_mat(2,2) = r_clo*r_shg - r_slt*r_slo*r_chg 
	r_mat(2,3) = -r_clo*r_chg - r_slt*r_slo*r_shg
	r_mat(3,1) = r_slt
	r_mat(3,2) = r_clt*r_chg
	r_mat(3,3) = r_clt*r_shg

c       find the translation vector

        r_radcur = rdir(r_a,r_e2,r_hdg0,r_lat0)

        i_type = 1
        r_h = 0.d0
c        call latlon(r_a,r_e2,r_p,r_lat0,r_lon0,r_h,i_type)
        call latlon_elp(r_a,r_e2,r_p,r_lat0,r_lon0,r_h,i_type)

        r_clt = cos(r_lat0)
        r_slt = sin(r_lat0)
        r_clo = cos(r_lon0)
        r_slo = sin(r_lon0)
        r_up(1) = r_clt*r_clo        
        r_up(2) = r_clt*r_slo
        r_up(3) = r_slt        

        do i=1,3
	   r_ov(i) = r_p(i) - r_radcur*r_up(i)
	enddo

        end  


