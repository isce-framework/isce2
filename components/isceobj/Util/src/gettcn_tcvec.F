c****************************************************************

	subroutine getTCN_TCvec(pos,vel,vec,TCvec)

c****************************************************************
c**
c**	FILE NAME: getTCN_TCvec
c**
c**     DATE WRITTEN: 3-97
c**
c**     PROGRAMMER:par
c**
c** 	FUNCTIONAL DESCRIPTION: This subroutine will compute the 
c**	projection of an xyz vector on the TC plane, in xyz
c**
c**     ROUTINES CALLED:cross,unitvec,lincomb
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       	implicit none

c	INPUT VARIABLES:
	real*8 pos(3), vel(3)
        real*8 vec(3)

c	structure /ellipsoid/ 
c	   real*8 r_a        
c	   real*8 r_e2
c	end structure
c	record /ellipsoid/ elp
        
	type ellipsoid
           sequence
	   real (8) r_a        
	   real (8) r_e2
	end type ellipsoid
	type (ellipsoid) elp

c   	OUTPUT VARIABLES:
	real*8 TCVec(3)

c	LOCAL VARIABLES:
        real*8 tvt, tvc,tvn, r_a, r_e2, lat, lon, rad
        real*8 r_temp(3),r_t(3),r_c(3),r_n(3), r_llh(3)
        integer i

	integer i_xyztollh,i_llhtoxyz
	parameter(i_xyztollh=2, i_llhtoxyz=1) 

c	DATA STATEMENTS:

C	FUNCTION STATEMENTS:

	real*8 dot

c  	PROCESSING STEPS:

	common /ellipsoid/ r_a, r_e2

c	compute a TCN basis vector set

	elp%r_a = r_a
	elp%r_e2 = r_e2
	call latlon(elp,pos,r_llh,i_xyztollh)
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

c	compute the angles

        tvt = dot(r_t,vec)
        tvc = dot(r_c,vec)
        tvn = dot(r_n,vec)
	do i = 1 , 3
	   TCvec(i) = tvt * r_t(i) + tvc * r_c(i)
	end do
        end

