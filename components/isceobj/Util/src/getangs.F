c****************************************************************

	subroutine getangs(pos,vel,vec,r_az,r_lk)

c****************************************************************
c**
c**	FILE NAME: getangs.f
c**
c**     DATE WRITTEN: 4-94
c**
c**     PROGRAMMER:par
c**
c** 	FUNCTIONAL DESCRIPTION: This subroutine will compute the look
c**	vector given the look angle,azimuth angle, and the position
c**	vector. 
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

c   	OUTPUT VARIABLES:
        real*8 r_az, r_lk

c	LOCAL VARIABLES:
        real*8 tvt, tvc,tvn, r_a, r_e2, lat, lon, rad
        real*8 r_temp(3),r_t(3),r_c(3),r_n(3), r_dd
	real*8 r_vecnorm, r_llh(3)

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

	common /ellipsoid/ r_a, r_e2

c	DATA STATEMENTS:

	integer i_xyztollh,i_llhtoxyz
	parameter(i_xyztollh=2, i_llhtoxyz=1) 

C	FUNCTION STATEMENTS:
	
	real*8 dot

c  	PROCESSING STEPS:

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

c only good for sphere
c
c	call unitvec(pos,r_n)
c        do i=1,3
c          r_n(i) = -r_n(i)
c        enddo

	r_dd = dot(r_n,vec)
	call norm(vec,r_vecnorm)
	r_lk = acos(r_dd/r_vecnorm)

	call cross(r_n,vel,r_temp)
        call unitvec(r_temp,r_c)

	call cross(r_c,r_n,r_temp)
        call unitvec(r_temp,r_t)

c	compute the angles
        tvt = dot(r_t,vec)
        tvc = dot(r_c,vec)
        tvn = dot(r_n,vec)

        r_az = atan2(tvc,tvt)
c       r_lk = atan2(tvc,tvn)
c	r_lk = -999999.
        end  

