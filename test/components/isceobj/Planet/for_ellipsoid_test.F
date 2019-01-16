       program ellipsoid_test

	type ellipsoid
           sequence
	   real (8) r_a        
	   real (8) r_e2
	end type ellipsoid
	type (ellipsoid) elp

	type pegtype
           sequence
	   real (8) r_lat
	   real (8) r_lon
	   real (8) r_hdg
	end type pegtype
	type (pegtype) peg
   
c   	OUTPUT VARIABLES:

	type pegtrans
          sequence
	  real (8) r_mat(3,3)
	  real (8) r_matinv(3,3)
	  real (8) r_ov(3)
	  real (8) r_radcur
        end type pegtrans
	type (pegtrans) ptm

        real(8) r_xyz(3), r_llh(3), r_sch(3), r_xyzdot(3), r_schdot(3)
        real(8), parameter :: r2d = 180.0d0/acos(-1.0d0)
        real(8), parameter :: d2r = acos(-1.0d0)/180.0d0
        integer, parameter :: i_llh_to_xyz = 1
        integer, parameter :: i_xyz_to_llh = 2
        integer, parameter :: i_sch_to_xyz = 0
        integer, parameter :: i_xyz_to_sch = 1
        integer i, j

        elp%r_a = 6378137.0d0
        elp%r_e2 = 0.0066943799901d0

        print*, "latlon xyz_to_llh"
        r_xyz(1) =  7000000.0d0
        r_xyz(2) = -7500000.0d0
        r_xyz(3) =  8000000.0d0
        print*, "r_xyz = ", r_xyz
        call latlon(elp,r_xyz,r_llh,i_xyz_to_llh)
        print*, "r_llh = ", r_llh(1)*r2d, r_llh(2)*r2d, r_llh(3)
        print*
        
        print*, "latlon llh_to_xyz"
        r_llh(1) = -33.0d0
        r_llh(2) = 118.0d0
        r_llh(3) = 2000.0d0
        print*, "r_llh = ", r_llh
        r_llh(1) = r_llh(1)*d2r
        r_llh(2) = r_llh(2)*d2r
        call latlon(elp,r_xyz,r_llh,i_llh_to_xyz)
        print*, "r_xyz = ", r_xyz
        print*

        peg%r_lat =   66.0d0
        peg%r_lon = -105.0d0
        peg%r_hdg =   36.0d0
        print*, "peg point = ", peg%r_lat, peg%r_lon, peg%r_hdg
        peg%r_lat = peg%r_lat*d2r
        peg%r_lon = peg%r_lon*d2r
        peg%r_hdg = peg%r_hdg*d2r
	call radar_to_xyz(elp,peg,ptm)
        print*, "ptm%r_radcur = ", ptm%r_radcur
        print*, "ptm%r_ov = ", ptm%r_ov
        print*, "ptm%r_mat = "
        do i = 1, 3
            print*, "    row1: ", (ptm%r_mat(i,j), j=1,3)
        enddo
        print*, "ptm%r_matinv = "
        do i = 1, 3
            print*, "    row1: ", (ptm%r_matinv(i,j), j=1,3)
        enddo
        print*

        print*, "convert_sch_to_xyz, sch_to_xyz"
        r_sch(1) =    1468.0d0
        r_sch(2) =    -234.0d0
        r_sch(3) =    7000.0d0
        print*, "r_sch = ", r_sch
	call convert_sch_to_xyz(ptm,r_sch,r_xyz,i_sch_to_xyz)
        print*, "r_xyz = ", r_xyz
        call latlon(elp,r_xyz,r_llh,i_xyz_to_llh)
        print*, "r_llh = ", r_llh(1)*r2d, r_llh(2)*r2d, r_llh(3)
        print*

        print*, "convert_sch_to_xyz, xyz_to_sch"
        r_xyz(1) =  -672100.0d0
        r_xyz(2) = -2514000.0d0
        r_xyz(3) =  5811000.0d0
        print*, "r_xyz = ", r_xyz
	call convert_sch_to_xyz(ptm,r_sch,r_xyz,i_xyz_to_sch)
        print*, "r_sch = ", r_sch
        call latlon(elp,r_xyz,r_llh,i_xyz_to_llh)
        print*, "r_llh = ", r_llh(1)*r2d, r_llh(2)*r2d, r_llh(3)
        print*

        
        print*, "convert_schdot_to_xyzdot, sch_to_xyz"
        r_sch(1) =    1468.0d0
        r_sch(2) =    -234.0d0
        r_sch(3) =    7000.0d0
        print*, "r_sch = ", r_sch
        r_schdot(1) =  800.0d0
        r_schdot(2) = -400.0d0
        r_schdot(3) =  100.0d0
        print*, "r_schdot = ", r_schdot
        call convert_schdot_to_xyzdot(ptm,r_sch,r_xyz,r_schdot,r_xyzdot,i_sch_to_xyz)
        call convert_sch_to_xyz(ptm,r_sch,r_xyz,i_sch_to_xyz)
        print*, "r_xyz = ", r_xyz
        print*, "r_xyzdot = ", r_xyzdot


        print*, "convert_schdot_to_xyzdot, xyz_to_sch"
        r_xyz(1) =  -672100.0d0
        r_xyz(2) = -2514000.0d0
        r_xyz(3) =  5811000.0d0
        print*, "r_xyz = ", r_xyz
        r_xyzdot(1) =  800.0d0
        r_xyzdot(2) = -400.0d0
        r_xyzdot(3) =  100.0d0
        print*, "r_xyzdot = ", r_xyzdot
        call convert_sch_to_xyz(ptm,r_sch,r_xyz,i_xyz_to_sch)
        call convert_schdot_to_xyzdot(ptm,r_sch,r_xyz,r_schdot,r_xyzdot,i_xyz_to_sch)
        print*, "r_sch = ", r_sch
        print*, "r_schdot = ", r_schdot

        end


c****************************************************************

	subroutine radar_to_xyz(elp,peg,ptm)

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

	type ellipsoid
           sequence
	   real (8) r_a        
	   real (8) r_e2
	end type ellipsoid
	type (ellipsoid) elp

	type pegtype
           sequence
	   real (8) r_lat
	   real (8) r_lon
	   real (8) r_hdg
	end type pegtype
	type (pegtype) peg
   
c   	OUTPUT VARIABLES:

	type pegtrans
          sequence
	  real (8) r_mat(3,3)
	  real (8) r_matinv(3,3)
	  real (8) r_ov(3)
	  real (8) r_radcur
        end type pegtrans
	type (pegtrans) ptm

c	LOCAL VARIABLES:
        integer i,j,i_type
        real*8 r_llh(3),r_p(3),r_slt,r_clt,r_clo,r_slo,r_up(3)
        real*8 r_chg,r_shg,rdir

c	DATA STATEMENTS:none

C	FUNCTION STATEMENTS:
        external rdir

c  	PROCESSING STEPS:

c       first determine the rotation matrix

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

c       find the translation vector

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

c****************************************************************
c
c       Various curvature functions
c 
c
c****************************************************************
c**
c**	FILE NAME: curvature.f
c**
c**     DATE WRITTEN: 12/02/93
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: This routine computes the curvature for 
c**     of various types required for ellipsoidal or spherical earth 
c**     calculations.  
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

        real*8 function  reast(r_a,r_e2,r_lat)

       	implicit none
        real*8 r_a,r_e2,r_lat
        
        reast = r_a/sqrt(1.d0 - r_e2*sin(r_lat)**2) 
      
        end  

        real*8 function  rnorth(r_a,r_e2,r_lat)

       	implicit none
        real*8 r_a,r_e2,r_lat
        
        rnorth = (r_a*(1.d0 - r_e2))/(1.d0 - r_e2*sin(r_lat)**2)**(1.5d0) 

        end

        real*8 function  rdir(r_a,r_e2,r_hdg,r_lat)

       	implicit none
        real*8 r_a,r_e2,r_lat,r_hdg,r_re,r_rn,reast,rnorth
        
        r_re = reast(r_a,r_e2,r_lat)
        r_rn = rnorth(r_a,r_e2,r_lat)

        rdir = (r_re*r_rn)/(r_re*cos(r_hdg)**2 + r_rn*sin(r_hdg)**2) 

        end      




c****************************************************************

	subroutine convert_sch_to_xyz(ptm,r_schv,r_xyzv,i_type)

c****************************************************************
c**
c**	FILE NAME: convert_sch_to_xyz.for
c**
c**     DATE WRITTEN:1/15/93 
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: This routine applies the affine matrix 
c**     provided to convert the sch coordinates xyz WGS-84 coordintes or
c**     the inverse transformation.
c**
c**     ROUTINES CALLED:latlon,matvec
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       	implicit none

c	INPUT VARIABLES:

c	structure /pegtrans/          !transformation parameters
c	   real*8 r_mat(3,3)
c	   real*8 r_matinv(3,3)
c	   real*8 r_ov(3)
c	   real*8 r_radcur
c	end structure
c	record /pegtrans/ ptm

	type pegtrans
          sequence
	  real (8) r_mat(3,3)
	  real (8) r_matinv(3,3)
	  real (8) r_ov(3)
	  real (8) r_radcur
        end type pegtrans
	type (pegtrans) ptm

        real*8 r_schv(3)              !sch coordinates of a point
        real*8 r_xyzv(3)              !WGS-84 coordinates of a point
        integer i_type                !i_type = 0 sch => xyz ; 
                                      !i_type = 1 xyz => sch
   
c   	OUTPUT VARIABLES: see input

c	LOCAL VARIABLES:
        integer i_t
        real*8 r_schvt(3),r_llh(3)
c	structure /ellipsoid/ 
c	   real*8 r_a        
c	   real*8 r_e2
c	end structure
c	record /ellipsoid/ sph

	type ellipsoid
           sequence
	   real (8) r_a        
	   real (8) r_e2
	end type ellipsoid
	type (ellipsoid) sph

c	DATA STATEMENTS:

C	FUNCTION STATEMENTS:none

c  	PROCESSING STEPS:

c       compute the linear portion of the transformation 

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


c****************************************************************

	subroutine convert_schdot_to_xyzdot(ptm,r_sch,r_xyz,r_schdot,
     +      r_xyzdot,i_type)

c****************************************************************
c**
c**	FILE NAME: convert_schdot_to_xyzdot.f
c**
c**     DATE WRITTEN:1/15/93 
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: This routine applies the affine matrix 
c**     provided to convert the sch velocity to xyz WGS-84 velocity or
c**     the inverse transformation.
c**
c**     ROUTINES CALLED: latlon,matvec
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       	implicit none

c	INPUT VARIABLES:

c	structure /pegtrans/          !transformation parameters
c	   real*8 r_mat(3,3)
c	   real*8 r_matinv(3,3)
c	   real*8 r_ov(3)
c	   real*8 r_radcur
c	end structure
c	record /pegtrans/ ptm

	type pegtrans            	!transformation parameters
           sequence
	   real*8 r_mat(3,3)
	   real*8 r_matinv(3,3)
	   real*8 r_ov(3)
	   real*8 r_radcur
	end type pegtrans
	type (pegtrans) ptm

	real*8 r_sch(3)                 !sch coordinates of a point
	real*8 r_xyz(3)                 !xyz coordinates of a point
        real*8 r_schdot(3)              !sch velocity
        real*8 r_xyzdot(3)              !WGS-84 velocity
        integer i_type                  !i_type = 0 sch => xyz 
                                        !i_type = 1 xyz => sch
   
c   	OUTPUT VARIABLES: see input

c	LOCAL VARIABLES:

        real*8 r_cs,r_ss,r_cc,r_sc,r_hu,r_huf,r_temp(3),r_vpxyz(3)
        real*8 r_tv(3),r_xp(3),r_xtemp,r_xn,r_xpr,r_xndot

c	DATA STATEMENTS:

C	FUNCTION STATEMENTS:none

c  	PROCESSING STEPS:

	if(i_type .eq. 0)then	!convert from sch velocity to xyz velocity
	   
c       To convert the velocity data, transfer the s and c velocities
c       to the surface and then compute the xyz prime velocity
	   
	   r_cs =  cos(r_sch(1)/ptm%r_radcur)
	   r_ss =  sin(r_sch(1)/ptm%r_radcur)
	   r_cc =  cos(r_sch(2)/ptm%r_radcur)
	   r_sc =  sin(r_sch(2)/ptm%r_radcur)
	   
	   r_hu = ptm%r_radcur + r_sch(3)
	   r_hu = ptm%r_radcur/r_hu
	   r_huf = 1.d0/r_hu
	   r_temp(1) = r_schdot(1)*r_hu*r_cc
	   r_temp(2) = r_schdot(2)*r_hu
	   
c       compute the primed velocity
	   
	   r_vpxyz(1) = -r_huf*r_cc*r_ss*r_temp(1) - r_huf*r_sc*r_cs*
     +	        r_temp(2) + r_cc*r_cs*r_schdot(3)
	   r_vpxyz(2) = r_huf*r_cc*r_cs*r_temp(1) - r_huf*r_sc*r_ss*
     +	        r_temp(2) + r_cc*r_ss*r_schdot(3)
	   r_vpxyz(3) = r_huf*r_cc*r_temp(2) + r_sc*r_schdot(3)
	   
c       convert to xyz velocity (WGS-84) 
	   
	   call matvec(ptm%r_mat,r_vpxyz,r_xyzdot)

	elseif(i_type .eq. 1)then   !convert from xyz velocity to sch velocity
	   
c       convert xyz position and velocity to primed position and velocity
	   
	   call matvec(ptm%r_matinv,r_xyzdot,r_vpxyz)
	   call lincomb(1.d0,r_xyz,-1.d0,ptm%r_ov,r_tv)
	   call matvec(ptm%r_matinv,r_tv,r_xp)
	   
c       convert to an sch velocity
	   
	   r_xtemp = ptm%r_radcur + r_sch(3)
	   r_xp(1) = r_xtemp*cos(r_sch(2)/ptm%r_radcur)*
     +	        cos(r_sch(1)/ptm%r_radcur)
	   r_xp(2) = r_xtemp*cos(r_sch(2)/ptm%r_radcur)* 
     +	        sin(r_sch(1)/ptm%r_radcur)
	   r_xp(3) = r_xtemp*sin(r_sch(2)/ptm%r_radcur)
	   
	   r_xn = sqrt(r_xp(1)**2+r_xp(2)**2+r_xp(3)**2)
	   r_xpr = r_xp(1)**2 + r_xp(2)**2
	   r_xndot = (r_xp(1)*r_vpxyz(1) + r_xp(2)*r_vpxyz(2) + 
     +	        r_xp(3)*r_vpxyz(3))/r_xn
	   
	   r_schdot(1) =  (ptm%r_radcur/r_xpr)*(r_xp(1)*
     +	        r_vpxyz(2)-r_xp(2)*r_vpxyz(1))
	   r_schdot(2) = (ptm%r_radcur/(r_xn*sqrt(r_xpr)))*
     +	        (r_xn*r_vpxyz(3) - r_xp(3)*r_xndot)
	   r_schdot(3) = r_xndot
	   
c       rescale to aircraft height
	   
	   r_schdot(1) = (sqrt(r_xpr)/ptm%r_radcur)*r_schdot(1)
	   r_schdot(2) = (r_xn/ptm%r_radcur)*r_schdot(2)
	   
	endif

	end



c****************************************************************
        subroutine latlon(elp,r_v,r_llh,i_type) 

c****************************************************************
c**   
c**   FILE NAME: latlon.f
c**   
c**   DATE WRITTEN:7/22/93 
c**   
c**   PROGRAMMER:Scott Hensley
c**   
c**   FUNCTIONAL DESCRIPTION:This program converts a vector to 
c**   lat,lon and height above the reference ellipsoid or given a
c**   lat,lon and height produces a geocentric vector. 
c**   
c**   ROUTINES CALLED:none
c**   
c**   NOTES: none
c**   
c**   UPDATE LOG:
c**   
c****************************************************************
        
        implicit none
        
c       INPUT VARIABLES:
        integer i_type          !1=lat,lon to vector,2= vector to lat,lon
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

        real*8 r_v(3)                    !geocentric vector (meters)
        real*8 r_llh(3)                  !latitude (deg -90 to 90),longitude (deg -180 to 180),height
   
c       OUTPUT VARIABLES: see input

c       LOCAL VARIABLES:
        real*8 pi,r_dtor,r_re,r_q2,r_q3,r_b,r_q
        real*8 r_p,r_tant,r_theta,r_a,r_e2

c       DATA STATEMENTS:
        data pi /3.141592653589793238d0/
        data r_dtor /1.74532925199d-2/

C       FUNCTION STATEMENTS:

c       PROCESSING STEPS:

        r_a = elp%r_a
        r_e2 = elp%r_e2

        if(i_type .eq. 1)then  !convert lat,lon to vector
           
           r_re = r_a/sqrt(1.d0 - r_e2*sin(r_llh(1))**2)
           
           r_v(1) = (r_re + r_llh(3))*cos(r_llh(1))*cos(r_llh(2))
           r_v(2) = (r_re + r_llh(3))*cos(r_llh(1))*sin(r_llh(2))
           r_v(3) = (r_re*(1.d0-r_e2) + r_llh(3))*sin(r_llh(1))               
           
        elseif(i_type .eq. 2)then  !convert vector to lat,lon 
           
           r_q2 = 1.d0/(1.d0 - r_e2)
           r_q = sqrt(r_q2)
           r_q3 = r_q2 - 1.d0
           r_b = r_a*sqrt(1.d0 - r_e2)
           
           r_llh(2) = atan2(r_v(2),r_v(1))
           
           r_p = sqrt(r_v(1)**2 + r_v(2)**2)
           r_tant = (r_v(3)/r_p)*r_q
           r_theta = atan(r_tant)
           r_tant = (r_v(3) + r_q3*r_b*sin(r_theta)**3)/
     +          (r_p - r_e2*r_a*cos(r_theta)**3)
           r_llh(1) =  atan(r_tant)
           r_re = r_a/sqrt(1.d0 - r_e2*sin(r_llh(1))**2)
           r_llh(3) = r_p/cos(r_llh(1)) - r_re          
  
        endif
      
        end  


c****************************************************************

	subroutine lincomb(r_k1,r_u,r_k2,r_v,r_w)

c****************************************************************
c**
c**	FILE NAME: lincomb.f
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: The subroutine forms the linear combination
c**	of two vectors.
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       	implicit none

c	INPUT VARIABLES:
        real*8 r_u(3)                              !3x1 vector
        real*8 r_v(3)                              !3x1 vector
        real*8 r_k1				 !scalar
        real*8 r_k2				 !scalar
   
c   	OUTPUT VARIABLES:
        real*8 r_w(3)                              !3x1 vector

c	LOCAL VARIABLES:none

c  	PROCESSING STEPS:

c       compute linear combination

	r_w(1) = r_k1*r_u(1) + r_k2*r_v(1)
	r_w(2) = r_k1*r_u(2) + r_k2*r_v(2)
	r_w(3) = r_k1*r_u(3) + r_k2*r_v(3)
      
        end  
	
 
c****************************************************************

	subroutine matvec(r_t,r_v,r_w)

c****************************************************************
c**
c**	FILE NAME: matvec.f
c**
c**     DATE WRITTEN: 7/20/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: The subroutine takes a 3x3 matrix 
c**     and a 3x1 vector a multiplies them to return another 3x1
c**	vector.
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       	implicit none

c	INPUT VARIABLES:
 	real*8 r_t(3,3)                            !3x3 matrix
        real*8 r_v(3)                              !3x1 vector
   
c   	OUTPUT VARIABLES:
        real*8 r_w(3)                              !3x1 vector

c	LOCAL VARIABLES:none

c  	PROCESSING STEPS:

c       compute matrix product

	r_w(1) = r_t(1,1)*r_v(1) + r_t(1,2)*r_v(2) + r_t(1,3)*r_v(3)
	r_w(2) = r_t(2,1)*r_v(1) + r_t(2,2)*r_v(2) + r_t(2,3)*r_v(3)
	r_w(3) = r_t(3,1)*r_v(1) + r_t(3,2)*r_v(2) + r_t(3,3)*r_v(3)
          
        end  
 

