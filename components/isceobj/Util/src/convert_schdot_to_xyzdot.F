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




