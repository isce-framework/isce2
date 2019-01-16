c****************************************************************

	subroutine geo_hdg(r_a,r_e2,r_lati,r_loni,r_latf,r_lonf,r_geohdg)

c****************************************************************
c**
c**	FILE NAME: geo_hdg.f
c**
c**     DATE WRITTEN:12/02/93 
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: This routine computes the heading along a geodesic
c**     for either an ellipitical or spherical earth given the initial latitude
c**     and longitude and the final latitude and longitude. 
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: These results are based on the memo
c**
c**        "Summary of Mocomp Reference Line Determination Study" , IOM 3346-93-163
c**
c**      and the paper
c**
c**        "A Rigourous Non-iterative Procedure for Rapid Inverse Solution of Very
c**         Long Geodesics" by E. M. Sadano, Bulletine Geodesique 1958
c**
c**     ALL ANGLES ARE ASSUMED TO BE IN RADIANS!   
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       	implicit none

c	INPUT VARIABLES:
        real*8 r_a                    !semi-major axis
	real*8 r_e2                   !square of eccentricity
        real*8 r_lati                 !starting latitude
        real*8 r_loni                 !starting longitude
        real*8 r_latf                 !ending latitude
        real*8 r_lonf                 !ending longitude  
     
c   	OUTPUT VARIABLES:
        real*8 r_geohdg

c	LOCAL VARIABLES:
        real*8 pi,r_t1,r_t2,r_e,r_ome2,r_sqrtome2,r_b0,r_f,r_ep,r_n
        real*8 r_k1,r_k2,r_k3,r_k4,r_k5,r_l,r_ac,r_bc,r_phi,r_phi0
        real*8 r_tanbetai,r_cosbetai,r_sinbetai,r_cosphi,r_sinphi
        real*8 r_tanbetaf,r_cosbetaf,r_sinbetaf,r_lambda,r_coslam,r_sinlam
        real*8 r_ca,r_cb,r_cc,r_cd,r_ce,r_cf,r_cg,r_ch,r_ci,r_cj,r_x,r_q
        real*8 r_sinlati,r_coslati,r_tanlatf,r_tanlati,r_coslon,r_sinlon
        real*8 r_sin2phi,r_cosph0,r_sinph0,r_cosbeta0,r_cos2sig,r_cos4sig
        real*8 r_cotalpha12,r_cotalpha21,r_lsign 
        logical l_first

c	DATA STATEMENTS:
        data pi /3.141592653589793d0/
        data l_first /.true./ 

c       SAVE STATEMENTS: (needed on Freebie only)
        save l_first,r_e,r_ome2,r_sqrtome2,r_b0,r_f,r_ep
        save r_n,r_k1,r_k2,r_k3,r_k4,r_k5
 
c	FUNCTION STATEMENTS: none

c  	PROCESSING STEPS:

        if(r_e2 .eq. 0)then   !use the simplier spherical formula

	   r_sinlati = sin(r_lati)
	   r_coslati = cos(r_lati)
           r_tanlatf = tan(r_latf)

           r_t1 =  r_lonf - r_loni
	   if(abs(r_t1) .gt. pi)then
	      r_t1 = -(2.d0*pi - abs(r_t1))*sign(1.d0,r_t1)
           endif 
 
           r_sinlon = sin(r_t1)
           r_coslon = cos(r_t1)
           r_t2 = r_coslati*r_tanlatf - r_sinlati*r_coslon

           r_geohdg = atan2(r_sinlon,r_t2)

        else   ! use the full ellipsoid formulation

          if(l_first)then 
             l_first = .false.
	     r_e = sqrt(r_e2)
	     r_ome2 = 1.d0 - r_e2
	     r_sqrtome2 = sqrt(r_ome2)
             r_b0 = r_a*r_sqrtome2
	     r_f = 1.d0 - r_sqrtome2
	     r_ep = r_e*r_f/(r_e2-r_f)
	     r_n = r_f/r_e2
	     r_k1 = (16.d0*r_e2*r_n**2 + r_ep**2)/r_ep**2   
             r_k2 = (16.d0*r_e2*r_n**2)/(16.d0*r_e2*r_n**2 + r_ep**2)
             r_k3 = (16.d0*r_e2*r_n**2)/r_ep**2
             r_k4 = (16.d0*r_n - r_ep**2)/(16.d0*r_e2*r_n**2 + r_ep**2)
             r_k5 = 16.d0/(r_e2*(16.d0*r_e2*r_n**2 + r_ep**2))
          endif

          r_tanlati = tan(r_lati)
          r_tanlatf = tan(r_latf)
          r_l  =  abs(r_lonf-r_loni)
          r_lsign = r_lonf - r_loni
          if(abs(r_lsign) .gt. pi)then
	     r_lsign = -(2.d0*pi - r_l)*sign(1.d0,-r_lsign)
	  endif
          r_sinlon = sin(r_l)
          r_coslon = cos(r_l)
 
          r_tanbetai = r_sqrtome2*r_tanlati
          r_tanbetaf = r_sqrtome2*r_tanlatf

          r_cosbetai = 1.d0/sqrt(1.d0 + r_tanbetai**2)
          r_cosbetaf = 1.d0/sqrt(1.d0 + r_tanbetaf**2)
          r_sinbetai = r_tanbetai*r_cosbetai        
          r_sinbetaf = r_tanbetaf*r_cosbetaf

          r_ac = r_sinbetai*r_sinbetaf        
          r_bc = r_cosbetai*r_cosbetaf        
 
          r_cosphi = r_ac + r_bc*r_coslon
          r_sinphi = sign(1.d0,r_sinlon)*sqrt(1.d0 - min(r_cosphi**2,1.d0))
          r_phi = abs(atan2(r_sinphi,r_cosphi))
          
          if(r_a*abs(r_phi) .gt. 1.0d-6)then

	     r_ca = (r_bc*r_sinlon)/r_sinphi
	     r_cb = r_ca**2
	     r_cc = (r_cosphi*(1.d0 - r_cb))/r_k1
	     r_cd = (-2.d0*r_ac)/r_k1
	     r_ce = -r_ac*r_k2
	     r_cf = r_k3*r_cc
	     r_cg = r_phi**2/r_sinphi
	     
	     r_x = ((r_phi*(r_k4 + r_cb) + r_sinphi*(r_cc + r_cd) + r_cg*(r_cf + r_ce))*r_ca)/r_k5
	     
	     r_lambda = r_l + r_x
	     
	     r_sinlam = sin(r_lambda)
	     r_coslam = cos(r_lambda)
	     
	     r_cosph0 = r_ac + r_bc*r_coslam
	     r_sinph0 = sign(1.d0,r_sinlam)*sqrt(1.d0 - r_cosph0**2)
	     
	     r_phi0 = abs(atan2(r_sinph0,r_cosph0))
	     
	     r_sin2phi = 2.d0*r_sinph0*r_cosph0
	     
	     r_cosbeta0 = (r_bc*r_sinlam)/r_sinph0
	     r_q = 1.d0 - r_cosbeta0**2
	     r_cos2sig = (2.d0*r_ac - r_q*r_cosph0)/r_q
	     r_cos4sig = 2.d0*(r_cos2sig**2 - .5d0)
	     
	     r_ch = r_b0*(1.d0 + (r_q*r_ep**2)/4.d0 - (3.d0*(r_q**2)*r_ep**4)/64.d0)
	     r_ci = r_b0*((r_q*r_ep**2)/4.d0 - ((r_q**2)*r_ep**4)/16.d0)
	     r_cj = (r_q**2*r_b0*r_ep**4)/128.d0
	     
	     r_t2 = (r_tanbetaf*r_cosbetai - r_coslam*r_sinbetai)
	     r_sinlon = r_sinlam*sign(1.d0,r_lsign)
	     
	     r_cotalpha12 = (r_tanbetaf*r_cosbetai - r_coslam*r_sinbetai)/r_sinlam
	     r_cotalpha21 = (r_sinbetaf*r_coslam - r_cosbetaf*r_tanbetai)/r_sinlam
	     
	     r_geohdg = atan2(r_sinlon,r_t2)
	     
          else
	     
	     r_geohdg = 0.0d0
c             type*, 'Out to lunch...'
	     
          endif
 
	endif
       
        end  

