c****************************************************************

	real*8 function fc(r_tsp)

       	implicit none

c	INPUT VARIABLES:
        real*8 r_tsp               !time since periapsis
	
c   	OUTPUT VARIABLES: none

c	LOCAL VARIABLES:
        real*8 r_sc_look, r_sc_az, r_az
	real*8 tpv(3)
        real*8 pos(3)
        real*8 vel(3)
        real*8 los_v(3)
	common /sc_point/ r_sc_look, r_sc_az
	common /target/ tpv
	
	
	real*8 rr
	real*8 pix, dop
	
	real*8 rd_ref, rsamp_dopp, wvl
	real*8 fd_coef_hertz(4)
	integer*4 i_type_fc           !0 for given angle angle, otherwise use dop_coef
	common /dopplercommon/ rd_ref, rsamp_dopp, wvl, fd_coef_hertz, i_type_fc

	if(i_type_fc .eq. 0) then
	  call orrmread1(16,r_tsp,pos,vel,0)
	  call lincomb(-1.d0,pos,1.d0,tpv,los_v)
	  call getangs(pos,vel,los_v,r_az,r_sc_look)
	  fc = r_az - r_sc_az
	  return
	else
	  call orrmread1(16,r_tsp,pos,vel,0)
	  call lincomb(-1.d0,pos,1.d0,tpv,los_v)
	  call norm(los_v,rr)
          pix = (rr-rd_ref)/(rsamp_dopp*1.d-3)
          dop = fd_coef_hertz(1) + fd_coef_hertz(2) * pix +
     $           fd_coef_hertz(3) * pix**2 + fd_coef_hertz(4) * pix**3
          fc =  dop - 1.d3 * ( vel(1) * los_v(1) 
     &	                       +vel(2) * los_v(2) 
     &	                       +vel(3) * los_v(3) ) / rr * 2. / ( wvl * 1000. )
          return
	endif
 	  

        end  

 
