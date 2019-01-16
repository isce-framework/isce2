c****************************************************************

        subroutine utmtoll(elp,i_zone,a_grid,r_v,r_lat,r_lon,i_type) 

c****************************************************************
c**   
c**   FILE NAME: utmtoll.f
c**   
c**   DATE WRITTEN:7/22/93 
c**   
c**   PROGRAMMER:Scott Hensley
c**   
c**   FUNCTIONAL DESCRIPTION: This routine converts between lat
c**   lon and utm coordinates for a datum determined from the input 
c**   a and e2.
c**   
c**   ROUTINES CALLED:none
c**   
c**   NOTES: none
c**   
c**   UPDATE LOG:
c**   
c****************************************************************
        
        implicit none
        
c     INPUT VARIABLES:

	type ellipsoid
           sequence
	   real (8) r_a        
	   real (8) r_e2
	end type ellipsoid
	type (ellipsoid) elp

        integer i_type                   !1=lat,lon to utm,2= utm to lat,lon
        real*8 r_a                       !ellispoid semi-major axis
        real*8 r_e2                      !ellipsoid eccentricity squared  
        real*8 r_v(2)                    !Easting , Northing
        real*8 r_lat                     !latitude (deg -90 to 90)
        real*8 r_lon                     !longitude (deg -180 to 180)
        integer i_zone                   !UTM zone
        character*1 a_grid               !UTM North-South grid
   
c       OUTPUT VARIABLES:see input

c       LOCAL VARIABLES:

        integer i_ft,i_gi,i_zoneu
        real*8 pi,r_dtor
        real*8 r_ep2,r_k0,r_k
        real*8 r_fe,r_fn(2)
        real*8 r_e4,r_e6,r_n,r_t,r_t2,r_c,r_c2,r_ba
        real*8 r_a2,r_a3,r_a4,r_a5,r_a6 
        real*8 r_d,r_d2,r_d3,r_d4,r_d5,r_d6
        real*8 r_lon0,r_lat1,r_m,r_m0,r_mu,r_lat0
        real*8 r_et,r_e1,r_e12,r_e13,r_e14,r_r
        character*1 a_griddes(20)

c       DATA STATEMENTS:

        data pi /3.141592653589793238d0/
        data r_dtor /1.74532925199d-2/
        data i_ft /0/
        data a_griddes /'C','D','E','F','G','H','J',
     +       'K','L','M','N','P','Q','R','S','T','U',
     +       'V','W','X'/
        data r_k0 /.9996d0/    !scale at center 
        data r_lat0 /0.d0/
        data r_fe,r_fn /500000.d0,0.d0,10000000.d0/

C       FUNCTION STATEMENTS:none

c       PROCESSING STEPS:

        r_a = elp%r_a
        r_e2 = elp%r_e2

        r_ep2 = r_e2/(1.d0 - r_e2)
        r_e4 = r_e2**2
        r_e6 = r_e2**3
        pi =  4.d0*atan(1.d0)
        r_dtor = pi/180.d0

        if(i_type .eq. 1)then  !convert lat,lon to UTM

           if(i_zone .ge. 0)then
              i_zone = int(mod(r_lon+3.d0*pi,2.d0*pi)/(r_dtor*6.d0)) + 1             
              i_zone = max(min(i_zone,60),1)
              i_zoneu = i_zone
           else
              i_zoneu = -i_zone
           endif

           r_lon0 = -pi + 6.d0*r_dtor*(i_zoneu-1) + 3.d0*r_dtor
           
           r_n = r_a/sqrt(1.d0 - r_e2*sin(r_lat)**2)
           r_t = tan(r_lat)**2
           r_t2 = r_t**2
           r_c = r_ep2*cos(r_lat)**2
           r_ba = (r_lon - r_lon0)*cos(r_lat)
           r_a2 = r_ba**2
           r_a3 = r_ba*r_a2 
           r_a4 = r_ba*r_a3
           r_a5 = r_ba*r_a4
           r_a6 = r_ba*r_a5
           r_m = r_a*((1.d0-r_e2/4 - 3.d0*r_e4/64.d0 - 
     +      5.d0*r_e6/256.d0)*r_lat - (3.d0*r_e2/8.d0 + 3.d0*r_e4/32.d0 + 
     +      45.d0*r_e6/1024.d0)*sin(2.d0*r_lat) +  (15.d0*r_e4/256.d0 + 
     +      45.d0*r_e6/1024.d0)*sin(4.d0*r_lat) - (35.d0*r_e6/3072.d0)*
     +      sin(6.d0*r_lat))
           r_m0 = r_a*((1.d0-r_e2/4 - 3.d0*r_e4/64.d0 - 
     +      5.d0*r_e6/256.d0)*r_lat0 - (3.d0*r_e2/8.d0 + 3.d0*r_e4/32.d0 + 
     +      45.d0*r_e6/1024.d0)*sin(2.d0*r_lat0) +  (15.d0*r_e4/256.d0 + 
     +      45.d0*r_e6/1024.d0)*sin(4.d0*r_lat0) - (35.d0*r_e6/3072.d0)*
     +      sin(6.d0*r_lat0))
           
           r_v(1) = r_k0*r_n*(r_ba+(1.d0-r_t+r_c)*r_a3/6.d0 + 
     +       (5.d0-18.d0*r_t+r_t2+72.d0*r_c-58.d0*r_ep2)*r_a5/120.d0)
           r_v(1) = r_v(1) + r_fe

           r_v(2) = r_k0*(r_m - r_m0 + r_n*tan(r_lat)*
     +       ( r_a2/2.d0 + (5.d0-r_t+9.d0*r_c+4.d0*r_c**2)*
     +        (r_a4/24.d0) + (61.d0-58.d0*r_t+r_t2+600.d0*r_c-
     +        330.d0*r_ep2)*(r_a6/720.d0) ))

           if(r_lat .ge. 0)then
              r_v(2) = r_v(2) + r_fn(1)
           else
              if(a_grid .eq. 'A')then
                 r_v(2) = r_v(2) 
              elseif(a_grid .eq. 'Z')then
                 r_v(2) = r_v(2) + 2.d0*r_fn(2)
              else
                 r_v(2) = r_v(2) + r_fn(2)
              endif
           endif

           r_k = r_k0*(1.d0+(1.d0+r_ep2*cos(r_lat)**2)*(r_v(1)-r_fe)**2/
     +          (2.d0*(r_k0**2)*r_n**2))

           i_gi = int((r_lat/r_dtor+80.d0)/8.d0) + 1
           i_gi = max(min(i_gi,20),1)
           a_grid = a_griddes(i_gi)
           
        elseif(i_type .eq. 2)then  !convert UTM to lat,lon 

           r_v(1) = r_v(1) - r_fe

           if(a_grid .eq. 'A')then
              r_v(2) = r_v(2) 
           elseif(a_grid .eq. 'Z')then
              if(r_v(2) .ge. r_fn(2))then
                 r_v(2) = r_v(2) - 2.d0*r_fn(2)  
              endif
           elseif(ichar(a_grid) .ge. ichar('C') .and. ichar(a_grid) .le. ichar('X'))then
              if(ichar(a_grid) .le. ichar('M'))then
                 r_v(2) = r_v(2) - r_fn(2)
              endif
           else
              r_v(2) = r_v(2)                 !assume Northern hemisphere
           endif

           r_lon0 = -pi + 6.d0*r_dtor*(i_zone-1) + 3.d0*r_dtor
           
           r_et = sqrt(1.d0-r_e2)
           r_e1 = (1.d0-r_et)/(1.d0+r_et)
           r_e12 = r_e1**2
           r_e13 = r_e1*r_e12
           r_e14 = r_e1*r_e13
           r_m = r_v(2)/r_k0
           r_mu = r_m/(r_a*(1.d0-r_e2/4.d0-3.d0*r_e4/64.d0-
     +          5.d0*r_e6/256.d0))
           r_lat1 = r_mu + (3.d0*r_e1/2.d0-27.d0*r_e13/32.d0)*sin(2.d0*r_mu)+
     +          (21.d0*r_e12/16.d0-55.d0*r_e14/32.d0)*sin(4.d0*r_mu) +  
     +          (151.d0*r_e13/96.d0)*sin(6.d0*r_mu) +  
     +          (1097.d0*r_e14/512.d0)*sin(8.d0*r_mu) 

           r_n = r_a/sqrt(1.d0 - r_e2*sin(r_lat1)**2)
           r_r = (r_a*(1.d0-r_e2))/sqrt(1.d0 - r_e2*sin(r_lat1)**2)**3
           r_t = tan(r_lat1)**2
           r_t2 = r_t**2
           r_c = r_ep2*cos(r_lat1)**2
           r_c2 = r_c**2
           r_d = r_v(1)/(r_n*r_k0)
           r_d2 = r_d**2
           r_d3 = r_d2*r_d
           r_d4 = r_d3*r_d
           r_d5 = r_d4*r_d
           r_d6 = r_d5*r_d
 
           r_lat = r_lat1 - (r_n*tan(r_lat1)/r_r)*(r_d2/2.d0 -
     +       (5.d0+3.d0*r_t+10.d0*r_c-4.d0*r_c2-9.d0*r_ep2)*r_d4/24.d0 +
     +       (61.d0+90*r_t+298.d0*r_c+45.d0*r_t2-252.d0*r_ep2-3.d0*r_c2)*
     +       (r_d6/720.d0))
           r_lon = r_lon0 + (r_d - (1.d0+2.d0*r_t+r_c)*r_d3/6.d0 + 
     +       (5.d0-2.d0*r_c+28.d0*r_t-3.d0*r_c2+8.d0*r_ep2+24.d0*r_t2)*
     +       (r_d5/120.d0))/cos(r_lat1)

        endif
      
        end  

