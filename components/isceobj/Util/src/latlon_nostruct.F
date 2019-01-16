c****************************************************************

        subroutine latlon_elp(r_a,r_e2,r_v,r_lat,r_lon,r_h,i_type) 

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
        
c     INPUT VARIABLES:
        integer i_type                   !1=lat,lon to vector,2= vector to lat,lon
        real*8 r_a                       !ellispoid semi-major axis
        real*8 r_e2                      !ellipsoid eccentricity squared  
        real*8 r_v(3)                    !geocentric vector (meters)
        real*8 r_lat                     !latitude (deg -90 to 90)
        real*8 r_lon                     !longitude (deg -180 to 180)
        real*8 r_h                       !height above ellipsoid (meters)
   
c       OUTPUT VARIABLES:see input

c       LOCAL VARIABLES:
        integer i_ft
        real*8 pi,r_dtor,r_re,r_q2,r_q3,r_b,r_q
        real*8 r_p,r_tant,r_theta

c       DATA STATEMENTS:
        data pi /3.141592653589793238d0/
        data r_dtor /1.74532925199d-2/
        data i_ft /0/

C       FUNCTION STATEMENTS:

c       PROCESSING STEPS:

        if(i_type .eq. 1)then  !convert lat,lon to vector
           
           r_re = r_a/sqrt(1.d0 - r_e2*sin(r_lat)**2)
           
           r_v(1) = (r_re + r_h)*cos(r_lat)*cos(r_lon)
           r_v(2) = (r_re + r_h)*cos(r_lat)*sin(r_lon)
           r_v(3) = (r_re*(1.d0-r_e2) + r_h)*sin(r_lat)               
           
        elseif(i_type .eq. 2)then  !convert vector to lat,lon 
           
           r_q2 = 1.d0/(1.d0 - r_e2)
           r_q = sqrt(r_q2)
           r_q3 = r_q2 - 1.d0
           r_b = r_a*sqrt(1.d0 - r_e2)
           
           r_lon = atan2(r_v(2),r_v(1))
           
           r_p = sqrt(r_v(1)**2 + r_v(2)**2)
           r_tant = (r_v(3)/r_p)*r_q
           r_theta = atan(r_tant)
           r_tant = (r_v(3) + r_q3*r_b*sin(r_theta)**3)/
     +          (r_p - r_e2*r_a*cos(r_theta)**3)
           r_lat =  atan(r_tant)
           r_re = r_a/sqrt(1.d0 - r_e2*sin(r_lat)**2)
           r_h = r_p/cos(r_lat) - r_re          
  
        endif
      
        end  


