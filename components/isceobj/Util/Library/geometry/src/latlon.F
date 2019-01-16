
            subroutine latlon(elp,r_v,r_llh,i_type)BIND(C,NAME='latlon_C') 

!c****************************************************************
!c**   
!c**   FILE NAME: latlon.f
!c**   
!c**   DATE WRITTEN:7/22/93 
!c**   
!c**   PROGRAMMER:Scott Hensley
!c**   
!c**   FUNCTIONAL DESCRIPTION:This program converts a vector to 
!c**   lat,lon and height above the reference ellipsoid or given a
!c**   lat,lon and height produces a geocentric vector. 
!c**   
!c**   ROUTINES CALLED:none
!c**   
!c**   NOTES: none
!c**   
!c**   UPDATE LOG:
!c**   
!c****************************************************************
                use, intrinsic :: iso_c_binding
                implicit none
        
!c       INPUT VARIABLES:
                integer(C_INT), value :: i_type    !1=lat,lon to vector,2= vector to lat,lon
                type (ellipsoidType) elp

                real(C_DOUBLE), dimension(3) :: r_v     !geocentric vector (meters)
                real(C_DOUBLE), dimension(3) :: r_llh   !latitude (deg -90 to 90),longitude (deg -180 to 180),height
   

!c       LOCAL VARIABLES:
                real*8 r_re,r_q2,r_q3,r_b,r_q
                real*8 r_p,r_tant,r_theta,r_a,r_e2,r_e4
                real*8 r_k,r_r,r_s,r_t,r_u,r_rv,r_w,r_d

!c       PROCESSING STEPS:

                r_a = elp%r_a
                r_e2 = elp%r_e2

                if(i_type .eq. LLH_2_XYZ)then  !convert lat,lon to vector
           
                    r_re = r_a/sqrt(1.d0 - r_e2*sin(r_llh(1))**2)
                    r_v(1) = (r_re + r_llh(3))*cos(r_llh(1))*cos(r_llh(2))
                    r_v(2) = (r_re + r_llh(3))*cos(r_llh(1))*sin(r_llh(2))
                    r_v(3) = (r_re*(1.d0-r_e2) + r_llh(3))*sin(r_llh(1))               
           
                elseif(i_type .eq. XYZ_2_LLH) then !convert vector to lat, lon
                    !!!Translated from python code in
                    !!!isceobj.Ellipsoid.xyz_to_llh
                    r_q2 = (r_v(1)**2 + r_v(2)**2)  !!xy2
                    r_q3 = r_a * r_a   !!a2
                    r_e4 = r_e2 * r_e2 

                    r_p = r_q2 / r_q3
                    r_q = (1.0d0 - r_e2)*(r_v(3)**2)/ r_q3
                    r_r = (r_p+r_q-r_e4)/6.0d0
                    r_s = (r_e4*r_p*r_q)/(4.0d0 * r_r**3)
                    r_t = (1.0d0 + r_s + sqrt(r_s *(2.0d0+ r_s)))**(1.0d0/3.0d0)
                    r_u = r_r * (1.0d0 + r_t + 1.0d0 / r_t)
                    r_rv = sqrt(r_u**2 + r_e4*r_q)
                    r_w = r_e2 * (r_u + r_rv - r_q)/(2.0d0 * r_rv)
                    r_k = sqrt(r_u + r_rv + r_w**2) - r_w
                    r_d = r_k * sqrt(r_q2) / (r_k + r_e2) 
                   
                    r_llh(1) = atan2(r_v(3), r_d)
                    r_llh(2) = atan2(r_v(2),r_v(1))
                    r_llh(3) = (r_k + r_e2 - 1.0d0) * sqrt(r_d**2 + r_v(3)**2)/r_k

                elseif(i_type .eq. XYZ_2_LLH_OLD)then  !convert vector to lat,lon 
           
                    r_q2 = 1.d0/(1.d0 - r_e2)
                    r_q = sqrt(r_q2)
                    r_q3 = r_q2 - 1.d0
                    r_b = r_a*sqrt(1.d0 - r_e2)
           
                    r_llh(2) = atan2(r_v(2),r_v(1))
           
                    r_p = sqrt(r_v(1)**2 + r_v(2)**2)
                    r_tant = (r_v(3)/r_p)*r_q
                    r_theta = atan(r_tant)
                    r_tant = (r_v(3) + r_q3*r_b*sin(r_theta)**3)/
     +                  (r_p - r_e2*r_a*cos(r_theta)**3)
                    r_llh(1) =  atan(r_tant)
                    r_re = r_a/sqrt(1.d0 - r_e2*sin(r_llh(1))**2)
                    r_llh(3) = r_p/cos(r_llh(1)) - r_re
  
                endif
      
            end subroutine latlon

