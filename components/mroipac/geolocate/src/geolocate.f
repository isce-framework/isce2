
!c23456789012345678901234567890123456789012345678901234567890123456789012
!c
!c compiled on moka with the command: /usr/bin/gfortran geolocate.f
!c
!c r_pos, r_range and r_a are in meters
!c r_vel is in m/s
!c r_squint is in radians and is for a right-looking SAR
!c r_llh consists of lat and lon in radians and height in meters
!c r_look_angle is in radians
!c
!c23456789012345678901234567890123456789012345678901234567890123456789012

      subroutine geolocate(r_pos, r_vel, r_range, r_squint, ip_side, r_a, r_e2,
     &                       r_llh, r_look_angle,r_incidence_angle)

      implicit none


ccccc declare parameters

      integer*4  i_schtoxyz
      parameter (i_schtoxyz = 0)

      integer*4  i_xyztollh
      parameter (i_xyztollh = 2)

      integer*4  i_xyztosch
      parameter (i_xyztosch = 1)


ccccc declare functions

      real*8 rdir


ccccc declare variables

      integer*4     i_i
      integer*4     i_side, ip_side ! sat lk direction (+1 right lk, -1 left lk)

      real*8 r_a
      real*8 r_a_dum
      real*8 r_b
      real*8 r_bias(3)
      real*8 r_cosg
      real*8 r_e2
      real*8 r_e2_dum
      real*8 r_enubias(3)
      real*8 r_enumat(3,3)
      real*8 r_enuvel(3)
      real*8 r_img_pln_rad
      real*8 r_incidence_angle
      real*8 r_lat
      real*8 r_lk_xyz(3)
      real*8 r_lk_xyz_mag
      real*8 r_llh(3)
      real*8 r_lon
      real*8 r_look_angle
      real*8 r_m
      real*8 r_mag
      real*8 r_pos(3)
      real*8 r_rad
      real*8 r_range
      real*8 r_sch2(3)
      real*8 r_sing
      real*8 r_sinm
      real*8 r_squint
      real*8 r_tanm
      real*8 r_target_d
      real*8 r_vel(3)
      real*8 r_xyz(3)

      real*8 sc_az_nom
      real*8 sc_d
      real*8 sc_h
      real*8 sc_hdg
      real*8 sc_lat
      real*8 sc_lon
      real*8 sc_r
      real*8 sc_sch(3)
      real*8 sc_vel
      real*8 u_lk(3)
      real*8 u_lk_xyz(3)
      real*8 u_n(3)
      real*8 xyz2enu(3,3)


ccccc declare derived data types

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

      type pegtrans
         sequence
         real (8) r_mat(3,3)
         real (8) r_matinv(3,3)
         real (8) r_ov(3)
         real (8) r_radcur
      end type pegtrans
      type (pegtrans) ptm2


ccccc common statements

c      r_a_dum = r_a
c      r_e2_dum = r_e2
 
      common /ellipsoid/ r_a_dum, r_e2_dum


ccccc data statements
      
      data r_bias /0.061d3,-0.285d3,-0.181d3/
      real*8, parameter :: r_dtor = atan(1.d0) / 45.d0

ccccc initialize

!!      i_side = +1                       ! right looking
      i_side = -1*ip_side                 !ISCE convention to code convention
      elp%r_a = r_a
      elp%r_e2 = r_e2
      r_a_dum = r_a
      r_e2_dum = r_e2
      sc_az_nom = dble(i_side) * (90.d0 * r_dtor - r_squint)
      r_b = sqrt(r_a**2 * (1.d0 - r_e2))

ccccc determine spacecraft info

      call norm(r_pos,sc_r)
      call norm(r_vel,sc_vel)
      call latlon(elp,r_pos,r_llh,i_xyztollh)
      sc_lat = r_llh(1)
      sc_lon = r_llh(2)
      sc_h = r_llh(3)
      call enubasis(sc_lat,sc_lon,r_enumat)
      call tranmat(r_enumat,xyz2enu)
      call matvec(r_enumat,r_bias,r_enubias)
      call matvec(xyz2enu,r_vel,r_enuvel)
      sc_hdg = atan2(r_enuvel(1),r_enuvel(2))


ccccc solve law of cosines to determine lk angle to reference ellipsoid

      peg%r_lat = sc_lat
      peg%r_lon = sc_lon
      peg%r_hdg = sc_hdg+sc_az_nom
      r_img_pln_rad = rdir(r_a,r_e2,sc_hdg+sc_az_nom,sc_lat)
      call radar_to_xyz(elp,peg,ptm2)
      call convert_sch_to_xyz(ptm2,sc_sch,r_pos,i_xyztosch)
      r_target_d = r_img_pln_rad
      sc_d = r_img_pln_rad + sc_sch(3)
      r_look_angle = acos((sc_d**2 + r_range**2 - r_target_d**2) /
     &                    (2.d0 * sc_d * r_range))


ccccc construct look vector (in SCH coord.) from computed look angle

      u_lk(1) = +sin(r_look_angle)
      u_lk(2) = 0.d0
      u_lk(3) = -cos(r_look_angle)


ccccc compute xyz vector from earth center to ellipsoid

      do i_i = 1 , 3
         r_xyz(i_i) = sc_sch(i_i) + u_lk(i_i) * r_range
      enddo
      r_m = sqrt(r_xyz(1)**2 + r_xyz(2)**2)
      r_tanm = r_m / (r_img_pln_rad+r_xyz(3))
      r_sinm = r_m / (r_m**2+(r_img_pln_rad+r_xyz(3))**2)
      r_cosg = r_xyz(1) / r_m
      r_sing = r_xyz(2) / r_m
      r_sch2(1) = r_img_pln_rad * atan(r_tanm * r_cosg)
      r_sch2(2) = r_img_pln_rad * asin(r_sinm * r_sing)
      r_sch2(3) = sqrt((r_img_pln_rad + r_xyz(3))**2 + r_m**2) -
     &                  r_img_pln_rad
      call convert_sch_to_xyz(ptm2,r_sch2,r_xyz,i_schtoxyz)
      call latlon(elp,r_pos,r_llh,i_xyztollh)
      r_lat = r_llh(1)
      r_lon = r_llh(2)
      r_rad = r_llh(3)


ccccc compute lat, lon and hgt

      call latlon(elp,r_xyz,r_llh,i_xyztollh)


ccccc compute ellipsoid outward unit surface normal

      r_mag = 2.d0 * sqrt((r_xyz(1)/r_a**2)**2 +
     &                    (r_xyz(2)/r_a**2)**2 + 
     &                    (r_xyz(3)/r_b**2)**2)
      u_n(1) = (2.d0 * r_xyz(1) / r_a**2) / r_mag
      u_n(2) = (2.d0 * r_xyz(2) / r_a**2) / r_mag
      u_n(3) = (2.d0 * r_xyz(3) / r_b**2) / r_mag
      
      
ccccc compute unit look vector in cartesian coordinates
      
      do i_i = 1 , 3
         r_lk_xyz(i_i) = r_xyz(i_i) - r_pos(i_i)
      enddo
      
      r_lk_xyz_mag = sqrt(r_lk_xyz(1)**2+r_lk_xyz(2)**2+r_lk_xyz(3)**2)
      
      do i_i = 1 , 3
         u_lk_xyz(i_i) = r_lk_xyz(i_i) / r_lk_xyz_mag
      enddo
      
      
ccccc compute incidence angle
      
      r_incidence_angle = acos(-u_n(1)*u_lk_xyz(1)-u_n(2)*u_lk_xyz(2)-
     &                          u_n(3)*u_lk_xyz(3))
      
      
      
      
ccccc write results

c      write (*,*) 'r_look_angle (deg): ' , r_look_angle / r_dtor
c      write (*,*)


      return

      end

c23456789012345678901234567890123456789012345678901234567890123456789012

c****************************************************************

        subroutine norm(r_v,r_n)

c****************************************************************
c**
c**     FILE NAME: norm.f
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c**     FUNCTIONAL DESCRIPTION: The subroutine takes vector and returns 
c**     its norm.
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

        implicit none

c       INPUT VARIABLES:
        real*8 r_v(3)                              !3x1 vector
   
c       OUTPUT VARIABLES:see input

c       LOCAL VARIABLES:
        real*8 r_n

c       PROCESSING STEPS:

c       compute vector norm

        r_n = sqrt(r_v(1)**2 + r_v(2)**2 + r_v(3)**2)
      
        end  

c23456789012345678901234567890123456789012345678901234567890123456789012

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
        integer i_type     !1=lat,lon to vector,2= vector to lat,lon
c       structure /ellipsoid/ 
c          real*8 r_a        
c          real*8 r_e2
c       end structure
c       record /ellipsoid/ elp
        
        type ellipsoid
           sequence
           real (8) r_a        
           real (8) r_e2
        end type ellipsoid
        type (ellipsoid) elp

        real*8 r_v(3)   !geocentric vector (meters)
        real*8 r_llh(3) !lat (deg -90 to 90),lon (deg -180 to 180),hgt
   
c       OUTPUT VARIABLES: see input

c       LOCAL VARIABLES:

        real*8 r_re,r_q2,r_q3,r_b,r_q
        real*8 r_p,r_tant,r_theta,r_a,r_e2

c       DATA STATEMENTS:

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

c23456789012345678901234567890123456789012345678901234567890123456789012

c****************************************************************

        subroutine enubasis(r_lat,r_lon,r_enumat)

c****************************************************************
c**
c**     FILE NAME: enubasis.f
c**
c**     DATE WRITTEN: 7/22/93
c**
c**     PROGRAMMER:Scott Hensley
c**
c**     FUNCTIONAL DESCRIPTION:Takes a lat and lon and returns a 
c**     change of basis matrix from ENU to geocentric coordinates.  
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c****************************************************************

        implicit none

c       INPUT VARIABLES:
        real*8 r_lat                   !latitude (deg)
        real*8 r_lon                   !longitude (deg)
   
c       OUTPUT VARIABLES:
        real*8 r_enumat(3,3)

c       LOCAL VARIABLES:
        real*8 r_slt,r_clt,r_clo,r_slo

c       DATA STATEMENTS:

C       FUNCTION STATEMENTS:

c       PROCESSING STEPS:

        r_clt = cos(r_lat)
        r_slt = sin(r_lat)
        r_clo = cos(r_lon)
        r_slo = sin(r_lon)

c     North  vector

        r_enumat(1,2) = -r_slt*r_clo        
        r_enumat(2,2) = -r_slt*r_slo
        r_enumat(3,2) = r_clt
      
c     East vector

        r_enumat(1,1) = -r_slo        
        r_enumat(2,1) = r_clo
        r_enumat(3,1) = 0.d0

c     Up vector 
      
        r_enumat(1,3) = r_clt*r_clo        
        r_enumat(2,3) = r_clt*r_slo
        r_enumat(3,3) = r_slt

        end  

c23456789012345678901234567890123456789012345678901234567890123456789012

c****************************************************************

        subroutine tranmat(r_a,r_b)

c****************************************************************
c**
c**     FILE NAME: tranmat.f
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c**     FUNCTIONAL DESCRIPTION: The subroutine takes a 3x3 matrix
c**     and computes its transpose.
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

        implicit none

c       INPUT VARIABLES:
        real*8 r_a(3,3)                      !3x3 matrix
   
c       OUTPUT VARIABLES:
        real*8 r_b(3,3)                      !3x3 matrix

c       LOCAL VARIABLES:
        integer i,j         

c       PROCESSING STEPS:

c       compute matrix product

        do i=1,3
           do j=1,3
             r_b(i,j) = r_a(j,i)
           enddo 
        enddo
          
        end  

c23456789012345678901234567890123456789012345678901234567890123456789012
 
c****************************************************************

        subroutine matvec(r_t,r_v,r_w)

c****************************************************************
c**
c**     FILE NAME: matvec.f
c**
c**     DATE WRITTEN: 7/20/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c**     FUNCTIONAL DESCRIPTION: The subroutine takes a 3x3 matrix 
c**     and a 3x1 vector a multiplies them to return another 3x1
c**     vector.
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

        implicit none

c       INPUT VARIABLES:
        real*8 r_t(3,3)                            !3x3 matrix
        real*8 r_v(3)                              !3x1 vector
   
c       OUTPUT VARIABLES:
        real*8 r_w(3)                              !3x1 vector

c       LOCAL VARIABLES:none

c       PROCESSING STEPS:

c       compute matrix product

        r_w(1) = r_t(1,1)*r_v(1) + r_t(1,2)*r_v(2) + r_t(1,3)*r_v(3)
        r_w(2) = r_t(2,1)*r_v(1) + r_t(2,2)*r_v(2) + r_t(2,3)*r_v(3)
        r_w(3) = r_t(3,1)*r_v(1) + r_t(3,2)*r_v(2) + r_t(3,3)*r_v(3)
          
        end  
 
c23456789012345678901234567890123456789012345678901234567890123456789012

c****************************************************************

        subroutine radar_to_xyz(elp,peg,ptm)

c****************************************************************
c**
c**     FILE NAME: radar_to_xyz.for
c**
c**     DATE WRITTEN:1/15/93 
c**
c**     PROGRAMMER:Scott Hensley
c**
c**     FUNCTIONAL DESCRIPTION:This routine computes the transformation
c**     matrix & translation vector needed to get between radar (s,c,h)
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

c       INPUT VARIABLES:

c       structure /ellipsoid/ 
c          real*8 r_a        
c          real*8 r_e2
c       end structure
c       record /ellipsoid/ elp

c       structure /peg/ 
c          real*8 r_lat
c          real*8 r_lon
c          real*8 r_hdg
c       end structure
c       record /peg/ peg

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
   
c       OUTPUT VARIABLES:

c       structure /pegtrans/ 
c          real*8 r_mat(3,3)
c          real*8 r_matinv(3,3)
c          real*8 r_ov(3)
c          real*8 r_radcur
c       end structure
c       record /pegtrans/ ptm

        type pegtrans
          sequence
          real (8) r_mat(3,3)
          real (8) r_matinv(3,3)
          real (8) r_ov(3)
          real (8) r_radcur
        end type pegtrans
        type (pegtrans) ptm

c       LOCAL VARIABLES:
        integer i,j,i_type
        real*8 r_llh(3),r_p(3),r_slt,r_clt,r_clo,r_slo
        real*8 r_up(3),r_chg,r_shg,rdir

c       DATA STATEMENTS:none

C       FUNCTION STATEMENTS:
        external rdir

c       PROCESSING STEPS:

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

c23456789012345678901234567890123456789012345678901234567890123456789012

c****************************************************************

        subroutine convert_sch_to_xyz(ptm,r_schv,r_xyzv,i_type)

c****************************************************************
c**
c**     FILE NAME: convert_sch_to_xyz.for
c**
c**     DATE WRITTEN:1/15/93 
c**
c**     PROGRAMMER:Scott Hensley
c**
c**     FUNCTIONAL DESCRIPTION: This routine applies the affine matrix 
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

c       INPUT VARIABLES:

c       structure /pegtrans/          !transformation parameters
c          real*8 r_mat(3,3)
c          real*8 r_matinv(3,3)
c          real*8 r_ov(3)
c          real*8 r_radcur
c       end structure
c       record /pegtrans/ ptm

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
   
c       OUTPUT VARIABLES: see input

c       LOCAL VARIABLES:
        integer i_t
        real*8 r_schvt(3),r_llh(3)
c       structure /ellipsoid/ 
c          real*8 r_a        
c          real*8 r_e2
c       end structure
c       record /ellipsoid/ sph

        type ellipsoid
           sequence
           real (8) r_a        
           real (8) r_e2
        end type ellipsoid
        type (ellipsoid) sph

c       DATA STATEMENTS:

C       FUNCTION STATEMENTS:none

c       PROCESSING STEPS:

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

c23456789012345678901234567890123456789012345678901234567890123456789012

c****************************************************************

        subroutine lincomb(r_k1,r_u,r_k2,r_v,r_w)

c****************************************************************
c**
c**     FILE NAME: lincomb.f
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c**     FUNCTIONAL DESCRIPTION: The subroutine forms the linear
c**     combination of two vectors.
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

        implicit none

c       INPUT VARIABLES:
        real*8 r_u(3)                              !3x1 vector
        real*8 r_v(3)                              !3x1 vector
        real*8 r_k1                              !scalar
        real*8 r_k2                              !scalar
   
c       OUTPUT VARIABLES:
        real*8 r_w(3)                              !3x1 vector

c       LOCAL VARIABLES:none

c       PROCESSING STEPS:

c       compute linear combination

        r_w(1) = r_k1*r_u(1) + r_k2*r_v(1)
        r_w(2) = r_k1*r_u(2) + r_k2*r_v(2)
        r_w(3) = r_k1*r_u(3) + r_k2*r_v(3)
      
        end  

c23456789012345678901234567890123456789012345678901234567890123456789012

c****************************************************************
c
c       Various curvature functions
c 
c
c****************************************************************
c**
c**     FILE NAME: curvature.f
c**
c**     DATE WRITTEN: 12/02/93
c**
c**     PROGRAMMER:Scott Hensley
c**
c**     FUNCTIONAL DESCRIPTION: This routine computes the curvature for 
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
        
        rnorth = (r_a*(1.d0 - r_e2))/(1.d0 - r_e2*sin(r_lat)**2)**1.5d0

        end

        real*8 function  rdir(r_a,r_e2,r_hdg,r_lat)

        implicit none
        real*8 r_a,r_e2,r_lat,r_hdg,r_re,r_rn,reast,rnorth
        
        r_re = reast(r_a,r_e2,r_lat)
        r_rn = rnorth(r_a,r_e2,r_lat)

        rdir = (r_re*r_rn)/(r_re*cos(r_hdg)**2 + r_rn*sin(r_hdg)**2) 

        end      

c23456789012345678901234567890123456789012345678901234567890123456789012

