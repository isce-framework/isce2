c****************************************************************

      subroutine get_peg_info

c****************************************************************
c**     
c**   FILE NAME: get_peg_info.f
c**     
c**   DATE WRITTEN: 6/10/95 
c**     
c**   PROGRAMMER: Scott Hensley
c**     
c**   FUNCTIONAL DESCRIPTION: This program reads simple emphemeris
c**   information and compute the appropriate peg frame as well as 
c**   generating 
c**     
c**   ROUTINES CALLED:none
c**     
c**   NOTES: none
c**     
c**   UPDATE LOG:
c**     changed ERR= to END= for orbit reads and decremented count EJF 2001/1/18
c*****************************************************************
      use get_peg_infoState
      implicit none

c     PARAMETER STATEMENTS:
      
      character*20000 MESSAGE
      real*8 r_awgs84,r_e2wgs84
      parameter(r_awgs84=6378137.d0,r_e2wgs84=.00669437999015d0)
      real*8 pi,r_dtor,r_rtod
      parameter(pi=3.141592653589793d0)     !if you have to ask, give it up
      parameter(r_rtod=180.d0/pi,r_dtor=pi/180.d0)  !radian to degree conversions

      integer i_xyztollh,i_llhtoxyz
      parameter(i_xyztollh=2, i_llhtoxyz=1) 
      integer i_schtoxyz,i_xyztosch
      parameter(i_schtoxyz=0,i_xyztosch=1) 

      integer i_rdf,i_file
      parameter(i_rdf=1,i_file=0)

      integer MAXOBS
      parameter(MAXOBS=20000)

      integer i_orbitnum
      parameter(i_orbitnum=75) 

c     INPUT VARIABLES:
	
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i,j,k
      integer i_nd,i_ma,i_list(3)
      real*8 r_time_scene_cen,r_dels
      real*8 r_schvec(3),r_xyzvec(3),r_x(10),r_hfit(10),r_t,r_xyzvel(3)
      real*8 r_xyzdot(3), r_schdot(3)
      real*8 r_hdotfit(10),r_sdotfit(10),r_cdotfit(10),r_hffdot,r_cffdot,r_sffdot
      real*8 r_cov(3,3),r_sig(10),r_chisq,r_hf,r_cff,r_cfit(10)
      real*8 vertfit(3), horizfit(3), vertvfit(2), horizvfit(2)
      real*8 r_earthgm,  r_earthspindot
      real*8 r_spinvec(3)

      type ellipsoid
         sequence
         real*8 r_a        
         real*8 r_e2
      end type ellipsoid
      type (ellipsoid) elp

      type peg_struct 
         sequence
         real*8 r_lat
         real*8 r_lon
         real*8 r_hdg
      end type peg_struct
      type (peg_struct) peg

      type pegtrans
         sequence
         real*8 r_mat(3,3)
         real*8 r_matinv(3,3)
         real*8 r_ov(3)
         real*8 r_radcur
      end type pegtrans
      type (pegtrans) ptm

      real*8 r_enumat(3,3),r_xyzenumat(3,3),r_enuvel(3)
      real*8 r_xyzpeg(3),r_llhpeg(3)
      real*8 r_tempv(3), r_tempa(3)
      real*8 r_tempvec(3), r_inertialacc(3), r_bodyacc(3)
      real*8 r_xyznorm, r_platsch(3)
      real*8 r_smin(2),r_smax(2),r_sref
      real*8 r_schvec1(3),r_xyzschmat(3,3),r_schxyzmat(3,3)
      real*8 r_xyzvec1(3),r_velnorm,r_delsint,r_scale
      real*8 r_endtimeslc,r_xyzvec11(3),r_schvec11(3)

      real*8, allocatable, dimension(:,:) :: r_llh1,r_sch1
      real*8, allocatable, dimension(:) :: r_hdg1,r_s1


c     OUTPUT VARIABLES:

c     DATA STATEMENTS:


      data r_earthspindot /7.29211573052d-5/
      data r_earthgm /3.98600448073d14/

c     COMMON BLOCKS:

c     SAVE STATEMENTS:

C     FUNCTION STATEMENTS:

      !Allocate the array that use to have MAXOBS size. Now use the actual value i_numobs
      allocate(r_llh1(3,i_numobs)) 
      allocate(r_sch1(3,i_numobs)) 
      allocate(r_hdg1(i_numobs)) 
      allocate(r_s1(i_numobs)) 

c     PROCESSING STEPS:

      
      elp%r_a = r_awgs84
      elp%r_e2 = r_e2wgs84      


c     Convert the position data to lat,lon and find the heading for each point 


c      write(6,*) ' '
      
      call writeStdOut('Transforming data orbit')
      do k=1,i_numobs

c     convert to lat,lon

         call latlon(elp,r_xyz1(1,k),r_llh1(1,k),i_xyztollh)

c     convert velocity to ENU frame

         call enubasis(r_llh1(1,k),r_llh1(2,k),r_enumat)
         call tranmat(r_enumat,r_xyzenumat)

c     determine the heading

         call matvec(r_xyzenumat,r_vxyz1(1,k),r_enuvel)
         r_hdg1(k) = atan2(r_enuvel(1),r_enuvel(2))
         write(MESSAGE,'(a,1x,i5)') 'Observation #: ',k
         call writeStdOut(MESSAGE)
         write(MESSAGE,'(a,1x,f10.5,1x,f10.5,1x,f12.3)') 'Lat, Lon & Height: ',r_llh1(1,k)*r_rtod,r_llh1(2,k)*r_rtod,r_llh1(3,k)
         call writeStdOut(MESSAGE)
	 write(MESSAGE,'(a,1x,f15.7)') 'Heading: ',r_hdg1(k)*r_rtod
         call writeStdOut(MESSAGE)
      enddo    !observations 


c     determine the peg lat,lon and heading to use - algorithm assumes a frame size scene and uses lat,lon at
c     scene center and heading a scene center for peg coordinates. Orbit 1 is the master frame and peg is determined
c     using it's orbit only. It is rough but should suffice for most applications.

      r_time_first_line = r_timeslc + (i_startline-1)/r_prf               !time to first line in Interferogram
      r_time_scene_cen = r_time_first_line + (i_numlines*i_looksaz)/(2.d0*r_prf)

      call writeStdOut('      << Output Data >> ')
      write(MESSAGE,'(a,1x,2(f12.3,1x))') 'Time to first/middle scene: ',r_time_first_line,
     +     r_time_scene_cen
      call writeStdOut(MESSAGE)

c     interpolate the motion data to the scene center using a quadratic interpolator

      call inter_motion(r_time,r_xyz1,i_numobs,r_time_scene_cen,r_xyzpeg)
      call inter_motion(r_time,r_vxyz1,i_numobs,r_time_scene_cen,r_vxyzpeg)
      write(MESSAGE,'(a,1x,3(f12.3,1x))') 'Pos Peg = ',(r_xyzpeg(j),j=1,3)
      call writeStdOut(MESSAGE)
      write(MESSAGE,'(a,1x,3(f12.6,1x))') 'Vel Peg = ',(r_vxyzpeg(j),j=1,3)
      call writeStdOut(MESSAGE)
      call norm(r_vxyzpeg,r_velnorm)

c     take the lat,lon as the peg point and the heading as the peg heading 

      call latlon(elp,r_xyzpeg,r_llhpeg,i_xyztollh)
      call enubasis(r_llhpeg(1),r_llhpeg(2),r_enumat)
      call tranmat(r_enumat,r_xyzenumat)
      call matvec(r_xyzenumat,r_vxyzpeg,r_enuvel)
      peg%r_hdg = atan2(r_enuvel(1),r_enuvel(2))      

      peg%r_lat = r_llhpeg(1)
      peg%r_lon = r_llhpeg(2) 

      call radar_to_xyz(elp,peg,ptm)

      r_pegLat = peg%r_lat*r_rtod
      r_pegLon = peg%r_lon*r_rtod
      r_pegHgt = r_llhpeg(3)
      r_pegHead = peg%r_hdg*r_rtod
      write(MESSAGE,'(a,1x,f12.7,1x,f12.7,1x,f12.3)') 'Peg Lat/Lon , H = ',
     +     peg%r_lat*r_rtod,peg%r_lon*r_rtod,r_llhpeg(3)
      call writeStdOut(MESSAGE)
      write(MESSAGE,'(a,1x,f15.7)') 'Peg Heading = ',peg%r_hdg*r_rtod
      call writeStdOut(MESSAGE)
      write(MESSAGE,'(a,1x,f15.5)') 'Radius Curvature = ',ptm%r_radcur
      call writeStdOut(MESSAGE)
      
      call writeStdOut('Rotation matrix ')
      write(MESSAGE,905) ' First row =  ',ptm%r_mat(1,1),ptm%r_mat(1,2),ptm%r_mat(1,3)
 905  format(a,1x,3(f12.9,1x))
      call writeStdOut(MESSAGE)
      write(MESSAGE,905) ' Second row = ',ptm%r_mat(2,1),ptm%r_mat(2,2),ptm%r_mat(2,3)
      call writeStdOut(MESSAGE)
      write(MESSAGE,905) ' Third row =  ',ptm%r_mat(3,1),ptm%r_mat(3,2),ptm%r_mat(3,3)
      call writeStdOut(MESSAGE)
      call writeStdOut('Translation vector ')
      write(MESSAGE,906) ' Vector = ',ptm%r_ov
 906  format(a,1x,3(f14.5,1x))
      call writeStdOut(MESSAGE)

      r_spinvec(1) = 0.
      r_spinvec(2) = 0.
      r_spinvec(3) = r_spindot

      call norm(r_xyzpeg,r_xyznorm)

      call cross(r_spinvec,r_xyzpeg,r_tempv)
      
      do k=1,3
         r_inertialacc(k) = -(r_gm*r_xyzpeg(k))/r_xyznorm**3
      enddo

      call cross(r_spinvec,r_vxyzpeg,r_tempa)
      call cross(r_spinvec,r_tempv,r_tempvec)
      
      do k=1,3
         r_bodyacc(k) = r_inertialacc(k) - 2.d0*r_tempa(k) - r_tempvec(k)
      enddo

c     convert back to a local SCH basis
      
      call convert_sch_to_xyz(ptm,r_platsch,r_xyzpeg,i_xyztosch)
      call schbasis(ptm,r_platsch,r_xyzschmat,r_schxyzmat)
      call matvec(r_xyzschmat,r_bodyacc,r_platacc)
      call matvec(r_xyzschmat,r_vxyzpeg,r_platvel)

      write(MESSAGE,'(a,x,3(f15.7,x))') 'Platform SCH Velocity (m/s): ',r_platvel
      call writeStdOut(MESSAGE)
      write(MESSAGE,'(a,x,3(f15.7,x))') 'Platform SCH Acceleration (m/s^2): ',r_platacc
      call writeStdOut(MESSAGE)


c     compute delta S on ground and in Orbit for SLC and Interferogram

      r_dels = r_platvel(1)/r_prf
      r_scale = ptm%r_radcur/(r_llhpeg(3) + ptm%r_radcur)

      r_delsint = r_dels*i_looksaz

      call writeStdOut('                       SLC        Interferogram')
      write(MESSAGE,'(a,1x,f10.5,1x,f10.5)') 'Delta S on Ground: ',r_dels*r_scale,r_delsint*r_scale
      call writeStdOut(MESSAGE)
      write(MESSAGE,'(a,1x,f10.5,1x,f10.5)') 'Delta S in Orbit:  ',r_dels,r_delsint
      call writeStdOut(MESSAGE)

c     convert the motion data to SCH coordinates

         
      call writeStdOut('Transforming data orbit: ')
      call writeStdOut('SCH positions ')
         
      r_smin(i) = 1.d25
      r_smax(i) = -1.d25

      do k=1,i_numobs

         call convert_sch_to_xyz(ptm,r_sch1(1,k),r_xyz1(1,k),i_xyztosch)
         write(MESSAGE,'(a,1x,3(f15.3,1x))') 'SCH : ',r_sch1(1,k),r_sch1(2,k),r_sch1(3,k)
         call writeStdOut(MESSAGE)
         r_smin(1) = min(r_smin(1),r_sch1(1,k))
         r_smax(1) = max(r_smax(1),r_sch1(1,k))
         r_s1(k) = r_sch1(1,k)
            
      enddo  !observations
 

c     compute the starting S coordinate for two scenes - and ending S coordinates

      r_endtimeslc = r_timeslc + i_slclines/r_prf
      call inter_motion(r_time,r_xyz1,i_numobs,r_timeslc,r_xyzvec1)
      call convert_sch_to_xyz(ptm,r_schvec1,r_xyzvec1,i_xyztosch)
      call inter_motion(r_time,r_xyz1,i_numobs,r_endtimeslc,r_xyzvec11)
      call convert_sch_to_xyz(ptm,r_schvec11,r_xyzvec11,i_xyztosch)

      r_sref = r_schvec1(1)

c     write out region of intersection if two orbits, and min,max x ccordinates

      write(MESSAGE,'(a,1x,f15.3,1x,f15.3)') 'Min, Max S for orbit 1:    ',r_smin(1),r_smax(1)
      call writeStdOut(MESSAGE)
      write(MESSAGE,'(a,1x,f15.3,1x,f15.3)') 'Min, Max S for orbit 1 II: ',r_schvec1(1),r_schvec11(1)
      call writeStdOut(MESSAGE)
      write(MESSAGE,'(a,1x,f15.3)') 'Reference S for fits: ',r_sref
      call writeStdOut(MESSAGE)
c     fit the height data to a quadratic for use in inverse3d

      call writeStdOut(' SCH Positions for 10 points along track ')

      do i=1,10

         r_t = r_time_first_line + (i_looksaz/r_prf)*((float(i_numlines)/9.d0)*(i-1))
c         r_x(i) = ((float(i_numlines)/(9.d0*i_looksaz))*(i-1) - 
c     +        float(i_numlines)/(2.d0*i_looksaz))*(r_dels*i_looksaz)
         call inter_motion(r_time,r_xyz1,i_numobs,r_t,r_xyzvec)
         call inter_motion(r_time,r_vxyz1,i_numobs,r_t,r_xyzdot)
         call convert_sch_to_xyz(ptm,r_schvec,r_xyzvec,i_xyztosch)
         call convert_schdot_to_xyzdot(ptm,r_schvec,r_xyzvec,r_schdot,r_xyzdot,i_xyztosch)

         r_hfit(i) = r_schvec(3)
         r_cfit(i) = r_schvec(2)

         r_hdotfit(i) = r_schdot(3)
         r_cdotfit(i) = r_schdot(2)
         r_sdotfit(i) = r_schdot(1)

         r_x(i) = r_schvec(1) - r_sref
         r_sig(i) = 1.d0

         write(MESSAGE,'(a,1x,f10.2,1x,3(f12.3,1x))') 'Time/Pos: ',r_t,r_schvec
         call writeStdOut(MESSAGE)

      enddo

c     fit orbit one C,H values to a quadratic

      i_nd = 10
      i_ma = 3
      i_list(1) = 1
      i_list(2) = 2
      i_list(3) = 3
      call lfit(r_x,r_hfit,r_sig,i_nd,r_af,i_ma,i_list,i_ma,r_cov,i_ma,r_chisq)
      call lfit(r_x,r_cfit,r_sig,i_nd,r_cf,i_ma,i_list,i_ma,r_cov,i_ma,r_chisq)
      vertfit = r_af
      horizfit = r_cf
      call writeStdOut(' * Quadratic Fit Coefficients for Height/Cross Track * ')
      write(MESSAGE,'(a,1x,3(e20.10,1x))') 'Vertical Fit:   ',r_af
      call writeStdOut(MESSAGE)
      write(MESSAGE,'(a,1x,3(e20.10,1x))') 'Horizontal Fit: ',r_cf
      call writeStdOut(MESSAGE)
      do i=1,10
         r_hf = r_af(1) + r_x(i)*(r_af(2) + r_x(i)*r_af(3))
         r_cff = r_cf(1) + r_x(i)*(r_cf(2) + r_x(i)*r_cf(3))
         write(MESSAGE,'(a,1x,f12.2,1x,f12.2,1x,f12.6)') 'Fit check h: ',r_hfit(i),r_hf,r_hf-r_hfit(i)
         call writeStdOut(MESSAGE)
         write(MESSAGE,'(a,1x,f12.2,1x,f12.2,1x,f12.6)') 'Fit check c: ',r_cfit(i),r_cff,r_cff-r_cfit(i)
         call writeStdOut(MESSAGE)
      enddo

c     fit orbit one Cdot,Hdot values to a line

      i_nd = 10
      i_ma = 2
      i_list(1) = 1
      i_list(2) = 2
      call lfit(r_x,r_sdotfit,r_sig,i_nd,r_sfdot,i_ma,i_list,i_ma,r_cov,i_ma,r_chisq)
      call lfit(r_x,r_hdotfit,r_sig,i_nd,r_afdot,i_ma,i_list,i_ma,r_cov,i_ma,r_chisq)
      call lfit(r_x,r_cdotfit,r_sig,i_nd,r_cfdot,i_ma,i_list,i_ma,r_cov,i_ma,r_chisq)
      vertvfit = r_afdot
      horizvfit = r_cfdot
      call writeStdOut(' * Linear Fit Coefficients for Height/Cross-Track/Along-Track * ')
      write(MESSAGE,'(a,1x,3(e20.10,1x))') 'Vertical Velocity Fit:   ',r_afdot
      call writeStdOut(MESSAGE)
      write(MESSAGE,'(a,1x,3(e20.10,1x))') 'Cross-Track Velocity Fit: ',r_cfdot
      call writeStdOut(MESSAGE)
      write(MESSAGE,'(a,1x,3(e20.10,1x))') 'Along-Track Velocity Fit: ',r_sfdot
      call writeStdOut(MESSAGE)
      do i=1,10
         r_hffdot = r_afdot(1) + r_x(i)*r_afdot(2) 
         r_cffdot = r_cfdot(1) + r_x(i)*r_cfdot(2)
         r_sffdot = r_sfdot(1) + r_x(i)*r_cfdot(2)
         write(MESSAGE,'(a,1x,f12.2,1x,f12.2,1x,f12.6)') 'Fit check H: ',r_hdotfit(i),r_hffdot,r_hffdot-r_hdotfit(i)
         call writeStdOut(MESSAGE)
         write(MESSAGE,'(a,1x,f12.2,1x,f12.2,1x,f12.6)') 'Fit check C: ',r_cdotfit(i),r_cffdot,r_cffdot-r_cdotfit(i)
         call writeStdOut(MESSAGE)
         write(MESSAGE,'(a,1x,f12.2,1x,f12.2,1x,f12.6)') 'Fit check S: ',r_sdotfit(i),r_sffdot,r_sffdot-r_cdotfit(i)
         call writeStdOut(MESSAGE)
      enddo

      do i=1,i_numlines
         r_t = r_time_first_line + (1.d0/r_prf)*(i-1)*i_looksaz
         call inter_motion(r_time,r_xyz1,i_numobs,r_t,r_xyzvec)
         call inter_motion(r_time,r_vxyz1,i_numobs,r_t,r_xyzvel)
         r_intPos(i,:) = r_xyzvec(:) 
         r_intVel(i,:) = r_xyzvel(:)
      enddo

      r_pegRadius = ptm%r_radcur
      r_grndSpace = r_delsint*r_scale
      r_transVect = ptm%r_ov
      r_transfMat =  ptm%r_mat
       
      end  

     
c***********************************************************************

      subroutine funcs(x,p,np)

      real*8 x 
      real*8 p(np)

      p(1) = 1.
      do j=2,np
        p(j) = p(j-1)*x
      enddo
      return
      end


