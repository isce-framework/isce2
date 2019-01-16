!****************************************************************

      subroutine get_frate(r_platsch,r_platvel,r_range,r_prf,r_tarsch,pln, &
                           i_lrl,r_wavl,peg,elp,r_doppler,r_frate)


	! r_platsch     -> Platform position SCH values
	! r_platvel     -> Platform velocity SCH values
	! r_range       -> Range to target 
	! r_prf         -> Pulse repetition frequency
	! r_tarsch      -> Location of target in SCH 
	! pln           -> Planet description
	! i_lrl         -> Left / Right looking
	! r_wavl        -> Wavelength
	! peg           -> Peg point
	! elp           -> Ellipse description
	! r_doppler     -> Doppler centroid  (Output)
	! r_frate       -> Doppler centroid rate (Output)

!****************************************************************
!**     
!**   FILE NAME: get_frate.f
!**     
!**   DATE WRITTEN:1/28/99
!**     
!**   PROGRAMMER: Paul Rosen 
!**     
!**   FUNCTIONAL DESCRIPTION: Compute the exact Doppler rate based
!**   on vector formula derived by Scott Hensley 
!**
!**   ROUTINES CALLED: several geometry subroutines
!**     
!**   NOTES: 
!**
!**   UPDATE LOG:
!**
!**   Date Changed        Reason Changed                  CR # and Version #
!**   ------------       ----------------                 -----------------
!**     
!*****************************************************************

      implicit none

!     INCLUDE FILES:

!     INPUT VARIABLES:
      type peg_type
         double precision :: r_lat                   !< Peg point latitude
         double precision :: r_lon                   !< Peg point longitude
         double precision :: r_hdg                   !< Peg point heading
      end type peg_type
      type pegtrans
         double precision :: r_mat(3,3)              !< Peg transformation matrix SCH -> XYZ
         double precision :: r_matinv(3,3)           !< Inverse peg transformation matrix XYZ -> SCH
         double precision :: r_ov(3)                 !< Peg origin offset vector
         double precision :: r_radcur                !< Radius of curvature
      end type pegtrans
      type ellipsoid
         double precision :: r_a                     !< Semi-major axis
         double precision :: r_e2                    !< Eccentricity squared
      end type ellipsoid
      type planet_type
         double precision :: r_spindot               !< Planet spin rate
         double precision :: r_gm                    !< Planet GM
      end type planet_type

      double precision :: r_platsch(3), r_platvel(3) !< Platform position and velocity in SCH coordinates
      double precision :: r_range                    !< Range to the target [m] 
      double precision :: r_prf                      !< Pulse repetition frequency [Hz]
      double precision :: r_tarsch(3)                !< Location of the target in SCH coordinates
      integer :: i_lrl                               !< Left or right looking radar
      double precision :: r_wavl                     !< Radar wavelength [m]
      type(planet_type) :: pln
      type(peg_type) :: peg                          !< Coordinate and heading defining the SCH coordinate system
      type(ellipsoid) :: elp
      
      real*8 r_platacc(3)        !platform acceleration
      real*8 r_yaw               !platform Yaw
      real*8 r_pitch             !platform Pitch
      real*8 r_azesa             !azimuth steering angle

!     OUTPUT VARIABLES:

      double precision :: r_doppler !< Doppler centroid value [Hz]
      double precision :: r_frate   !< Doppler centroid rate for target [Hz/s]

!     LOCAL VARIABLES:

      integer i_schtoxyz 
      integer i_xyztosch 
      real*8 r_x1, r_x2, r_l3
      real*8 r_look, r_lookvec(3)
      real*8 r_vdotl, r_adotl, r_veln , r_accn
      real*8 r_spinvec(3)
      integer k
      real*8 r_xyz(3),r_tempv(3), r_inertialacc(3),r_tempa(3),r_tempvec(3),r_xyzdot(3)
      real*8 r_bodyacc(3),r_xyzschmat(3,3),r_schxyzmat(3,3),r_xyznorm,r_dx,r_dcnorm

      type(pegtrans) :: ptm !< SCH transformation parameters

!     COMMON BLOCKS:

!     EQUIVALENCE STATEMENTS:

!     DATA STATEMENTS:

      real*8, parameter :: r_dtor = atan(1.d0) / 45.d0

!     FUNCTION STATEMENTS:
      
      real*8 dot

!     SAVE STATEMENTS:

!     PROCESSING STEPS:

      i_schtoxyz = 0  !< Convert from sch => xyz
      i_xyztosch = 1  !< Convert from xyz => sch

      
      ! Assume no yaw, pitch, or azimuth steering
      r_yaw = 0.D0
      r_pitch = 0.D0
      r_azesa = 0.D0

      
      ! Assume that the target is on the ellipsoid
      do k=1,3
         r_tarsch(k) = 0.D0
      enddo

      !     Calculate Peg point transformation parameters
      call radar_to_xyz(elp,peg,ptm)


      !     acceleration - use Newton's Universal Law of Gravitation
      r_spinvec(1) = 0.
      r_spinvec(2) = 0.
      r_spinvec(3) = pln%r_spindot

      !     Convert position to XYZ coordinates
      call convert_sch_to_xyz(ptm,r_platsch,r_xyz,i_schtoxyz)

      ! Normalize
      call norm(r_xyz,r_xyznorm)

      ! Compute cross product
      call cross(r_spinvec,r_xyz,r_tempv)
      
      ! Use gravity for inertial acceleration
      do k=1,3
         r_inertialacc(k) = -(pln%r_gm*r_xyz(k))/r_xyznorm**3
      enddo

      ! Transform SCH velocity to XYZ
      call convert_schdot_to_xyzdot(ptm,r_platsch,r_platvel,r_xyzdot,i_schtoxyz)

      ! Cross product of spin and velocity
      call cross(r_spinvec,r_xyzdot,r_tempa)
      call cross(r_spinvec,r_tempv,r_tempvec)
      
      do k=1,3
         r_bodyacc(k) = r_inertialacc(k) - 2.d0*r_tempa(k) - r_tempvec(k)
      enddo

!     convert acceleration back to a local SCH basis
      
      call schbasis(ptm,r_platsch,r_xyzschmat,r_schxyzmat)
      call matvec(r_xyzschmat,r_bodyacc,r_platacc)

!     compute the Doppler and Frate

      r_x1 = (ptm%r_radcur + r_platsch(3))     !Radius to satellite
      r_x2 = (ptm%r_radcur + r_tarsch(3))      !Radius to target

      r_l3 = (r_x1**2 + r_range**2 - r_x2**2)/(2.d0*r_x1*r_range)    !Cosine law
      r_look = acos((r_l3 + sin(r_azesa)*sin(r_pitch))/(cos(r_pitch)*cos(r_azesa)))  !Look angle
     
      ! Look vector components 
      r_lookvec(1) = (cos(r_look)*sin(r_pitch)*cos(r_yaw) + sin(r_look)*sin(r_yaw)*i_lrl)* &
          cos(r_azesa) - sin(r_azesa)*cos(r_pitch)*cos(r_yaw)
      r_lookvec(2) = (-cos(r_look)*sin(r_pitch)*sin(r_yaw) + sin(r_look)*cos(r_yaw)*i_lrl)* &
          cos(r_azesa) + sin(r_azesa)*cos(r_pitch)*sin(r_yaw)
      r_lookvec(3) = -cos(r_look)*cos(r_pitch)*cos(r_azesa) - sin(r_azesa)*sin(r_pitch)

      ! Dot product of look vector and velocity
      r_vdotl = dot(r_lookvec,r_platvel)
      call norm(r_platvel,r_veln)
      call norm(r_platacc,r_accn)
      
      r_doppler = 2.d0*r_vdotl/r_wavl	    !Doppler formula
      r_dcnorm = r_doppler/r_prf	    !Normalized doppler
      r_dx = r_veln/r_prf		    !Azimuth spacing
      
      !Dot product of acceleration and look vector
      r_adotl = dot(r_lookvec,r_platacc)

      !Doppler rate including the acceleration term
      r_frate = 2.d0*(r_adotl + (r_vdotl**2 - r_veln**2)/r_range)/(r_wavl)
       
    end
