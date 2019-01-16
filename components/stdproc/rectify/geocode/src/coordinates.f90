module coordinates
  !!********************************************************
  !*
  !*    DESCRIPTION: ! This module contains functions to transform betwen llh, xyz, and sch
  !*
  !*    FUNCTION LIST: radar_to_xyz, rdir, reast, rnorth, latlon
  !*            convert_sch_to_xyz
  !*
  !!*********************************************************
  use linalg
  use fortranUtils
  implicit none
  
  
  ! declare data types
  type :: ellipsoid 
     real*8 r_a         ! semi-major axis
     real*8 r_e2                ! eccentricity-squared of earth ellipsoid
  end type ellipsoid
  
  type :: pegpoint 
     real*8 r_lat               ! peg latitude
     real*8 r_lon               ! peg longitude
     real*8 r_hdg               ! peg heading
  end type pegpoint
  
  type :: pegtrans 
     real*8 r_mat(3,3)  !transformation matrix SCH->XYZ
     real*8 r_matinv(3,3)       !transformation matrix XYZ->SCH
     real*8 r_ov(3)     !Offset vector SCH->XYZ
     real*8 r_radcur    !peg radius of curvature
  end type pegtrans
  
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! SUBROUTINES & FUNCTIONS
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
contains
  subroutine radar_to_xyz(elp,peg,ptm,height)
    !c****************************************************************
    !c**
    !c**        FILE NAME: radar_to_xyz.f
    !c**
    !c**     DATE WRITTEN:1/15/93 
    !c**
    !c**     PROGRAMMER:Scott Hensley
    !c**
    !c**        FUNCTIONAL DESCRIPTION: This routine computes the transformation
    !c**     matrix and translation vector needed to get between radar (s,c,h)
    !c**     coordinates and (x,y,z) WGS-84 coordinates.
    !c**
    !c**     ROUTINES CALLED: latlon,rdir
    !c**  
    !c**     NOTES: none
    !c**
    !c**     UPDATE LOG:
    !c**
    !c*****************************************************************
    
    
    ! input/output variables
    type(ellipsoid), intent(in) :: elp
    type(pegpoint), intent(in) :: peg
    type(pegtrans), intent(out) :: ptm

    real*8, intent(in), optional :: height
    
    ! local variables
    integer i,j,i_type
    real*8 r_radcur,r_llh(3),r_p(3),r_slt,r_clt,r_clo,r_slo,r_up(3)
    real*8 r_chg,r_shg
    real*8 r_height
    
    ! processing steps
    !Check if the height is given
    if (present(height)) then
        r_height = height
    else
        r_height = 0.0d0
    endif

    ! first determine the rotation matrix
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
    
    ! find the translation vector
    ptm%r_radcur = rdir(elp%r_a,elp%r_e2,peg%r_hdg,peg%r_lat) + r_height
    
    i_type = 1
    r_llh(1) = peg%r_lat
    r_llh(2) = peg%r_lon
    r_llh(3) = r_height
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
  end subroutine radar_to_xyz
  
  
  !c****************************************************************
  !c       Various curvature functions 
  !c****************************************************************
  !c**
  !c**  FILE NAME: curvature.f
  !c**
  !c**     DATE WRITTEN: 12/02/93
  !c**
  !c**     PROGRAMMER:Scott Hensley
  !c**
  !c**  FUNCTIONAL DESCRIPTION: This routine computes the curvature for 
  !c**     of various types required for ellipsoidal or spherical earth 
  !c**     calculations.  
  !c**
  !c**     ROUTINES CALLED: none
  !c**  
  !c**     NOTES: none
  !c**
  !c**     UPDATE LOG:
  !c**
  !c*****************************************************************      
  real*8 function rdir(r_a,r_e2,r_hdg,r_lat)
    real*8, intent(in) :: r_a,r_e2,r_lat,r_hdg
    real*8 :: r_re, r_rn  
    r_re = reast(r_a,r_e2,r_lat)
    r_rn = rnorth(r_a,r_e2,r_lat)
    rdir = (r_re*r_rn)/(r_re*cos(r_hdg)**2 + r_rn*sin(r_hdg)**2) 
  end function rdir
  
  real*8 function reast(r_a,r_e2,r_lat)
    real*8, intent(in) ::  r_a,r_e2,r_lat      
    reast = r_a/sqrt(1.d0 - r_e2*sin(r_lat)**2) 
  end function reast
  
  real*8 function rnorth(r_a,r_e2,r_lat)
    real*8, intent(in) ::  r_a,r_e2,r_lat
    rnorth = (r_a*(1.d0 - r_e2))/(1.d0 - r_e2*sin(r_lat)**2)**(1.5d0) 
  end function rnorth
  
  
  subroutine latlon(elp,r_v,r_llh,i_type) 
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
    
    
    ! input/output variables
    integer, intent(in) :: i_type          !1=lat,lon to vector,2= vector to lat,lon
    type(ellipsoid), intent(in) :: elp
    real*8, intent(inout), dimension(3) :: r_v      !geocentric vector (meters)
    real*8, intent(inout), dimension(3) :: r_llh    !latitude (deg -90 to 90),
    !longitude (deg -180 to 180),height
    
    ! local variables
    integer i_ft
    real*8 r_re,r_q2,r_q3,r_b,r_q
    real*8 r_p,r_tant,r_theta,r_a,r_e2
    real*8  pi, r_dtor

    pi = getPi()
    r_dtor = pi/180.d0
    
    
    ! processing steps
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
       r_tant = (r_v(3) + r_q3*r_b*sin(r_theta)**3)/(r_p - r_e2*r_a*cos(r_theta)**3)
       r_llh(1) =  atan(r_tant)
       r_re = r_a/sqrt(1.d0 - r_e2*sin(r_llh(1))**2)
       r_llh(3) = r_p/cos(r_llh(1)) - r_re          
    endif
  end subroutine latlon
  
  
  subroutine convert_sch_to_xyz(ptm,r_schv,r_xyzv,i_type)
    !c****************************************************************
    !c**
    !c**        FILE NAME: convert_sch_to_xyz.for
    !c**
    !c**     DATE WRITTEN:1/15/93 
    !c**
    !c**     PROGRAMMER:Scott Hensley
    !c**
    !c**        FUNCTIONAL DESCRIPTION: This routine applies the affine matrix 
    !c**     provided to convert the sch coordinates xyz WGS-84 coordintes or
    !c**     the inverse transformation.
    !c**
    !c**     ROUTINES CALLED: latlon,matvec,lincomb
    !c**  
    !c**     NOTES: none
    !c**
    !c**     UPDATE LOG:
    !c**
    !c*****************************************************************
    
    ! input/output variables
    type(pegtrans), intent(in) :: ptm
    real*8, intent(inout) :: r_schv(3)              !sch coordinates of a point
    real*8, intent(inout) ::  r_xyzv(3)              !WGS-84 coordinates of a point
    integer, intent(in) :: i_type       !i_type = 0 sch => xyz ; 
    !i_type = 1 xyz => sch
    
    ! local variables
    integer i_t
    real*8 r_schvt(3),r_llh(3)
    type(ellipsoid) :: sph
    
    ! processing steps
    
    ! compute the linear portion of the transformation 
    sph%r_a = ptm%r_radcur
    sph%r_e2 = 0.0d0
    
    if(i_type .eq. 0) then
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
    end if
  end subroutine convert_sch_to_xyz
  
  
  
  
end module coordinates
