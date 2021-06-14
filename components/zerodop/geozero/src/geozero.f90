subroutine geozero(demAccessor,inAccessor,demCropAccessor,outAccessor,inband,outband,iscomplex,method,lookSide)
  use geozeroState
  use geozeroReadWrite
  use geozeroMethods
  use poly1dModule
  use geometryModule
  use orbitModule
  use linalg3Module
  use fortranUtils, ONLY: getPI
  
  implicit none
  include 'omp_lib.h'


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! DECLARE LOCAL VARIABLES
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  integer inband,outband
  integer iscomplex,method,lookSide
  integer stat,cnt
  integer*8 inAccessor,demAccessor
  integer*8 outAccessor,demCropAccessor
  real*4, dimension(:),allocatable :: dem
  integer*2, dimension(:),allocatable :: dem_crop

  !!!!Image limits
  real*8 tstart, tend, tline, tprev
  real*8 rngstart, rngend, rngpix

  !!!! Satellite positions
  real*8, dimension(3) :: xyz_mid, vel_mid
  real*8 :: tmid, rngmid, temp

  type(ellipsoidType) :: elp
  real*8 :: llh(3),xyz(3)
  real*8 :: satx(3), satv(3)
  real*8 :: dr(3)
  integer :: pixel,line, ith
  integer :: min_lat_idx,max_lat_idx
  integer :: min_lon_idx,max_lon_idx
  complex,allocatable,dimension(:) :: geo

  !!!Debugging - PSA
  !real*4, allocatable, dimension(:,:) :: distance

  real*8 :: lat0,lon0
  integer :: geo_len, geo_wid,i_type,k
  real*8 ::  az_idx, rng_idx
  integer :: idxlat,idxlon
  complex, allocatable,dimension(:,:) :: ifg
  complex z
  integer :: int_rdx,int_rdy
  real*8 :: fr_rdx,fr_rdy
  integer :: i,j,lineNum
  real*8 :: dtaz, dmrg

  real*8 :: min_latr,min_lonr,max_latr,max_lonr
  real*8 :: lat_firstr,lon_firstr,dlonr,dlatr
  real*8 :: c1,c2,c3
  real*8 :: dopfact,fdop,fdopder

  integer :: numOutsideDEM
  integer :: numOutsideImage

  real*4 :: timer0, timer1  

  ! declare constants
  real*8 pi,rad2deg,deg2rad 
  real*8 BAD_VALUE
  parameter(BAD_VALUE = -10000.0d0)

  !! Cross product holder, for comparison to lookSide
  real*8 :: look_side_vec(3)
  real*8 look_side_sign
  integer pixel_side

  !Doppler factor
  type(poly1dType) :: fdvsrng, fddotvsrng

    procedure(readTemplate), pointer :: readBand => null()
    procedure(writeTemplate), pointer :: writeBand => null()
    procedure(intpTemplate), pointer :: intp_data => null()

    !!Set up the correct readers and writers
    if(iscomplex.eq.1) then
        readBand => readCpxLine
        writeBand => writeCpxLine
    else
        readBand => readRealLine
        writeBand => writeRealLine
    endif

!    method = NEAREST_METHOD

    if (method.eq.SINC_METHOD) then
        intp_data => intp_sinc
        print *, 'Using Sinc interpolation'
    else if (method.eq.BILINEAR_METHOD) then
        intp_data => intp_bilinear
        print *, 'Using bilinear inteprolation'
    else if (method.eq.BICUBIC_METHOD) then
        intp_data => intp_bicubic
        print *, 'Using bicubic'
    else if (method.eq.NEAREST_METHOD) then
        intp_data => intp_nearest
        print *, 'Using nearest neighbor interpolation'
    else
        print *, 'Undefined interpolation method.'
        stop
    endif
  
  pi = getPi()
  rad2deg = 180.d0/pi
  deg2rad = pi/180.d0

  ! get starting time
  timer0 = secnds(0.0)
  cnt = 0

  !$OMP PARALLEL
  !$OMP MASTER
  ith = omp_get_num_threads() !total num threads
  !$OMP END MASTER
  !$OMP END PARALLEL
  print *, "threads",ith


  elp%r_a= majorSemiAxis
  elp%r_e2= eccentricitySquared


  tstart = t0
  dtaz = Nazlooks / prf
  tend  = t0 + (length-1)* dtaz
  tmid = 0.5d0*(tstart+tend)

  print *, 'Starting Acquisition time: ', tstart
  print *, 'Stop Acquisition time: ', tend
  print *, 'Azimuth line spacing in secs: ', dtaz

  rngstart = rho0
  dmrg = Nrnglooks * drho
  rngend = rho0 + (width-1)*dmrg
  rngmid = 0.5d0*(rngstart+rngend)
  print *, 'Near Range in m: ', rngstart 
  print *, 'Far  Range in m: ', rngend
  print *, 'Range sample spacing in m: ', dmrg

  print *, 'Input Lines: ', length
  print *, 'Input Width: ', width


  ! Convert everything to radians
  dlonr = dlon*deg2rad
  dlatr = dlat*deg2rad
  lon_firstr = lon_first*deg2rad
  lat_firstr = lat_first*deg2rad


  ! allocate
  allocate(dem(demwidth))
  allocate(ifg(width,length))
  dem = 0
  ifg = 0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! PROCESSING STEPS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  print *, "reading interferogram ..."

  ! convert deg to rad
  min_latr = min_lat*deg2rad
  max_latr = max_lat*deg2rad
  min_lonr = min_lon*deg2rad
  max_lonr = max_lon*deg2rad
  min_lat_idx=(min_latr-lat_firstr)/dlatr + 1 
  min_lon_idx=(min_lonr-lon_firstr)/dlonr
  max_lat_idx=(max_latr-lat_firstr)/dlatr
  max_lon_idx=(max_lonr-lon_firstr)/dlonr + 1
  geo_len = (min_lat_idx - max_lat_idx) 
  geo_wid = (max_lon_idx - min_lon_idx) 
 
!!  call printOrbit_f(orbit)

  print *, 'Geocoded Lines:  ', geo_len
  print *, 'Geocoded Samples:', geo_wid 

  call init_RW(max(width,geo_wid),iscomplex)

  ! Read in the data
  do i=1,length
      call readBand(inAccessor,ifg(:,i),inband,lineNum,width)
  enddo

  ! allocate a line of the output geocoded image
  allocate(geo(geo_wid),dem_crop(geo_wid))

  !!!!Allocate arrays for interpolation if needed 
  call prepareMethods(method)


  !!!!Setup doppler polynomials
  call initPoly1D_f(fdvsrng, dopAcc%order)
  fdvsrng%mean = rho0 + dopAcc%mean * drho !!drho is original full resolution.
  fdvsrng%norm = dopAcc%norm * drho   !!(rho/drho) is the proper original index for Doppler polynomial

  !!!Coeff indexing is zero-based
  do k=1,dopAcc%order+1
     temp = getCoeff1d_f(dopAcc,k-1)
     temp = temp*prf
     call setCoeff1d_f(fdvsrng, k-1, temp)
  end do

  !!!Set up derivative polynomial
  if (fdvsrng%order .eq. 0) then
      call initPoly1D_f(fddotvsrng, 0)
      call setCoeff1D_f(fddotvsrng, 0, 0.0d0)
  else
      call initPoly1D_f(fddotvsrng, fdvsrng%order-1)
      fddotvsrng%mean = fdvsrng%mean
      fddotvsrng%norm = fdvsrng%norm

      do k=1,dopAcc%order
        temp = getCoeff1d_f(fdvsrng, k)
        temp = k*temp/fdvsrng%norm
        call setCoeff1d_f(fddotvsrng, k-1, temp)
      enddo
  endif



  !!!!Initialize satellite positions
  tline = tmid
  stat =  interpolateWGS84Orbit_f(orbit, tline, xyz_mid, vel_mid)

  if (stat.ne.0) then
      print *, 'Cannot interpolate orbits at the center of scene.'
      stop
  endif


  print *, "geocoding on ",ith,' threads...'
 
  numOutsideDEM = 0
  numOutsideImage = 0

  do line = 1, geo_len
     geo = cmplx(0.,0.)
     dem_crop = 0

     !!Read online of the DEM to process
     idxlat = max_lat_idx + (line-1)
     if (idxlat.lt.0.or.idxlat.gt.(demlength-1)) then
         numOutsideDEM = numOutsideDEM + demwidth
         goto 300
     endif

     pixel = idxlat+1 
     call getLine(demAccessor,dem,pixel)

     !$OMP PARALLEL DO private(pixel,i_type,k)&
     !$OMP private(xyz,llh,rngpix,tline,satx,satv)&
     !$OMP private(rng_idx,z,idxlon,dr,c1,c2,tprev)&
     !$OMP private(az_idx,int_rdx,int_rdy,fr_rdx,fr_rdy)&
     !$OMP private(dopfact,fdop,fdopder,c3) &
     !$OMP shared(geo_len,geo_wid,f_delay) &
     !$OMP shared(width,length,ifg)&
     !$OMP shared(dem,fintp,demwidth,demlength) &
     !$OMP shared(line,elp,ilrl,tstart,tmid,rngstart,rngmid) &
     !$OMP shared(xyz_mid,vel_mid,idxlat,fdvsrng,fddotvsrng) &
     !$OMP shared(max_lat_idx,min_lon_idx,dtaz,dmrg) &
     !$OMP shared(lat_firstr,lon_firstr,dlatr,dlonr)&
     !$OMP shared(numOutsideDEM,numOutsideImage,wvl,orbit) 
     do pixel = 1,geo_wid
        
        !!Default values
        z = cmplx(0., 0.)  !!Default value if out of grid
        llh(3) = 0.        !!Default height if point requested outsideDEM

        idxlat = max_lat_idx + (line-1)
        llh(1) = lat_firstr + idxlat * dlatr 
        
        idxlon = min_lon_idx + (pixel-1)
        llh(2) = lon_firstr + idxlon * dlonr
        if (idxlon.lt.0.or.idxlon.gt.(demwidth-1)) goto 200

        
        llh(3) = dem(idxlon+1)
        ! catch bad SRTM pixels
        if(llh(3).lt.-1500) then
            goto 100
        endif


200        continue

        i_type = LLH_2_XYZ
        call latlon(elp,xyz,llh,i_type)


        !!!!Actual iterations
        tline = tmid
        satx = xyz_mid
        satv = vel_mid

        ! Check that the pixel is on the correct side of the platform
        ! https://github.com/isce-framework/isce2/issues/294#issuecomment-853413396
        dr = xyz - satx
        call cross(dr, satv, look_side_vec)
        look_side_sign = dot(look_side_vec, satx)
        if(look_side_sign.gt.0) then
            pixel_side = -1
        else
            pixel_side = 1
        endif
        ! Skip if the current pixel side doesn't matches the look side
        if(pixel_side.ne.lookSide) then
            ! print *, "Skipp. lookSide ", lookSide, "look_side_sign", look_side_sign
            goto 100
        endif

        do k=1,21
            tprev = tline  
!!            print *, pixel, k, tline
            dr = xyz - satx
            rngpix = norm(dr)    

            dopfact  = dot(dr,satv) / rngpix
            fdop = 0.5d0 * wvl*evalPoly1d_f(fdvsrng,rngpix)
            fdopder = 0.5d0 * wvl * evalPoly1d_f(fddotvsrng,rngpix)

            !!!c1 is misfit at current guess location
            c1 = dopfact - fdop

            !!!c2 is correction term when zero doppler geometry is used
            c2 = dot(satv, satv)/rngpix

            !!!c3 is additional correction term when native doppler geometry is used
            c3 = dopfact * (fdop / rngpix +  fdopder)

            tline = tline + c1/(c2-c3)

            stat = interpolateWGS84Orbit_f(orbit,tline,satx,satv)

            if (stat.ne.0) then
                tline = BAD_VALUE
                rngpix = BAD_VALUE
                exit
            endif

            if (abs(tline - tprev).lt.5.0d-7) exit
        enddo


        az_idx = ((tline - tstart)/dtaz) + 1
        rng_idx = ((rngpix-rngstart)/dmrg) + 1

        if(rng_idx.le.f_delay) then
            numOutsideImage = numOutsideImage + 1
            goto 100
        endif

        if(rng_idx.ge.width-f_delay) then
            numOutsideImage = numOutsideImage + 1
            goto 100
        endif

        if(az_idx.le.f_delay) then
            numOutsideImage = numOutsideImage + 1
            goto 100
        endif

        if(az_idx.ge.length-f_delay) then
            numOutsideImage = numOutsideImage + 1
            goto 100
        endif

        cnt = cnt + 1


        int_rdx=int(rng_idx+f_delay)
        fr_rdx=rng_idx+f_delay-int_rdx
        int_rdy=int(az_idx+f_delay)
        fr_rdy=az_idx+f_delay-int_rdy

        !! The indices are offset by f_delay for sinc
        !! Other methods adjust this bias in intp_call
        z = intp_data(ifg,int_rdx,int_rdy,fr_rdx,fr_rdy,width,length)


100        continue

           geo(pixel) = z
           dem_crop(pixel) = llh(3)

        enddo
        !$OMP END PARALLEL DO

        ! write output file
300     call writeBand(outAccessor,geo,outband,geo_wid)

        if(demCropAccessor.gt.0) then
            call setLineSequential(demCropAccessor,dem_crop)
        endif
    enddo

  print *, 'Number of pixels with outside DEM:  ', numOutsideDEM
  print *, 'Number of pixels outside the image: ', numOutsideImage
  print *, 'Number of pixels with valid data:   ', cnt

  !!!!Clean polynomials
  call cleanpoly1d_f(fdvsrng)
  call cleanpoly1d_f(fddotvsrng)

  call finalize_RW(iscomplex)
  call unprepareMethods(method)

  
  geowidth = geo_wid
  geolength = geo_len
  geomin_lat = (lat_first + min_lat_idx*dlat)
  geomax_lat = (lat_first + max_lat_idx*dlat)
  geomin_lon = (lon_first + min_lon_idx*dlon) 
  geomax_lon = (lon_first + max_lon_idx*dlon)
  
  deallocate(dem,geo,dem_crop)
  deallocate(ifg)

  nullify(readBand,writeBand,intp_data)

  timer1 = secnds(timer0)
  print *, 'elapsed time = ',timer1,' seconds'
end 
        
