subroutine bistaticgeo2rdr(latAcc,lonAcc,hgtAcc,azAcc,rgAcc,azoffAcc,rgoffAcc)
  use bistaticgeo2rdrState
  use poly1dModule
  use geometryModule
  use orbitModule
  use linalg3Module
  use fortranUtils, ONLY: getPI,getSpeedOfLight
  
  implicit none
  include 'omp_lib.h'


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! DECLARE LOCAL VARIABLES
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  integer stat,cnt
  integer*8 latAcc,lonAcc,hgtAcc
  integer*8 azAcc,rgAcc
  integer*8 azOffAcc,rgOffAcc
  real*8, dimension(:),allocatable :: lat
  real*8, dimension(:),allocatable :: lon
  real*8, dimension(:),allocatable :: dem
  real*8, dimension(:),allocatable :: rgm
  real*8, dimension(:),allocatable :: azt
  real*8, dimension(:),allocatable :: rgoff
  real*8, dimension(:),allocatable :: azoff
  real*4, dimension(:),allocatable :: distance

  !!!!Image limits
  real*8 tstart, tend, tline, tprev
  real*8 rngstart, rngend, rngpix
  real*8 arng, prng

  !!!! Satellite positions
  real*8, dimension(3) :: axyz_mid, avel_mid, aacc_mid
  real*8, dimension(3) :: pxyz_mid, pvel_mid, pacc_mid
  real*8 :: tmid, rngmid, temp

  type(ellipsoidType) :: elp
  real*8 :: llh(3),xyz(3)
  real*8 :: asatx(3), asatv(3),asata(3)
  real*8 :: psatx(3), psatv(3),psata(3)

  real*8 :: adr(3), pdr(3)
  integer :: pixel,line, ith

  integer :: i_type,k,conv
  real*8 :: dtaz, dmrg
  real*8 :: adopfact, pdopfact, fdop,fdopder

  real*8 :: c1,c2

  integer :: numOutsideImage

  real*4 :: timer0, timer1  

  !!Function pointer for orbit interpolation
  procedure(interpolateOrbit_f), pointer :: intp_orbit => null()

  ! declare constants
  real*8 pi,rad2deg,deg2rad,sol 
  real*8 fn, fnprime
  real*4 BAD_VALUE
  parameter(BAD_VALUE = -999999.0)


  !Doppler factor
  type(poly1dType) :: fdvsrng,fddotvsrng

  pi = getPi()
  sol = getSpeedOfLight()
  rad2deg = 180.d0/pi
  deg2rad = pi/180.d0

       !!!Set up orbit interpolation method
        if (orbitmethod .eq. HERMITE_METHOD) then
            intp_orbit => interpolateWGS84Orbit_f

            if(aorbit%nVectors .lt. 4) then
                print *, 'Need atleast 4 state vectors for using hermite polynomial interpolation'
                stop
            endif

            if(porbit%nVectors .lt. 4) then
                print *, 'Need atleast 4 state vectors for using hermite polynomial interpolation'
                stop
            endif


            print *, 'Orbit interpolation method: hermite'
        else if (orbitmethod .eq. SCH_METHOD) then
            intp_orbit => interpolateSCHOrbit_f

            if(aorbit%nVectors .lt. 4) then
                print *, 'Need atleast 4 state vectors for using SCH interpolation'
                stop
            endif
            if(porbit%nVectors .lt. 4) then
                print *, 'Need atleast 4 state vectors for using SCH interpolation'
                stop
            endif


            print *, 'Orbit interpolation method: sch'
        else if (orbitmethod .eq. LEGENDRE_METHOD) then
            intp_orbit => interpolateLegendreOrbit_f

            if(aorbit%nVectors .lt. 9) then
                print *, 'Need atleast 9 state vectors for using legendre polynomial interpolation'
                stop
            endif
            if(porbit%nVectors .lt. 9) then
                print *, 'Need atleast 9 state vectors for using legendre polynomial interpolation'
                stop
            endif
            print *, 'Orbit interpolation method: legendre'
        else
            print *, 'Undefined orbit interpolation method.'
            stop
        endif


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

  rngstart = arho0
  dmrg = Nrnglooks * drho
  rngend = arho0 + (width-1)*dmrg
  rngmid = 0.5d0*(rngstart+rngend)
  print *, 'Near Range in m: ', rngstart 
  print *, 'Far  Range in m: ', rngend
  print *, 'Range sample spacing in m: ', dmrg

  print *, 'Radar Image Lines: ', length
  print *, 'Radar Image Width: ', width



  ! allocate
  allocate(lat(demwidth))
  allocate(lon(demwidth))
  allocate(dem(demwidth))
  allocate(rgm(demwidth))
  allocate(azt(demwidth))
  allocate(rgoff(demwidth))
  allocate(azoff(demwidth))
  allocate(distance(demwidth))

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! PROCESSING STEPS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  print *, "reading dem ..."

  print *, 'Geocoded Lines:  ', demlength
  print *, 'Geocoded Samples:', demwidth 



  !!!!Setup doppler polynomials
  call initPoly1D_f(fdvsrng, dopAcc%order)
  fdvsrng%mean = arho0 + dopAcc%mean * drho !!drho is original full resolution.
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


  print *, 'Dopplers: ', evalPoly1d_f(fdvsrng, rngstart), evalPoly1d_f(fdvsrng,rngend)

  !!!!Initialize satellite positions
  tline = tmid
  stat =  intp_orbit(aorbit, tline, axyz_mid, avel_mid)

  if (stat.ne.0) then
      print *, 'Cannot interpolate orbits at the center of scene.'
      stop
  endif

  stat = computeAcceleration_f(aorbit, tline, aacc_mid)

  if (stat.ne.0) then
      print *, 'Cannot compute acceleration at the center of scene.'
      stop
  endif

  stat =  intp_orbit(porbit, tline, pxyz_mid, pvel_mid)

  if (stat.ne.0) then
      print *, 'Cannot interpolate orbits at the center of scene.'
      stop
  endif

  stat = computeAcceleration_f(porbit, tline, pacc_mid)

  if (stat.ne.0) then
      print *, 'Cannot compute acceleration at the center of scene.'
      stop
  endif



  print *, "geo2rdr on ",ith,' threads...'
 
  numOutsideImage = 0

!!  open(31, file='fndistance',access='direct',recl=4*demwidth,form='unformatted')
  do line = 1, demlength
     !!Initialize
     azt = BAD_VALUE
     rgm = BAD_VALUE
     rgoff = BAD_VALUE
     azoff = BAD_VALUE
     distance = BAD_VALUE

     !!Read in positions
     call getLineSequential(hgtAcc,dem,pixel)
     call getLineSequential(latAcc,lat,pixel)
     call getLineSequential(lonAcc,lon,pixel)

     if (mod(line,1000).eq.1) then
         print *, 'Processing line: ', line, numoutsideimage
     endif
     conv = 0

     !$OMP PARALLEL DO private(pixel,i_type,k)&
     !$OMP private(xyz,llh,rngpix,tline,asatx,asatv)&
     !$OMP private(c1,c2,tprev,stat,fn,fnprime)&
     !$OMP private(adopfact,pdopfact,fdop,fdopder,asata) &
     !$OMP private(psatx,psatv,psata,adr,pdr,arng,prng)&
     !$OMP shared(length,width,demwidth) &
     !$OMP shared(rgm,azt,rgoff,azoff) &
     !$OMP shared(line,elp,ilrl,tstart,tmid,rngstart,rngmid) &
     !$OMP shared(axyz_mid,avel_mid,aacc_mid,fdvsrng,fddotvsrng) &
     !$OMP shared(pxyz_mid,pvel_mid,pacc_mid)&
     !$OMP shared(lat,lon,dem,dtaz,dmrg,deg2rad,bistatic,sol) &
     !$OMP shared(numOutsideImage,wvl,aorbit,porbit,conv,distance) 
     do pixel = 1,demwidth 
       
        llh(1) = lat(pixel) * deg2rad
        llh(2) = lon(pixel) * deg2rad
        llh(3) = dem(pixel)

        i_type = LLH_2_XYZ
        call latlon(elp,xyz,llh,i_type)


        !!!!Actual iterations
        tline = tmid
        asatx = axyz_mid
        asatv = avel_mid
        asata = aacc_mid
        psatx = pxyz_mid
        psatv = pvel_mid
        psata = pacc_mid

        do k=1,51
            tprev = tline  

            adr = xyz - asatx
            pdr = xyz - psatx

            arng = norm(adr)
            prng = norm(pdr)
            rngpix = 0.5d0*(arng+prng)

            adopfact = dot(adr,asatv)/arng 
            pdopfact = dot(pdr,psatv)/prng
            fdop = wvl * evalPoly1d_f(fdvsrng,rngpix)
            fdopder = wvl * evalPoly1d_f(fddotvsrng,rngpix)

            fn = adopfact + pdopfact

            c1 = -((dot(asatv,asatv)+adopfact*adopfact)/arng) - ((dot(psatv,psatv)+pdopfact*pdopfact)/prng) 
            c2 = 0.0d0 

            fnprime = c1 !+ c2*dopfact

!!            if (abs(fn) .le. 1.0d-5) then
!!                conv = conv + 1
!!                exit
!!            endif

            tline = tline - fn / fnprime

!!            print *, c1, c2, rngpix

            stat = intp_orbit(aorbit,tline,asatx,asatv)

            if (stat.ne.0) then
                tline = BAD_VALUE
                rngpix = BAD_VALUE
                exit
            endif

            
            stat = intp_orbit(porbit,tline,psatx,psatv)

            if (stat.ne.0) then
                tline = BAD_VALUE
                rngpix = BAD_VALUE
                exit
            endif


!            stat = computeAcceleration_f(orbit,tline,sata)
!            if (stat.ne.0) then
!                tline = BAD_VALUE
!                rngpix = BAD_VALUE
!                exit
!            endif

            !!!Check for convergence
            if (abs(tline - tprev).lt.5.0d-9) then
                conv = conv + 1
                exit
            endif
        enddo


        if(tline.lt.tstart) then
            numOutsideImage = numOutsideImage + 1
            goto 100
        endif

        if(tline.gt.tend) then
            numOutsideImage = numOutsideImage + 1
            goto 100
        endif

        adr = xyz - asatx
        pdr = xyz - psatx

        arng = norm(adr)
        prng = norm(pdr)
        rngpix = 0.5d0*(arng+prng)

        if(rngpix.lt.rngstart) then
            numOutsideImage = numOutsideImage + 1
            goto 100
        endif

        if(rngpix.gt.rngend) then
            numOutsideImage = numOutsideImage + 1
            goto 100
        endif

        cnt = cnt + 1
        rgm(pixel) = rngpix
        azt(pixel) = tline

        rgoff(pixel) = ((rngpix - rngstart)/dmrg) - 1.0d0*(pixel-1)
        azoff(pixel) = ((tline - tstart)/dtaz) - 1.0d0*(line-1)
        distance(pixel) = tline - tprev

100        continue


        enddo
        !$OMP END PARALLEL DO



        ! write output file
        if (azAcc.gt.0) then
            call setLineSequential(azAcc,azt)
        endif

        if (rgAcc.gt.0) then
            call setLineSequential(rgAcc,rgm)
        endif

        if (azoffAcc.gt.0) then
            call setLineSequential(azoffAcc,azoff)
        endif

        if (rgoffAcc.gt.0) then
            call setLineSequential(rgoffAcc,rgoff)
        endif
!!        write(31,rec=line)(distance(pixel),pixel=1,demwidth)
    enddo


    print *, 'Number of pixels outside the image: ', numOutsideImage
    print *, 'Number of pixels with valid data:   ', cnt
    print *, 'Number of pixels that converged:    ', conv

    !!!!Clean polynomials
    call cleanpoly1d_f(fdvsrng)
    call cleanpoly1d_f(fddotvsrng)

!!    close(31) 
    deallocate(lat,lon,dem)
    deallocate(azt,rgm)
    deallocate(azoff,rgoff)
    deallocate(distance)

     timer1 = secnds(timer0)
     print *, 'elapsed time = ',timer1,' seconds'
end 
        
