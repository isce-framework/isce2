!c  topocorrect - approximate topo correction
!c  Reference :
!http://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Appendix.pdf

    subroutine topo(demAccessor, dopAccessor, slrngAccessor)
      use topozeroState
      use topozeroMethods
      use geometryModule
      use orbitModule
      use linalg3Module
      use fortranUtils, ONLY : getPI

      implicit none
      include 'omp_lib.h'
      integer*8 demAccessor, dopAccessor, slrngAccessor
      integer lineFile, stat
      !integer width, length
      real*8, allocatable ::lat(:),lon(:),z(:),zsch(:)
      real*4, allocatable :: losang(:), incang(:), elevang(:)
      real*4, allocatable :: distance(:)
      real*8, allocatable :: rho(:), dopline(:)
      real*4, allocatable :: dem(:,:), demline(:)

      integer*1, allocatable :: mask(:), omask(:)
      real*8, allocatable :: orng(:), ctrack(:)
      real*8, allocatable :: oview(:)
      real*8 ctrackmin, ctrackmax, dctrack
      real*8 sch(3),xyz(3),llh(3),delta(3)

      real*8 tline, rng, dopfact
      real*8 llh_prev(3), xyz_prev(3)
      real*8 xyzsat(3), velsat(3)
      real*8 schsat(3), llhsat(3)
      real*8 ltpsat(3), ltpvel(3)
      real*8 enu(3)
      real*8 n_img(3), n_img_enu(3), n_trg_enu(3)
      real*8 that(3), chat(3), nhat(3), vhat(3)
      real*8 enumat(3,3), xyz2enu(3,3)

!!      real*8 xyz2(3), vxyz2(3)

      integer, allocatable :: converge(:)
      integer totalconv, owidth, ofactor

      real*8 height, rcurv, vmag
      real*8 aa, bb,cc
      real*8 r2d,refhgt,hnadir
      integer pixel
      integer nearrangeflag
      real*8 beta, alpha, gamm
      real*8 costheta,sintheta,cosalpha
      real*8 arg,rminoraxis,rlatg,st,ct
      real*8 fraclat, fraclon
      real*4 z1,z2,demlat,demlon
      real*4 demmax
      real*8 cospsi
      integer line,iter,ind
      integer idemlat,idemlon,i_type,i,j!,i_cnt1,i_cnt2,i_loff,i_el,i_sl

      !!!Variables for cropped DEM
      integer udemwidth, udemlength
      integer ustartx, uendx
      integer ustarty, uendy
      double precision umin_lon, umax_lon
      double precision umin_lat, umax_lat
      double precision ufirstlat, ufirstlon
      double precision hgts(2)

      real*8 pi
      integer,parameter :: b1=1
      integer,parameter :: b2=1
      integer binarysearch

      !!Geometry objects
      type(ellipsoidType) :: elp
      type(pegType) :: peg
      type(pegtransType) :: ptm

      procedure(intpTemplate), pointer :: intp_dem => null()
      procedure(interpolateOrbit_f), pointer :: intp_orbit => null()


        !!!Set up DEM interpolation method
        if (method.eq.SINC_METHOD) then
          intp_dem => intp_sinc
        else if (method.eq.BILINEAR_METHOD) then
          intp_dem => intp_bilinear
        else if (method.eq.BICUBIC_METHOD) then
          intp_dem => intp_bicubic
        else if (method.eq.NEAREST_METHOD) then
          intp_dem => intp_nearest
        else if (method.eq.AKIMA_METHOD) then
          intp_dem => intp_akima
        else if (method.eq.BIQUINTIC_METHOD) then
            intp_dem => intp_biquintic
        else
          print *, 'Undefined interpolation method.'
          stop
        endif
        call prepareMethods(method)


        !!!Set up orbit interpolation method
        if (orbitmethod .eq. HERMITE_METHOD) then
            intp_orbit => interpolateWGS84Orbit_f

            if(orbit%nVectors .lt. 4) then
                print *, 'Need atleast 4 state vectors for using hermite polynomial interpolation'
                stop
            endif
            print *, 'Orbit interpolation method: hermite'
        else if (orbitmethod .eq. SCH_METHOD) then
            intp_orbit => interpolateSCHOrbit_f

            if(orbit%nVectors .lt. 4) then
                print *, 'Need atleast 4 state vectors for using SCH interpolation'
                stop
            endif
            print *, 'Orbit interpolation method: sch'
        else if (orbitmethod .eq. LEGENDRE_METHOD) then
            intp_orbit => interpolateLegendreOrbit_f

            if(orbit%nVectors .lt. 9) then
                print *, 'Need atleast 9 state vectors for using legendre polynomial interpolation'
                stop
            endif
            print *, 'Orbit interpolation method: legendre'
        else
            print *, 'Undefined orbit interpolation method.'
            stop
        endif


        ofactor = 2
        owidth = ofactor*width + 1
        pi = getPI()
        hgts(1) = MIN_H
        hgts(2) = MAX_H

        lineFile  = 0

        totalconv = 0

        height = 0.0d0
        min_lat = 10000.
        max_lat = -10000.
        min_lon = 10000.
        max_lon = -10000.

!$omp parallel
        if(omp_get_thread_num().eq.1) then
            write(6,*), 'Max threads used: ', omp_get_num_threads()
        end if
!$omp end parallel

        if ((slrngAccessor.eq.0).and.(r0.eq.0.0d0)) then
            print *, 'Both the slant range accessor and starting range are zero'
            stop
        endif

!c  allocate variable arrays
        allocate (lat(width))
        allocate (lon(width))
        allocate (z(width))
        allocate (zsch(width))
        allocate (rho(width))
        allocate (dopline(width))
        allocate (distance(width))
        allocate (losang(2*width))
        allocate (incang(2*width))
        allocate (elevang(width))
        allocate (converge(width)) !!PSA

        if (maskAccessor.gt.0) then
            allocate (omask(owidth))
            allocate (orng(owidth))
            allocate (mask(width))
            allocate (ctrack(owidth))
            allocate (oview(owidth))
        endif

!c  some constants
        refhgt=0
        r2d=180.d0/pi
        elp%r_a = major
        elp%r_e2 = eccentricitySquared


        !!!PSA - Keep track of near range issues
        nearrangeflag = 0

        !!!Determining the bbox of interest
        !!!For detailed explanation of steps - see main loop below
        line=1
        !!!Doppler for geometry (not carrier) is const / range variant only
        call getLine_r8(dopAccessor, dopline, line)
        call getLine(slrngAccessor, rho, line)

        !!!First line
        do line=1,2
            tline = t0 + (line-1) * NAzlooks * (length-1.0d0)/prf
!!            stat = interpolateWGS84Orbit_f(orbit, tline, xyzsat, velsat)
            stat = intp_orbit(orbit, tline, xyzsat, velsat)
            if (stat.ne.0) then
                print *, 'Error getting statevector for bounds computation'
                exit
            endif
            vmag = norm(velsat)
            call unitvec(velsat, vhat)
            i_type = XYZ_2_LLH
            call latlon(elp, xyzsat, llhsat, i_type)
            height = llhsat(3)
            call tcnbasis(xyzsat, velsat, elp, that, chat, nhat)

            peg%r_lat = llhsat(1)
            peg%r_lon = llhsat(2)
            peg%r_hdg = peghdg
            call radar_to_xyz(elp, peg, ptm)
            rcurv = ptm%r_radcur


            do ind=1,2
                pixel = (ind-1)*(width-1) + 1
!                rng=r0 + (pixel-1) * Nrnglooks *rspace
                rng = rho(pixel)
                dopfact = (0.5d0 * wvl * dopline(pixel)/vmag) * rng

                do iter=1,2

                    !!PSA - SWOT specific near range check
                    !!If slant range vector doesn't hit ground, pick nadir point
                    if (rng .le. (llhsat(3)-hgts(iter)+1.0d0)) then
                        llh = llhsat
                        print *, 'Possible near nadir imaging'
                        nearrangeflag = 1
                    else
                        zsch(pixel) = hgts(iter)
                        aa =  height + rcurv
                        bb = rcurv + zsch(pixel)
                        costheta = 0.5*((aa/rng) + (rng/aa) - (bb/aa)*(bb/rng))
                        sintheta = sqrt(1.0d0 - costheta*costheta)
                        gamm = costheta * rng
                        alpha  = (dopfact - gamm * dot(nhat,vhat)) / dot(vhat,that)
                        beta  = -ilrl * sqrt(rng*rng*sintheta*sintheta - alpha*alpha)
                        delta = gamm * nhat + alpha * that + beta * chat
                        xyz = xyzsat + delta
                        i_type=XYZ_2_LLH
                        call latlon(elp,xyz,llh,i_type)
                    endif

                    min_lat = min(min_lat, llh(1)*r2d)
                    max_lat = max(max_lat, llh(1)*r2d)
                    min_lon = min(min_lon, llh(2)*r2d)
                    max_lon = max(max_lon, llh(2)*r2d)
                end do
            end do
        end do

        !!!Account for margins
        min_lon = min_lon - MARGIN
        max_lon = max_lon + MARGIN
        min_lat = min_lat - MARGIN
        max_lat = max_lat + MARGIN



        print *,'DEM parameters:'
        print *,'Dimensions: ',idemwidth,idemlength
        print *,'Top Left: ',firstlon,firstlat
        print *,'Spacing: ',deltalon,deltalat
        print *, 'Lon: ', firstlon, firstlon+(idemwidth-1)*deltalon
        print *, 'Lat: ', firstlat+(idemlength-1)*deltalat, firstlat

        print *, ' '
        print *, 'Estimated DEM bounds needed for global height range: '
        print *, 'Lon: ', min_lon, max_lon
        print *, 'Lat: ', min_lat, max_lat


        !!!!Compare with what has been provided as input
        umin_lon = max(min_lon, firstlon)
        if (min_lon .lt. firstlon) then
            print *, 'Warning: west limit may be insufficient for global height range'
        endif

        umax_lon = min(max_lon, firstlon + (idemwidth-1)*deltalon)
        if (max_lon .gt. (firstlon + (idemwidth-1)*deltalon)) then
            print *, 'Warning: east limit may be insufficient for global height range'
        endif

        umax_lat = min(max_lat, firstlat)
        if (max_lat .gt. firstlat) then
            print *, 'Warning: north limit may be insufficient for global height range'
        endif

        umin_lat = max(min_lat, firstlat + (idemlength-1)*deltalat)
        if (min_lat .lt. (firstlat + (idemlength-1)*deltalat)) then
            print *, 'Warning: south limit may be insufficient for global height range'
        endif



        !!!!Usable part of the DEM limits
        ustartx = int((umin_lon - firstlon)/deltalon)+1
        if (ustartx .lt. 1) ustartx = 1

        uendx = int((umax_lon-firstlon)/deltalon + 0.5d0)+1
        if (uendx.gt.idemwidth) uendx = idemwidth

        ustarty = int((umax_lat-firstlat)/deltalat)+1
        if (ustarty.lt.1) ustarty=1

        uendy = int((umin_lat-firstlat)/deltalat + 0.5) + 1
        if (uendy.gt.idemlength) ustarty=idemlength

        ufirstlon = firstlon + deltalon * (ustartx-1)
        ufirstlat = firstlat + deltalat * (ustarty-1)

        udemwidth = uendx - ustartx + 1
        udemlength = uendy - ustarty + 1

        print *, ' '
        print *, 'Actual DEM bounds used: '
        print *,'Dimensions: ',udemwidth,udemlength
        print *,'Top Left: ',ufirstlon,ufirstlat
        print *,'Spacing: ',deltalon,deltalat
        print *, 'Lon: ', ufirstlon, ufirstlon + deltalon*(udemwidth-1)
        print *, 'Lat: ', ufirstlat + deltalat * (udemlength-1), ufirstlat
        print *, 'Lines: ', ustarty, uendy
        print *, 'Pixels: ', ustartx, uendx

!c  allocate dem array
        allocate (dem(udemwidth,udemlength))
        allocate (demline(idemwidth))

        !!!Read the useful part of the DEM
        do j=1,udemlength
            lineFile = j + ustarty - 1
!            print *, 'Line: ', lineFile
            call getLine_r4(demAccessor,demline,lineFile)
            dem(:,j) = demline(ustartx:uendx)
        enddo

        demmax = maxval(dem)
        print *, 'Max DEM height: ', demmax

        print *, 'Primary iterations: ', numiter
        print *, 'Secondary iterations: ', extraiter
        print *, 'Distance threshold : ', thresh

        !!Initialize range values
!!        do pixel=1,width
!!           rho(pixel) = r0 + rspace*(pixel-1)*Nrnglooks
!!        enddo

        height = 0.0d0
        min_lat = 10000.
        max_lat = -10000.
        min_lon = 10000.
        max_lon = -10000.

      !!!File for debugging
!!      open(31, file='distance',access='direct',recl=4*width,form='unformatted')

      do line=1, length         !c For each line


         !!!!Set up the geometry
         !!Step 1: Get satellite position
         !!Get time
         tline = t0 + Nazlooks*(line - 1.0d0)/prf
         !!Get state vector

!!         stat = interpolateLegendreOrbit_f(orbit, tline, xyz2, vxyz2)
!!         print *, 'Line: ', line
!!         print *, tline, xyz2, vxyz2
!!         stat = interpolateWGS84Orbit_f(orbit, tline, xyzsat, velsat)
         stat = intp_orbit(orbit, tline, xyzsat, velsat)
!!         print *, tline, xyzsat, velsat

         call unitvec(velsat, vhat)
         vmag = norm(velsat)
         !!vhat is unit vector along velocity
         !!vmag is the magnitude of the velocity



         !!Step 2: Get local radius of curvature along heading
         !!Convert satellite position to lat lon
         i_type = XYZ_2_LLH
         call latlon(elp, xyzsat, llhsat, i_type)
         height = llhsat(3)

!!         print *, 'Sat pos: ', line
!!         print *, llhsat(1)*r2d, llhsat(2)*r2d, llhsat(3)

         !!Step 3: Get TCN basis using satellite basis
         call tcnbasis(xyzsat, velsat, elp, that, chat, nhat)
         !!that is along local tangent to the planet
         !!chat is along the cross track direction
         !!nhat is along the local normal

         !!Step 4: Get Doppler information for the line
         !! For native doppler, this corresponds to doppler polynomial
         !! For zero doppler, its a constant zero polynomial
         call getLineSequential(dopAccessor, dopline, i_type)
!!         print *, 'VEL:', velsat
!!         print *, 'TCN:', xyzsat
!!         print *, that
!!         print *, chat
!!         print *, nhat
!!         print *, vhat
         !!Get the slant range
         call getLineSequential(slrngAccessor, rho, i_type)

         !!Step 4: Set up SCH basis right below the satellite
         peg%r_lat = llhsat(1)
         peg%r_lon = llhsat(2)
         peg%r_hdg = peghdg
         hnadir = 0.0d0

!!         print *, 'Heading: ', peghdg
         call radar_to_xyz(elp, peg, ptm)
         rcurv = ptm%r_radcur
         converge = 0
         z = 0.
         zsch = 0.

         if (mod(line,1000).eq.1) then
             print *, 'Processing line: ', line, vmag
             print *, 'Dopplers: ', dopline(1), dopline(width/2), dopline(width)
         endif

        !!Initialize lat,lon to middle of input DEM
        lat(:) = ufirstlat + 0.5d0*deltalat*udemlength
        lon(:) = ufirstlon + 0.05d0*deltalon*udemwidth


        !!PSA - SWOT specific near range check
        !!Computing nadir height
        if (nearrangeflag .ne. 0) then
            demlat=(llhsat(1)*r2d-ufirstlat)/deltalat+1
            demlon=(llhsat(2)*r2d-ufirstlon)/deltalon+1
            if(demlat.lt.1)demlat=1
            if(demlat.gt.udemlength-1)demlat=udemlength-1
            if(demlon.lt.1)demlon=1
            if(demlon.gt.udemwidth-1)demlon=udemwidth-1

            !!!!! This whole part can be put into a function
            idemlat=int(demlat)
            idemlon=int(demlon)
            fraclat=demlat-idemlat
            fraclon=demlon-idemlon
            hnadir = intp_dem(dem,idemlon,idemlat,fraclon,fraclat,udemwidth,udemlength)
        endif

         !!!!Start the iterations
         do iter=1,numiter+extraiter+1

            !$omp parallel do private(pixel,sch,beta,alpha,gamm) &
            !$omp private(i_type,llh,idemlat,idemlon,xyz,arg) &
            !$omp private(z1,z2,fraclat,fraclon,demlat,demlon) &
            !$omp private(llh_prev,xyz_prev,aa,bb,cc, rng) &
            !$omp private(costheta,sintheta,delta,dopfact)&
            !$omp shared(ufirstlat,ufirstlon,deltalat,deltalon)&
            !$omp shared(xyzsat,that,chat,nhat,vhat,peg,ptm)&
            !$omp shared(length,width,Nazlooks,height,r2d,dem) &
            !$omp shared(rcurv,rho,elp,lat,lon,z,zsch,line)&
            !$omp shared(extraiter,ilrl,iter,dopline,vmag,hnadir) &
            !$omp shared(distance,converge,thresh,numiter)&
            !$omp shared(udemwidth,udemlength,totalconv,wvl)
            do pixel=1,width
               rng = rho(pixel)
               dopfact = (0.5d0 * wvl * dopline(pixel)/vmag) * rng

               !!PSA - Check for near range issues
!!                if (nearrangeflag .ne. 0) then
!!                    if (rng .le. (llhsat(2)-hnadir+1.0d0)) then
!!                endif



               !! If pixel hasnt converged
               if(converge(pixel).eq.0) then

                   !!!!Use previous llh in degrees and meters
                   llh_prev(1) = lat(pixel)/r2d
                   llh_prev(2) = lon(pixel)/r2d
                   llh_prev(3) = z(pixel)

!!                  print *, 'ITER: ', iter
!!                  print *, 'PREV: ', lat(pixel), lon(pixel), z(pixel)

                  !!!!Solve for new position at height zsch
                  aa =  height + rcurv
                  bb = rcurv + zsch(pixel)

!!                  print *, aa, bb, rng
                  !!!!Normalize reasonably to avoid overflow
                  costheta = 0.5*((aa/rng) + (rng/aa) - (bb/aa)*(bb/rng))
                  sintheta = sqrt(1.0d0 - costheta*costheta)

!!                  print *, costheta, sintheta
                  !!Components along unit vectors

                  !!Vector from satellite to point on ground can be written as
                  !! vec(dr) = alpha * vec(that) + beta * vec(chat) + gamma *
                  !! vec(nhat)
                  gamm = costheta * rng
                  alpha  = (dopfact - gamm * dot(nhat,vhat)) / dot(vhat,that)
                  beta  = -ilrl * sqrt(rng*rng*sintheta*sintheta - alpha*alpha)
!!                  print *, alpha, beta, gamm

                  !!! xyz position of target
                  delta = gamm * nhat + alpha * that + beta * chat
                  xyz = xyzsat + delta

                  i_type=XYZ_2_LLH
                  call latlon(elp,xyz,llh,i_type)

!!                  print *, 'NOW:', llh(1)*r2d, llh(2)*r2d, llh(3)
                  !c  convert lat, lon, hgt to xyz coordinates
                  lat(pixel)=llh(1)*r2d
                  lon(pixel)=llh(2)*r2d
                  demlat=(lat(pixel)-ufirstlat)/deltalat+1
                  demlon=(lon(pixel)-ufirstlon)/deltalon+1
                  if(demlat.lt.1)demlat=1
                  if(demlat.gt.udemlength-1)demlat=udemlength-1
                  if(demlon.lt.1)demlon=1
                  if(demlon.gt.udemwidth-1)demlon=udemwidth-1

                  !!!!! This whole part can be put into a function
                  idemlat=int(demlat)
                  idemlon=int(demlon)
                  fraclat=demlat-idemlat
                  fraclon=demlon-idemlon
!!!                  z1=dem(idemlon,idemlat)*(1-fraclon)+dem(idemlon+1,idemlat)*fraclon
!!!                  z2=dem(idemlon,idemlat+1)*(1-fraclon)+dem(idemlon+1,idemlat+1)*fraclon
                  !!!Can change this to Akima
!!!                  z(pixel)=z1*(1-fraclat)+z2*fraclat

                  z(pixel) = intp_dem(dem,idemlon,idemlat,fraclon,fraclat,udemwidth,udemlength)
                  !!!!!! This whole part can be put into a function



                  if(z(pixel).lt.-500.0)z(pixel)=-500.0

                  ! given llh, where h = z(pixel,line) in WGS84, get the SCH height
                  llh(1) = lat(pixel)/r2d
                  llh(2) = lon(pixel)/r2d
                  llh(3) = z(pixel)

!!                  print *, 'UPDATED: ', lat(pixel), lon(pixel), z(pixel)
                  i_type = LLH_2_XYZ
                  call latlon(elp,xyz,llh,i_type)

                  i_type = XYZ_2_SCH
                  call convert_sch_to_xyz(ptm,sch,xyz,i_type)
                  ! print *, 'after = ', sch
!!                  print *, 'ZSCH:' , zsch(pixel), sch(3)
                  zsch(pixel) = sch(3)

                  !!!!Absolute distance
                  distance(pixel) = sqrt((xyz(1)-xyzsat(1))**2 +(xyz(2)-xyzsat(2))**2 + (xyz(3)-xyzsat(3))**2) - rng
!!                  print *, 'DIST: ', distance(pixel)
                  if(abs(distance(pixel)).le.thresh) then
                     zsch(pixel) = sch(3)
                     converge(pixel) = 1
                     totalconv = totalconv+1

                 else if(iter.gt.(numiter+1)) then

                    i_type=LLH_2_XYZ
                    call latlon(elp, xyz_prev,llh_prev,i_type)

                     xyz(1) = 0.5d0*(xyz_prev(1)+xyz(1))
                     xyz(2) = 0.5d0*(xyz_prev(2)+xyz(2))
                     xyz(3) = 0.5d0*(xyz_prev(3)+xyz(3))

                     !!!!Repopulate lat,lon,z
                     i_type=XYZ_2_LLH
                     call latlon(elp,xyz,llh,i_type)
                     lat(pixel) = llh(1)*r2d
                     lon(pixel) = llh(2)*r2d
                     z(pixel) = llh(3)

                     i_type=XYZ_2_SCH
                     call convert_sch_to_xyz(ptm,sch,xyz,i_type)
                     zsch(pixel) = sch(3)
                     !!!!Absolute distance
                     distance(pixel) = sqrt((xyz(1)-xyzsat(1))**2 +(xyz(2)-xyzsat(2))**2 + (xyz(3)-xyzsat(3))**2) - rng
                 endif
             endif

            end do
            !$omp end parallel do

         end do


         !!!!Final computation.
         !!!! The output points are exactly at range pixel
         !!!!distance from the satellite


         !$omp parallel do private(pixel,cosalpha) &
         !$omp private(xyz,llh,delta,rng,i_type,sch,aa,bb) &
         !$omp private(n_img,n_img_enu,n_trg_enu,cospsi) &
         !$omp private(alpha,beta,gamm,costheta,sintheta,dopfact) &
         !$omp private(enumat,xyz2enu,enu) &
         !$omp private(demlat,demlon,idemlat,idemlon,fraclat,fraclon)&
         !$omp shared(zsch,line,rcurv,rho,height,losang,width,velsat) &
         !$omp shared(peghdg,r2d,ilrl,lat,lon,z,xyzsat,distance,incang)&
         !$omp shared(elp,ptm,that,chat,vhat,nhat,vmag,dopline,wvl,dem)&
         !$omp shared(udemwidth,udemlength,ufirstlat,ufirstlon)&
         !$omp shared(deltalat,deltalon,elevang)
         do pixel=1,width

            rng = rho(pixel)
            dopfact = (0.5d0 * wvl * dopline(pixel)/vmag) * rng

            !!!!Solve for new position at height zsch
            aa =  height + rcurv
            bb = rcurv + zsch(pixel)

            costheta = 0.5*((aa/rng) + (rng/aa) - (bb/aa)*(bb/rng))
            sintheta = sqrt(1.0d0 - costheta*costheta)

            gamm = costheta * rng
            alpha  = (dopfact -gamm * dot(nhat,vhat)) / dot(vhat,that)
            beta  = -ilrl * sqrt(rng*rng*sintheta*sintheta - alpha*alpha)

            !!! xyz position of target
            delta = gamm * nhat + alpha * that + beta * chat
            xyz = xyzsat + delta

            i_type=XYZ_2_LLH
            call latlon(elp,xyz,llh,i_type)

            !!!!Copy into output arrays
            lat(pixel) = llh(1)*r2d
            lon(pixel) = llh(2)*r2d
            z(pixel)   = llh(3)

!!            distance(pixel) = ((xyz(1)-xyzsat(1))* velsat(1)+(xyz(2)-xyzsat(2))*velsat(2) + (xyz(3)-xyzsat(3))*velsat(3)) - dopfact * vmag
            distance(pixel) = sqrt((xyz(1)-xyzsat(1))**2 + (xyz(2)-xyzsat(2))**2 + (xyz(3)-xyzsat(3))**2) - rng

            !!!Computations in ENU coordinates around target
            call enubasis(llh(1), llh(2), enumat)
            xyz2enu = transpose(enumat)
            enu = matmul(xyz2enu,delta)

            cosalpha = abs(enu(3)) / norm(enu)

            !!!!LOS vectors
            losang(2*pixel-1) = acos(cosalpha)*r2d
            losang(2*pixel) = (atan2(-enu(2), -enu(1))-0.5*pi)*r2d
            elevang(pixel) = acos(costheta)*r2d

            !!!ctrack gets stored in zsch
            zsch(pixel) = rng * sintheta

            !!!!Get local incidence angle

            demlat=(lat(pixel)-ufirstlat)/deltalat+1
            demlon=(lon(pixel)-ufirstlon)/deltalon+1
            if(demlat.lt.2)demlat=2
            if(demlat.gt.udemlength-1)demlat=udemlength-1
            if(demlon.lt.2)demlon=2
            if(demlon.gt.udemwidth-1)demlon=udemwidth-1

            !!!!! This whole part can be put into a function
            idemlat=int(demlat)
            idemlon=int(demlon)
            fraclat=demlat-idemlat
            fraclon=demlon-idemlon

            !!!Slopex
            aa = intp_dem(dem,idemlon-1,idemlat,fraclon,fraclat,udemwidth,udemlength)
            bb  = intp_dem(dem,idemlon+1,idemlat,fraclon,fraclat,udemwidth,udemlength)
            gamm = lat(pixel)/r2d
            alpha = (bb-aa)* r2d / (2.0d0 * reast(elp, gamm) * deltalon)

            !!!Slopey
            aa = intp_dem(dem,idemlon,idemlat-1,fraclon,fraclat,udemwidth,udemlength)
            bb  = intp_dem(dem,idemlon,idemlat+1,fraclon,fraclat,udemwidth,udemlength)
            beta = (bb-aa)*r2d/(2.0d0 * rnorth(elp,gamm)*deltalat)

            enu = enu / norm(enu)
            costheta = (enu(1)*alpha + enu(2)*beta-enu(3))/sqrt(1.0d0+alpha*alpha+beta*beta)
            incang(2*pixel) = acos(costheta)*r2d

            !!!! Calculate psi angle between image plane and local slope
            call cross(delta, velsat, n_img)
            call unitvec(n_img, n_img)
            n_img_enu = matmul(xyz2enu, -ilrl*n_img)
            n_trg_enu = [-alpha, -beta, 1.0d0]
            cospsi = dot(n_trg_enu, n_img_enu) / (norm(n_trg_enu)*norm(n_img_enu))
            incang(2*pixel-1) = acos( cospsi )*r2d

            !!! Temporary hack needed by dense baseline in the
            !!! derivative computation. Todo: create two new layers
            !incang(2*pixel-1) = alpha                                                                                                                                                            !incang(2*pixel) = beta


         end do
         !$omp end parallel do


         !c Maybe add hmin and hmax?
         min_lat = min(minval(lat), min_lat)
         max_lat = max(maxval(lat), max_lat)
         min_lon = min(minval(lon), min_lon)
         max_lon = max(maxval(lon), max_lon)
!!         write(31,rec=line)(distance(j),j=1,width)
         call setLineSequential_r8(latAccessor, lat)
         call setLineSequential_r8(lonAccessor, lon)
         call setLineSequential_r8(heightAccessor, z)
         if(losAccessor.gt.0) then
             call setLineSequential_r4(losAccessor,losang)
         endif

         if (incAccessor.gt.0) then
             call setLineSequential_r4(incAccessor, incang)
         endif


         if (maskAccessor.gt.0) then
             ctrackmin = minval(zsch) - demmax
             ctrackmax = maxval(zsch) + demmax
             dctrack = (ctrackmax-ctrackmin)/(owidth-1.0d0)

             !!!Sort lat / lon by ctrack
             call InsertionSort(zsch, lat, lon, width)

             !!!Interpolate heights to regular ctrack grid

             !$omp parallel do private(pixel,llh,xyz,rng,aa,bb,i_type)&
             !$omp private(demlat,demlon,idemlat,idemlon,fraclat,fraclon)&
             !$omp shared(ctrackmin,ctrackmax,dctrack,dem,r2d)&
             !$omp shared(orng,owidth,lat,lon,ufirstlat,ufirstlon)&
             !$omp shared(deltalat,deltalon,udemlength,udemwidth)&
             !$omp shared(xyzsat,elp,ctrack,oview,nhat)
             do pixel=1,owidth
                aa = ctrackmin + (pixel-1)*dctrack
                ctrack(pixel) = aa
                i_type = binarysearch(zsch, width, aa)
!!                print *, line, pixel, aa, i_type
                if (i_type.eq.width) i_type = width-1
                if (i_type.eq.0) i_type=1

                !!!!Simple bi-linear interpolation
                fraclat = (aa - zsch(i_type)) / (zsch(i_type+1) - zsch(i_type))
                demlat = lat(i_type) + fraclat*(lat(i_type+1)-lat(i_type))
                demlon = lon(i_type) + fraclat*(lon(i_type+1)-lon(i_type))

                llh(1) = demlat/r2d
                llh(2) = demlon/r2d

                demlat=(demlat-ufirstlat)/deltalat+1
                demlon=(demlon-ufirstlon)/deltalon+1
                if(demlat.lt.2)demlat=2
                if(demlat.gt.udemlength-1)demlat=udemlength-1
                if(demlon.lt.2)demlon=2
                if(demlon.gt.udemwidth-1)demlon=udemwidth-1

                !!!!! This whole part can be put into a function
                idemlat=int(demlat)
                idemlon=int(demlon)
                fraclat=demlat-idemlat
                fraclon=demlon-idemlon
                llh(3) = intp_dem(dem,idemlon,idemlat,fraclon,fraclat,udemwidth,udemlength)
                i_type=LLH_2_XYZ
                call latlon(elp,xyz,llh,i_type)

                xyz = xyz - xyzsat
                bb = norm(xyz)
                orng(pixel) = bb
                aa = abs(sum(nhat*xyz))
                oview(pixel) = acos(aa/bb)*r2d
            end do
            !$omp end parallel do


            !!!Again sort in terms of slant range
            call InsertionSort(orng, ctrack, oview, owidth)

            mask = 0
            omask = 0
            aa = elevang(1)
            do pixel=2,width
                bb=elevang(pixel)
                if (bb.le.aa) then
                    mask(pixel) = 1
                else
                    aa = bb
                endif
            end do

            aa = elevang(width)
            do pixel=width-1,1,-1
                bb = elevang(pixel)
                if (bb.ge.aa) then
                    mask(pixel) = 1
                else
                    aa = bb
                endif
            end do

            !!!!If we wanted to work with shadow
            !!!!in cross track sorted coords
            !aa = oview(1)
            !do pixel=2,owidth
                !bb = oview(pixel)
                !if (bb.le.aa) then
                    !omask(pixel) = 1
                !else
                    !aa = bb
                !endif
            !enddo

            !aa = oview(width)
            !do pixel=width-1,1,-1
                !bb = oview(pixel)
                !if (bb.ge.aa) then
                    !omask(pixel) = 1
                !else
                    !aa = bb
                !endif
            !end do


            aa = ctrack(1)
            do pixel=2,width
                bb = ctrack(pixel)
                if ((bb.le.aa).and.(omask(pixel).lt.2)) then
                    omask(pixel) = omask(pixel) + 2
                else
                    aa = bb
                endif
            end do

            aa = ctrack(owidth)
            do pixel=owidth-1,1,-1
                bb = ctrack(pixel)
                if ((bb.ge.aa).and.(omask(pixel).lt.2)) then
                    omask(pixel) = omask(pixel) + 2
                else
                    aa = bb
                endif
            end do


            do pixel=1, owidth
                if (omask(pixel).gt.0) then
!!                    idemlat = nint((orng(pixel) - r0)/ (rspace * Nrnglooks))+1
                    idemlat = binarysearch(rho, width, orng(pixel))
                    if ((idemlat.ge.1) .and. (idemlat.le.width)) then
                        if (mask(idemlat) .lt. omask(pixel)) then
                            mask(idemlat) = mask(idemlat) + omask(pixel)
                        endif
                    endif
                endif
            enddo

            !!!!!If using shadow from ctrack coords
            !do pixel=1, owidth
                !if (omask(pixel).gt.0) then
                    !idemlat = nint((orng(pixel) - r0)/ (rspace * Nrnglooks))+1
                    !if ((idemlat.ge.1) .and. (idemlat.le.width)) then
                            !mask(idemlat) = omask(pixel)
                    !endif
                !endif
            !enddo



            call setLineSequential(MaskAccessor, mask)
        endif
      end do

      print *, 'Total convergence:', totalconv, ' out of ', width*length
      call unprepareMethods(method)
!!       close(31)

       if (maskAccessor.gt.0) then
          deallocate(omask)
          deallocate(orng)
          deallocate(mask)
          deallocate(ctrack)
          deallocate(oview)
       endif

       deallocate (demline)
       deallocate (converge)
       deallocate (distance)
       deallocate (lat)
       deallocate (lon)
       deallocate (z)
       deallocate (zsch)
       deallocate (rho)
       deallocate (dem)
       deallocate (losang)
       deallocate (incang)
       deallocate (elevang)
      end


      SUBROUTINE InsertionSort(a,b,c,num)
        REAL*8, DIMENSION(num) :: a,b,c
        REAL*8 :: tempa,tempb,tempc
        INTEGER :: i, j, num

        DO i = 2, num
            j = i - 1
            tempa = a(i)
            tempb = b(i)
            tempc = c(i)
            DO WHILE (j>=1 .AND. a(j)>tempa)
                a(j+1) = a(j)
                b(j+1) = b(j)
                c(j+1) = c(j)
                j = j - 1
            END DO
            a(j+1) = tempa
            b(j+1) = tempb
            c(j+1) = tempc
        END DO
     END SUBROUTINE InsertionSort


    function binarysearch(array, length, val)
        implicit none
        integer :: length
        real*8, dimension(length) :: array
        real*8 :: val

        integer :: binarysearch, ind

        integer :: left, middle, right


        left = 1
        right = length
        do
            if (left > right) then
                exit
            endif
            middle = nint((left+right) / 2.0)

            if (left .eq. (right-1)) then
                binarySearch = left
                return
            elseif (array(middle).le.val) then
                left = middle
            elseif (array(middle).gt.val) then
                right = middle
            end if
        end do

        binarysearch = left
    end function binarysearch
