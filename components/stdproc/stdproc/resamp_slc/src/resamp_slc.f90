!c***************************************************************
      subroutine resamp_slc(slcInAccessor,slcOutAccessor,residazAccessor,residrgAccessor)

      use fortranUtils, ONLY:getPI
      use resamp_slcState
      use resamp_slcMethods

      implicit none
      include 'omp_lib.h'

!c    PARAMETER STATEMENTS:

      integer*8 slcInAccessor,slcOutAccessor
      integer*8 residrgAccessor, residazAccessor
      integer lineNum

      real*8 PI

      integer istats, iflatten
      integer ist, nr, naz, i_numpnts
      integer i, j, k
      integer int_az_off
      integer i_na
      integer ith, thnum, ithorig
        
      integer ii, jj
      integer chipi, chipj
      real*8 r_ro, r_ao, r_rt, r_at, r_ph, r_dop
      
      real*4 t0, t1 

      real*8 r_azcorner,r_racorner,fracr,fraca

      complex, allocatable, dimension(:,:) :: cin
      complex, allocatable, dimension(:) :: cout
      complex, allocatable, dimension(:) :: cline
      complex, allocatable, dimension(:,:,:) :: chip 

      complex cval
      real*4, allocatable, dimension(:,:) :: rin
      real*4, allocatable, dimension(:) :: rout

      real*8, allocatable, dimension(:) :: residaz
      real*8, allocatable, dimension(:) :: residrg

      integer kk,ifrac

!c    PROCESSING STEPS:

      PI = getPI()

      iscomplex = 1
      t0 = secnds(0.0)

      print *, ' '       
      print *,  ' << Resample one image to another image coordinates >>'
      print *, ' ' 

      print *, 'Input Image Dimensions: '
      print *, inwidth, ' pixels'
      print *, inlength, 'lines'

      print *, ' '
      print *, 'Output Image Dimensions: '
      print *, outwidth, 'pixels'
      print *, outlength, 'lines'
      print *, '  '

      istats=0

      if ((iscomplex.ne.0) .and. (method.ne.SINC_METHOD)) then
          print *, 'WARNING!!!'
          print *, 'Currently Only Sinc interpolation is available for complex data.'
          print *, 'Setting interpolation method to sinc'
          method = SINC_METHOD
      endif


      !$OMP PARALLEL
      !$OMP MASTER
      ith = omp_get_num_threads()
      !$OMP END MASTER
      !$OMP END PARALLEL

      ithorig = ith
      ith = min(ith,8)
      print *, 'Number of threads: ', ith
      call omp_set_num_threads(ith)

!c  allocate the big arrays
      if (iscomplex.ne.0) then
        allocate (cin(inwidth,inlength))
        allocate (cout(outwidth))
        allocate (cline(inwidth))
        allocate (chip(sincone,sincone,ith))
        print *, 'Complex data interpolation'
      else
        allocate (rin(inwidth,inlength))
        allocate (rout(outwidth))
        print *, 'Real data interpolation'
      endif


      allocate(residaz(outwidth))
      allocate(residrg(outwidth))

      call prepareMethods(method)
        
      print *, 'Azimuth Carrier Poly'
      call printpoly2d_f(azCarrier)

      print *, 'Range Carrier Poly'
      call printpoly2d_f(rgCarrier)

      print *, 'Range offsets poly'
      call printpoly2d_f(rgOffsetsPoly)

      print *, 'Azimuth offsets poly'
      call printpoly2d_f(azOffsetsPoly)


      print *, 'Reading in the image'
!c  read in the master image
      if (iscomplex.ne.0) then
        lineNum = 1

        !!!!All carriers are removed from the data up front
        do j = 1,inlength
            call getLineSequential(slcInAccessor,cline,lineNum)
            r_at = j

            !$OMP PARALLEL DO private(i,r_rt,r_ph)&
            !$OMP shared(inwidth,r_at,cin,cline)&
            !$OMP shared(rgCarrier,azCarrier,j)
            do i = 1,inwidth
                r_rt = i
                r_ph = evalPoly2d_f(rgCarrier, r_at, r_rt) + evalPoly2d_f(azCarrier,r_at,r_rt)
                r_ph = modulo(r_ph,2.0d0*PI)
                cin(i,j) = cline(i) * cmplx(cos(r_ph), -sin(r_ph))
            enddo
            !$OMP END PARALLEL DO

            if (mod(j,1000).eq.0) then
                print *, 'At line ', j
            endif

        enddo
      else
        lineNum=1
        do j = 1,inlength
            call getLineSequential(slcInAccessor, rin(:,j), lineNum)

            if (mod(j,1000).eq.0) then
                print *, 'At line ',j
            endif
        enddo
      endif

      residaz = 0.
      residrg = 0.

!c  loop over lines
       print *, 'Interpolating image'

       !!!!Interpolation of complex images
       if (iscomplex.ne.0) then
          do j=1,outlength
            if(mod(j,1000).eq.0) then
               print *,'At line ',j
            end if

            if(residazAccessor .ne. 0) then
              call getLineSequential(residAzAccessor, residaz, lineNum)
            endif

            if(residRgAccessor .ne. 0) then
              call getLineSequential(residRgAccessor, residrg, lineNum)
            endif
          
            cout=cmplx(0.,0.)

            !!!Start of the parallel loop
            !$OMP PARALLEL DO private(i,r_rt,r_at,r_ro,r_ao,k,kk)&
            !$OMP private(fracr,fraca,ii,jj,r_ph,cval,thnum,r_dop) &
            !$OMP private(chipi,chipj) &
            !$OMP shared(rgOffsetsPoly,azOffsetsPoly,residrg,residaz) &
            !$OMP shared(j,cin,chip,cout,flatten,WVL,SLR,inlength) &
            !$OMP shared(rgCarrier,azCarrier,outwidth,inwidth,dopplerPoly)&
            !$OMP shared(REFR0, REFSLR, R0, REFWVL)
            do i=1,outwidth
                
               !!!Get thread number
               thnum = omp_get_thread_num() + 1

               r_rt=i
               r_at=j

               r_ro = evalPoly2d_f(rgOffsetsPoly,r_at,r_rt) + residrg(i)
               r_ao = evalPoly2d_f(azOffsetsPoly,r_at,r_rt) + residaz(i)


               k=int(i+r_ro)   !range offset
               fracr=i+r_ro-k

               if ((k .le. sinchalf) .or. (k.ge.(inwidth-sinchalf))) then
                 cycle
               endif

               kk=int(j+r_ao)  !azimuth offset
               fraca=j+r_ao-kk

               if ((kk .le. sinchalf) .or. (kk.ge.(inlength-sinchalf))) then
                 cycle
               endif

               r_dop = evalPoly2d_f(dopplerPoly, r_at, r_rt)

               !!!!!!Data chip without the carriers
               do jj=1,sincone
                  chipj = kk + jj - 1 - sinchalf
                  cval = cmplx(cos((jj-5.0d0)*r_dop),-sin((jj-5.0d0)*r_dop))
                  do ii=1,sincone
                     chipi = k + ii - 1 - sinchalf

                     !!!Take out doppler in azimuth
                     chip(ii,jj,thnum) = cin(chipi,chipj)*cval
                  end do
               end do
    
               !!!Doppler to be added back
               r_ph = r_dop*fraca

               !!Evaluate carrier that needs to be added back after interpolation
               r_rt = i+r_ro
               r_at = j+r_ao
               r_ph = r_ph + evalPoly2d_f(rgCarrier, r_at, r_rt) + evalPoly2d_f(azCarrier,r_at,r_rt)

               if (flatten.ne.0) then
                   r_ph = r_ph + (4.0d0 * PI/WVL) * ((R0-REFR0) + (i-1.0d0)*(SLR-REFSLR) +  r_ro*SLR) + (4.0d0*PI*(REFR0+(i-1.0d0)*REFSLR)) * (1.0d0/REFWVL - 1.0d0/WVL)
               endif

               r_ph = modulo(r_ph,2.0d0*PI)

               jj = sinchalf+1
               ii = sinchalf+1

               cval = intp_sinc_cx(chip(1:sincone,1:sincone,thnum),ii,jj,fracr,fraca,sincone,sincone)

               cout(i)=cval * cmplx(cos(r_ph), sin(r_ph))

            end do
            !$OMP END PARALLEL DO

            call setLineSequential(slcOutAccessor,cout)
        enddo

     !!!!!Interpolation of real images
     else


           print *, 'Real data interpolation not implemented yet.'
      
     endif



!cc XXX End of line loop

      t1 = secnds(t0)
      print *, 'Elapsed time: ', t1

      call unprepareMethods(method)

      deallocate(residaz)
      deallocate(residrg)

      if (iscomplex .ne. 0) then
          deallocate(cin)
          deallocate(cout)
          deallocate(cline)
          deallocate(chip)
      else
          deallocate(rin)
          deallocate(rout)
      endif

      !Reset number of threads
      call omp_set_num_threads(ithorig)
      end
      

