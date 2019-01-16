!c  offset  - offset between two band images
!c     from ampoffset -  estimate offsets in two complex images by 
!c     cross-correlating magnitudes
!c     modified 24apr98 to use fft correlations rather than time-domain
!c     can also accommodate differing prfs
      subroutine estimateoffsets(band1Accessor,band2Accessor)      
      use estimateoffsetsState
      use estimateoffsetsRead
      implicit none

      integer*8 band1Accessor, band2Accessor
      real cmax,amean,cave,a2mean
      real delta,dsampdn,dsampac
      real peak,snr,snr_min,offac,offdn
      integer i,ipeak,ioff,idnloc,iip,ii
      integer irec,n,jj,jpeak,joff,jjp
      integer linedelta,koff,k,kk,loff
      integer j,ndnloc,irow,  icnt

!!Make the arrays allocatable
      complex, allocatable :: a(:,:), aa(:,:)
      complex, allocatable :: b(:,:), bb(:,:)
      complex, allocatable :: cpa(:,:), cpb(:,:)
      complex, allocatable :: corr(:,:), corros(:,:)
      real, allocatable :: pa(:,:), pb(:,:)
      real, allocatable :: c(:,:)
      integer, allocatable :: ic(:,:)
      real, allocatable :: red(:,:), green(:,:)
      complex, allocatable :: cdata(:)
      real :: normfactor1, normfactor2

      integer dsamp,ilin1(0:16384)
      logical ex
      integer statb(13),stat
      integer*8 nbytes,filelen
      logical isnan
      integer lenbig

      !Interface statements for reading cpx or real bands
       interface
            subroutine readTemplate(acc,arr,irow,band,n,carr)
                integer*8 :: acc
                complex, dimension(:) :: carr
                real, dimension(:)  :: arr
                integer :: irow,band,n
            end subroutine readTemplate
       end interface

       procedure(readTemplate), pointer :: readBand1 => null()
       procedure(readTemplate), pointer :: readBand2 => null()


!Piyush's allocation statements
      allocate(a(NPTS, NPTS))       !Data from file
      allocate(aa(2*NPTS, 2*NPTS))  !Oversampled by 2
      allocate(b(2*NPTS, 2*NPTS))   !Data from file
      allocate(bb(4*NPTS, 4*NPTS))  !Oversampled by 2
      allocate(cpa(4*NPTS, 4*NPTS)) !Amplitude for channel 1
      allocate(cpb(4*NPTS, 4*NPTS)) !Amplitude for channel 2
      allocate(corr(NDISP, NDISP))  !Window around the maximum
      allocate(corros(NDISP*NOVS, NDISP*NOVS)) !Oversampled around maximum 
      allocate(pa(2*NPTS, 2*NPTS))   !Real valued amplitude
      allocate(pb(4*NPTS, 4*NPTS))   !Real valued amplitude
      allocate(c(-NOFF:NOFF,-NOFF:NOFF))
      allocate(ic(-NOFF-NDISP:NOFF+NDISP,-NOFF-NDISP:NOFF+NDISP))


      allocate(red(len1,NPTS*2))       !Amplitude from File 1
      allocate(green(len2,NPTS*2))     !Amplitude from File 2

      !Correct readers for each band
      if(iscpx1.eq.1) then 
          readBand1 => readCpxAmp
          print *, 'Setting first image to complex', band1
      else
          readBand1 => readAmp
          print *, 'Setting first image to real', band1
      endif

      if(iscpx2.eq.1) then
          readBand2 => readCpxAmp
          print *, 'Setting second image to complex', band2
      else
          readBand2 => readAmp
          print *, 'Setting second image to real', band2
      endif

      if(talk.eq.'y') then
          print *,'** RG offsets from cross-correlation **'
          print *,' Capture range is +/- ',NOFF/2,' pixels'
          print *,' Initializing ffts'
      endif

      !c Set up FFT plans
      do i=3,14
         k=2**i
         call cfft1d_jpl(k,a,0)
      end do



      dsampac=float(isamp_f-isamp_s)/float(nloc-1)
      print *,'across step size: ',dsampac
      dsamp=dsampac
      if(dsampac-dsamp.ge.1.e10)
     +   print *,'Warning: non-integer across sampling'

      dsampdn=float(isamp_fdn-isamp_sdn)/float(nlocdn-1)
      print *,'down step size: ',dsampdn

      ndnloc=nlocdn
      do j=0,ndnloc-1
         ilin1(j)=isamp_sdn+j*dsampdn
      end do
      
      snr_min=2.
      
      delta=(1./prf1-1./prf2)*prf1


      print *,'Input lines:',lines1, lines2
      print *,'Input bands:', band1, band2
      print *,'Input widths:', len1, len2

      lenbig = max(len1, len2)
      allocate(cdata(lenbig))

!c  loop over line locations
      icnt = 0
      do idnloc=0,ndnloc-1
         if(mod(idnloc,10).eq.0) then
            print *,'On line, location ',idnloc,ilin1(idnloc)
         endif

         if(talk.eq.'y') then
             print *
             print *,'down file 1: ', ilin1(idnloc)
         endif

!c  read in the data to data array
         irec=ilin1(idnloc)-NPTS/2-1 !offset in down
!!         print *, 'refLineStart: ', irec
         red = 0.0
         do j=1,NPTS*2
            i=band1
            irow = irec + j
            call readBand1(band1Accessor,red(:,j),irow,i,len1,cdata)
         end do

!c  channel two data
         linedelta=delta*ilin1(idnloc)
         irec=ilin1(idnloc)-NPTS/2-1+ioffdn+linedelta !offset in down
!!         print *, 'SearchLineStart:', irec
         green = 0.0
         do j=1,NPTS*2
            i=band2
            irow = irec + j
            call readBand2(band2Accessor,green(:,j),irow,i,len2,cdata)
         end do

!!         print *, 'RefRange:', isamp_s+1, isamp_s+(nloc-1)*dsamp+NPTS, len1
!!         print *, 'SrchRange: ', isamp_s+ioffac-NPTS/2, isamp_s+ioffac+3*NPTS/2+(nloc-1)*dsamp, len2



         do n=1,nloc
!c  copy data from first image
            do j=1,NPTS         !read input data (stationary part)
               do i=1,NPTS
                  a(i,j)=red(i+(n-1)*dsamp+isamp_s,j+NPTS/2)
               end do
            end do
!c     estimate and remove the phase carriers on the data
            call dephase(a,NPTS)
!c     interpolate the data by 2
            call interpolate(a,aa,NPTS)
!c  detect and store interpolated result in pa, after subtracting the mean
            amean=0.
            a2mean=0.
            do i=1,NPTS*2
               do j=1,NPTS*2
                  pa(i,j)=cabs(aa(i,j))
                  amean=amean+pa(i,j)
                  a2mean=a2mean+pa(i,j)*pa(i,j)
               end do
            end do
            

            amean=amean/NPTS**2/4.
            a2mean=a2mean/NPTS**2/4.

            normfactor1 = sqrt(a2mean-amean*amean)

            if ((amean.lt.1e-20).or.(normfactor1.lt.1e-20)) then
                normfactor1=1.
            endif
!           normfactor1 = 1.0

!            print *, '1: ', amean, log10(amean), log10(a2mean), log10(normfactor1)
            do i=1,NPTS*2
               do j=1,NPTS*2
                  pa(i,j)=(pa(i,j)-amean)/normfactor1
               end do
            end do
!c            print *,(pa(k,NPTS),k=NPTS-3,NPTS+3)
!c     read in channel 2 data (twice as much)
            do j=1,NPTS*2
               do i=1,NPTS*2
                  b(i,j)=green(i+ioffac-NPTS/2+(n-1)*dsamp+isamp_s,j)
               end do
            end do
!c     estimate and remove the phase carriers on the data
            call dephase(b,NPTS*2)
!c     interpolate the data by 2
            call interpolate(b,bb,NPTS*2)

!c  detect and store interpolated result in pb, after subtracting the mean
            amean=0.
            a2mean=0.
            do i=1,NPTS*4
               do j=1,NPTS*4
                  pb(i,j)=cabs(bb(i,j))
                  amean=amean+pb(i,j)
                  a2mean=a2mean+pb(i,j)*pb(i,j)
               end do
            end do
            amean=amean/NPTS**2/16.
            a2mean=a2mean/NPTS**2/16.

            normfactor2 = sqrt(a2mean-amean*amean)

            if ((amean.lt.1e-20).or.(normfactor2.lt.1e-20)) then
                normfactor2=1.
            endif
!             normfactor2 = 1.0
!!            print *, '2: ', amean, log10(amean), log10(a2mean), log10(normfactor2)
            do i=1,NPTS*4
               do j=1,NPTS*4
                  cpb(i,j)=(pb(i,j)-amean)/normfactor2
               end do
            end do

!c  get freq. domain cross-correlation
!c  first put pa array in double-size to match pb
            do i=1,NPTS*4
               do j=1,NPTS*4
                  cpa(j,i)=cmplx(0.,0.)
               end do
            end do
            do i=1,NPTS*2
               do j=1,NPTS*2
                  cpa(i+NPTS,j+NPTS)=pa(i,j)
               end do
            end do
!c  fft correlation
            call fft2d(cpa,NPTS*4,-1)
            call fft2d(cpb,NPTS*4,-1)
            do i=1,NPTS*4
               do j=1,NPTS*4
                  cpa(i,j)=conjg(cpa(i,j))*cpb(i,j)
               end do
            end do
            call fft2d(cpa,NPTS*4,1)
!c  get peak
            cmax=0.
            do ioff=-NOFF,NOFF
               do joff=-NOFF,NOFF
                  koff=ioff
                  loff=joff
                  if(koff.le.0)koff=koff+NPTS*4
                  if(loff.le.0)loff=loff+NPTS*4
                  c(ioff,joff)=cabs(cpa(koff,loff))**2
                  if(c(ioff,joff).ge.cmax)then
                     cmax=max(cmax,c(ioff,joff))
                     ipeak=ioff
                     jpeak=joff
                  end if
!c                  print *,cmax
               end do
            end do
!c  get integer peak representation, calculate 'snr'
            cave=0.
            do ioff=-NOFF,NOFF
               do joff=-NOFF,NOFF
                  ic(ioff,joff)=100.*c(ioff,joff)/cmax
                  cave=cave+abs(c(ioff,joff))
               end do
            end do
            snr=cmax/(cave/(2*NOFF+1)**2)
!c            print *, cmax, cave, snr
            if(cave.lt.1.e-20)snr=0.0
            if(isnan(snr))snr=0.0
!c  print out absolute correlations at original sampling rate
            if(talk.eq.'y') then
                print *,'Absolute offsets, original sampling interval:'
                do kk=-NDISP*2,NDISP*2,2
                    print '(1x,17i4)',(ic(k,kk),k=-NDISP*2,NDISP*2,2)
                end do
                print *, 'Expansion of peak, sample interval 0.5 * original:'
                do kk=jpeak-NDISP,jpeak+NDISP
                   print '(1x,17i4)',(ic(k,kk),k=ipeak-NDISP,ipeak+NDISP)
                end do
            endif
!c  get interpolated peak location from fft and oversample by NOVS
!c  load corr with correlation surface
            if(ipeak.gt.NOFF-NDISP/2)ipeak=NOFF-NDISP/2
            if(ipeak.lt.-NOFF+NDISP/2)ipeak=-NOFF+NDISP/2
            if(jpeak.gt.NOFF-NDISP/2)jpeak=NOFF-NDISP/2
            if(jpeak.lt.-NOFF+NDISP/2)jpeak=-NOFF+NDISP/2
            do ii=1,NDISP
               do jj=1,NDISP
                  corr(ii,jj)=cmplx(c(ipeak+ii-NDISP/2,jpeak+jj-NDISP/2),0.)
               end do
            end do
            call interpolaten(corr,corros,NDISP,NOVS)
            peak=0.
            do ii=1,(NDISP*NOVS)
               do jj=1,(NDISP*NOVS)
                  if(cabs(corros(ii,jj)).ge.peak)then
                     peak=cabs(corros(ii,jj))
                     iip=ii
                     jjp=jj
                  end if
               end do
            end do
            offac = (iip - (NDISP*NOVS)/2 -1)/(1.0*NOVS)
            offdn = (jjp - (NDISP*NOVS)/2 -1)/(1.0*NOVS)
!c            offac=iip/32.-65/32.
!c            offdn=jjp/32.-65/32.

            if(talk.eq.'y') then
                print *,'Interpolated across peak at ', offac+ioffac+ipeak/2.
                print *,'Interpolated down peak at   ', offdn+ioffdn+linedelta+jpeak/2.
                print *,'SNR: ',snr
            endif
            icnt = icnt + 1
            locationAcross(icnt) = (n-1)*dsamp+isamp_s
!c            locationAcrossOffset(icnt) = offac+ioffac+ipeak/2.
            locationAcrossOffset(icnt) = ioffac + (offac+ipeak)/2. 
            locationDown(icnt) = ilin1(idnloc)
!c            locationDownOffset(icnt) = offdn+ioffdn+linedelta+jpeak/2.
            locationDownOffset(icnt) = ioffdn + linedelta + (offdn+jpeak)/2.
            snrRet(icnt) = snr
            !print *, locationAcross(icnt),locationDown(icnt)
            !print *, locationAcrossOffset(icnt),locationDownOffset(icnt)
            !print *, snrRet(icnt)
         end do
      end do
      readBand1 => null()
      readBand2 => null()
      deallocate(red)
      deallocate(green)

! Piyush dellocate
      deallocate(a)
      deallocate(aa)
      deallocate(b)
      deallocate(bb)
      deallocate(cpa)
      deallocate(cpb)
      deallocate(corr)
      deallocate(corros)
      deallocate(pa)
      deallocate(pb)
      deallocate(c)
      deallocate(ic)
      deallocate(cdata)
    
      end

      subroutine dephase(a,n)
      complex a(n,n),csuma,csumd

!c  estimate and remove phase carriers in a complex array
      csuma=cmplx(0.,0.)
      csumd=cmplx(0.,0.)
!c  across first
      do i=1,n-1
         do j=1,n
            csuma=csuma+a(i,j)*conjg(a(i+1,j))
         end do
      end do
!c  down next
      do i=1,n
         do j=1,n-1
            csumd=csumd+a(i,j)*conjg(a(i,j+1))
         end do
      end do

      pha=atan2(aimag(csuma),real(csuma))
      phd=atan2(aimag(csumd),real(csumd))
!c      print *,'average phase across, down: ',pha,phd

!c  remove the phases
      do i=1,n
         do j=1,n
            a(i,j)=a(i,j)*cmplx(cos(pha*i+phd*j),sin(pha*i+phd*j))
         end do
      end do

      return
      end

      subroutine interpolate(a,b,n)
      complex a(n,n),b(n*2,n*2)
!c  zero out b array
      do i=1,n*2
         do j=1,n*2
            b(i,j)=cmplx(0.,0.)
         end do
      end do
!c  interpolate by 2, assuming no carrier on data
      call fft2d(a,n,-1)
!c  shift spectra around
      do i=1,n/2
         do j=1,n/2
            b(i,j)=a(i,j)
            b(i+3*n/2,j)=a(i+n/2,j)
            b(i,j+3*n/2)=a(i,j+n/2)
            b(i+3*n/2,j+3*n/2)=a(i+n/2,j+n/2)
         end do
      end do
!c  inverse transform
      call fft2d(b,n*2,1)
      return
      end

      subroutine fft2d(data,n,isign)
      complex data(n,n), d(8192)

      do i = 1 , n
         call cfft1d_jpl(n,data(1,i),isign)
      end do
      do i = 1 , n
         do j = 1 , n
            d(j) = data(i,j)
         end do
         call cfft1d_jpl(n,d,isign)
        do j = 1 , n
               data(i,j) = d(j)/n/n
         end do
      end do

      return
      end

      subroutine interpolaten(a,b,n,novr)
      complex a(n,n),b(n*novr,n*novr)

!c  zero out b array
      do i=1,n*novr
         do j=1,n*novr
            b(i,j)=cmplx(0.,0.)
         end do
      end do
!c  interpolate by novr, assuming no carrier on data
      call fft2d(a,n,-1)
!c  shift spectra around
      do i=1,n/2
         do j=1,n/2
            b(i,j)=a(i,j)
            b(i+(2*novr-1)*n/2,j)=a(i+n/2,j)
            b(i,j+(2*novr-1)*n/2)=a(i,j+n/2)
            b(i+(2*novr-1)*n/2,j+(2*novr-1)*n/2)=a(i+n/2,j+n/2)
         end do
      end do
!c  inverse transform
      call fft2d(b,n*novr,1)
      return
      end


