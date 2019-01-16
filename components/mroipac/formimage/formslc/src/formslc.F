      subroutine formslc(ptImageRaw,ptImageSLC)
        
      use fortranUtils
      use formslcState

      implicit none

      complex*8, DIMENSION(:,:), ALLOCATABLE :: trans 
      complex*8, DIMENSION(:), ALLOCATABLE :: ref, ref1, ref2, refn2
      real*4, DIMENSION(:,:), ALLOCATABLE :: spec
      real*4       unpacki1(0:255),unpackq1(0:255)
      real*8       fd1, fdd1, fddd1, fdddd1
      real*8       fd3, fdd3, fddd3, fdddd3
      real*8       r01, r1, r2, r, delr
      real*8       slpr, intr, slpa, inta, coefa(2)
      real*8       sloper, interr, slopea, intera
      real*8       dsloper, dinterr, dslopea, dintera
      real*8       dp1, fr1, slope1, slope2
      integer*4    iq, nr_fftf, nr_fftr, npow
      integer*4    i, j, k
      integer*4    nl, nls, ifrst, irec, npts
      integer*4    ier, ipatch, ifs
      integer*4    icaltone1, npfin1
      integer*4    k1start, k1end
      integer*4     iradelta, iusedopp
      real*4       win1, phase, t, t0, ts
      real*4       wgt, a21, a41
      real*4       t1, gcal, rhww1
      real*4       fwidth, pi, pi2
      real*4       pctbw, pctbwaz, dxsamp1
      integer*4    ioutrec1, ioutrec2
      integer*4    ioutrc1, ioutrc2
      integer*4    nint, NextPower
      integer*4    nagc, ndwp, offsetSet
      real*4 t2, t3, t4
      
      !c++ middleman objects pointers attached to the SLC and RAW images 
      INTEGER*8 :: ptImageSLC, ptImageRaw
      
      ! memory buffer to contain a slc line
      COMPLEX*8, allocatable, DIMENSION(:) ::dataLineSet
      REAL*4, DIMENSION(:), ALLOCATABLE :: agc, dwp
      ! memory buffer to contain a raw line
      INTEGER*1, allocatable, DIMENSION(:) ::  dataLineGet

      real*4 seconds
      external seconds

      double precision :: sol

!c initialize all arrays
    
      sol = getSpeedOfLight()
      pi =  getPI()
      pi2 = 2.0 * pi

!c get image line sizes.
      allocate(dataLineSet(nlinesaz))
      allocate(dataLineGet(nbytes1))
!c     
!**********  input parameters  **********
!c     
!c     enter input parameters
!c
      write(6,*) 'ROI - Repeat Orbit Interferometric processor'
      write(6,*) ''
!c     
      nagc = 0

      ndwp = 0
!c     compute the azimuth parameters
      
      iradelta = 0
      iusedopp = 1

c Giangi initialize some of the quantities that have been pass as arrays but in the fortran 
c where single variables
      fd1 = dopplerCoefficients(1)
      fdd1 = dopplerCoefficients(2)
      fddd1 = dopplerCoefficients(3)
      fdddd1 = dopplerCoefficients(4)

      pctbw = spectralShiftFrac(1)   
      pctbwaz = spectralShiftFrac(2)   
      
      sloper = linearResampCoeff(1)
      interr = linearResampCoeff(2)
      slopea = linearResampCoeff(3)
      intera = linearResampCoeff(4)

      dsloper = linearResampDeltas(1)
      dinterr = linearResampDeltas(2)
      dslopea = linearResampDeltas(3)
      dintera = linearResampDeltas(4)
      
      dxsamp1 = vel1/prf1
      write(6,*) 'actual orbital pulse spacing ', dxsamp1

      delr=sol/fs/2.
      write(6,*) ''
      write(6,*) 'First line to read in file 1 (start at 0) ', ifirstline 
      write(6,*) '# of range input patches ', npatches
      write(6,*) 'First sample pair to use ',ifirst
      write(6,*) 'Azimuth patch size ',nnn
      write(6,*) 'Number of valid points in azimuth ',na_valid
      write(6,*) 'Deskew the image? ',ideskew
      write(6,*) 'First range bin to save in file 1 ', isave
      write(6,*) 'Number of range bins to process ', nlinesaz
      write(6,*) 'Delta range pixels for second file: ',iradelta
      write(6,*) 'Caltone location 1 ',caltone1
      write(6,*) 'Doppler centroid quad coef 1 (Hz/prf) ', fd1, fdd1,
     $     fddd1, fdddd1
      write(6,*) 'Using Doppler flag ', iusedopp
      write(6,*) 'Effective S/C Body fixed velocity 1 (m/s) ', vel1
      write(6,*) 'Effective Azimuth sample spacing 1 (m) ', dxsamp1
      write(6,*) 'Earth Radius (m) ', re
      write(6,*) 'Spacecraft height 1 (m) ', ht1
      write(6,*) 'Range of first pixel in range compressed file 1 (m) ',
     $     r001
      write(6,*) 'PRF 1 (pps) ', prf1
      write(6,*) 'i/q means, i1,q1 ', xmi1, xmq1
      write(6,*) 'Flip i/q ', iflip
      write(6,*) 'Desired azimuth resolution (m) ', azres
      write(6,*) 'Number azimuth looks (m) ', nlooks
      write(6,*) 'Range sampling rate (Hz) ', fs
      write(6,*) 'Chirp Slope (Hz/s) ', slope
      write(6,*) 'Pulse Duration (s) ', pulsedur
      write(6,*) 'Chirp extension ', nextend
      write(6,*) 'Secondary Range Correction ', isrm
      write(6,*) 'Radar Wavelength (m) ', wavl
      write(6,*) 'Range Spectral Weighting ', rhww
      write(6,*) 'Fractional bandwidth to remove ', pctbw, pctbwaz
      write(6,*) 'linear resampling coefs:  sloper, intr, slopea, inta',
     .     sloper,interr,slopea, intera
      write(6,*) 'linear resampling deltas: dsloper, dintr, dslopea, dinta'
     $     ,dsloper,dinterr,dslopea, dintera


      write(6,*) ''

      t0 = seconds(0.0)           ! start timer



      iq = 1

!!      Currently, not considering offset video 
!!      if(iqflip .eq. 'O' .or. iqflip .eq. 'o') iq = 0

      fwidth=slope*pulsedur
      if(iq .eq. 0) then 
         npts=2.*fs*pulsedur
         ts=1./(2.*fs)
      else
         npts=fs*pulsedur
         ts=1./(fs)
      end if
      if(mod(npts,2) .eq. 0) npts=npts+1
      write(6,*) 'Pulse length in points: ',npts,ts

      if(iq .eq. 1) then
         npow = NextPower((ngood1+npts)/2)
         nr_fftf = 2 ** npow
         nr_fftr = nr_fftf
      else
         npow = NextPower((ngood1+npts))
         nr_fftf = 2 ** npow
         nr_fftr = nr_fftf/2
      end if

      if (nint(2**(log(float(nnn))/log(2.))) .ne. nnn) then
         write(*,*) 'log ',log(float(nnn))/log(2.)
         stop 'Azimuth patch size not a power of 2'
      end if

      ALLOCATE( trans(nnn,nlinesaz) )
      ALLOCATE( ref(max(nr_fftf+1,nnn)) )
      ALLOCATE( ref1(max(nr_fftf+1,nnn)) )
      ALLOCATE( ref2(max(nr_fftf+1,nnn)) )
      ALLOCATE( refn2(max(nr_fftf+1,nnn)) )
      ALLOCATE( spec(max(nr_fftf+1,nnn)/32,3) )

      trans = 0
      ref = 0
      ref1 = 0
      ref2 = 0
      refn2 = 0
      spec = 0
c      initialize transforms

      
      call cfft1d_jpl(nnn,ref,0)
      call cfft1d_jpl(nr_fftf,ref,0)
      call cfft1d_jpl(nr_fftr,ref,0)

      write(6,*) 'fft lengths ', nnn, nr_fftf, nr_fftr

c     compute range reference function

      fd1 = fd1 * prf1
      fdd1 = fdd1 * prf1
      fddd1 = fddd1 * prf1
      fdddd1 = fdddd1 * prf1

c     save to later compute coefficients referenced to updated starting range

      fd3    = fd1
      fdd3   = fdd1
      fddd3  = fddd1
      fdddd3 = fdddd1

      write(6,*) 'Doppler coefficients referenced to range bin index: '
      write(6,*) 'fd (Hz)                ', fd1
      write(6,*) 'd fd/dr (Hz/s)         ', fdd1
      write(6,*) 'd^2 fd/dr^2 (Hz/s/s)   ', fddd1
      write(6,*) 'd^3 fd/dr^3 (Hz/pix^3) ', fdddd1
      


c reference doppler coefficients to range

      call radopp(fd1,fdd1,fddd1,fdddd1,r001,delr)

      write(6,*) 'Doppler coefficients referenced to absolute range: '
      write(6,*) 'fd (Hz)                ', fd1
      write(6,*) 'd fd/dr (Hz/m)         ', fdd1
      write(6,*) 'd^2 fd/dr^2 (Hz/m^2)   ', fddd1
      write(6,*) 'd^3 fd/dr^3 (Hz/m^3)   ', fdddd1
      
      if(iusedopp .eq. 0) then
         write(6,*) 'Doppler check: ',fd1+fdd1*r001+fddd1*r001*r001+fdddd1*r001*r001*r001 
      else
         r = r001 + float(iradelta)*delr
         write(6,*) 'Doppler check: ',fd1+fdd1*r001+fddd1*r001*r001+fdddd1*r001*r001*r001 
      endif

c     update r001 and r002 to reflect the offsets given in isave,
c     iradelta, and nextend
      
      r01 = r001 + (isave-nextend-1) * delr

      slc_r0 = r01
      write(6,*) 'updated start ranges (m) ',r01

      call outdopp(fd3,fdd3,fddd3,fdddd3,r001,r01,delr)

      write(6,*) 'Doppler coefficients referenced to output range bin index: '
      write(6,*) 'fd (Hz)              ', fd3
      write(6,*) 'd fd/dr (Hz/pix)       ', fdd3
      write(6,*) 'd^2 fd/dr^2 (Hz/pix^2) ', fddd3
      write(6,*) 'd^3 fd/dr^3 (Hz/pix^3) ', fdddd3
      write(6,*) 'OUTDOP ', fd3, fdd3, fddd3, fdddd3

      r1  = r01 + delr*(nlinesaz-1)
      write(6,*) 'far range 1,2 (m) ', r1,r2
      npfin1 = r1*wavl/(2.0*azres*dxsamp1)+2
      write(6,*) 'filter points in far range ', npfin1

      a21 = -2.0*pi/(dxsamp1*float(nnn)) ! for deskew
      a41 = wavl/(2.0*azres*dxsamp1) ! for np
      write(6,*) 'coefficient for filter points a4 ',a41
      write(6,*) 'near range chirp rate ', -2. * (vel1*sqrt(re/(re+ht1)))**2
     $     /(wavl*r01) 
      write(6,*) 'far  range chirp rate ', -2. * (vel1*sqrt(re/(re+ht1)))**2
     $     /(wavl*r1) 
      
c secondary range migration correction to chirp rate (from CY memo)
c
      slope1 = slope
      slope2 = slope
      if(isrm .ne. 0) then

         r1 = r001 + delr* nlinesaz/2.
         fr1 = -2. * vel1**2/(wavl*r1) 
         dp1 = fd1 + fdd1 * r1 + fddd1 * r1**2 + fdddd1 * r1**3
         write(6,*) r1, fr1, dp1
         slope1 = slope / (1.d0 + slope * wavl**2 * dp1**2 / fr1 /
     $        sol**2)
         
      end if
      write(6,*) 'range chirp rate and SRC corrections ', slope, slope1

      rhww1 = 1.0-rhww

      k=0
      if(pctbw.ge.0.0)then
         k1start=abs(pctbw)*npts
         k1end=npts
      else
         k1start=0
         k1end=npts-abs(pctbw)*npts
      end if
      
      write(6,*) 'npts, k1 start, end'
      write(6,*) npts,k1start,k1end
      
      do i=-npts/2,npts/2
         k=k+1
         t=i*ts
         if(iq.eq.0) then
            phase = pi*slope1*t*t+pi*fs*t
            if(k.ge.k1start.and.k.le.k1end)
     +           ref1(i+npts/2+1)=cmplx(cos(phase),0.)
            phase = pi*slope2*t*t+pi*fs*t
            ref(i+npts/2+1)=cmplx(cos(phase),0.)
         else
            phase = pi*slope1*t*t
            if(k.ge.k1start.and.k.le.k1end)
     +           ref1(i+npts/2+1)=cmplx(cos(phase),sin(phase))
            phase = pi*slope2*t*t
            ref(i+npts/2+1)=cmplx(cos(phase),sin(phase))
         end if
      end do
      
      do i=1,npts
         win1 = (rhww-rhww1*cos((2.*pi*float(i-1))/float(npts-1)))
         ref(i)=ref(i)*win1
         ref1(i)=ref1(i)*win1
      end do

      
      if(nextend .gt. 0) then
         do i = 1 , npts
            k = i - nextend
             if(k .le. 0) k = k + nr_fftf
            ref(k) = ref(i)
            ref1(k) = ref1(i)
         end do
         do i = 1 , nextend     
            k = npts - nextend+i
            if(k .le. 0) k = k + nr_fftf
            ref(k) = cmplx(0.,0.)
            ref1(k) = cmplx(0.,0.)
         end do
      end if

      write(6,*) 'reference calculated in time domain.'
      
c     calculate fft of range reference function

      call cfft1d_jpl(nr_fftf,ref,-1)
      call cfft1d_jpl(nr_fftf,ref1,-1)

c zero out dc and caltone location
      icaltone1=nint(caltone1*nr_fftf)
      write(6,*) 'Caltone bins: ',icaltone1

      do i = 1, 6
         wgt = 0.5 - 0.5 * cos((i-1)/5.*pi)
         ref(i) = ref(i) * wgt
         ref1(i) = ref1(i) * wgt
         ref(nr_fftf+1-i) = ref(nr_fftf+1-i) * wgt
         ref1(nr_fftf+1-i) = ref1(nr_fftf+1-i) * wgt

         if(iq .eq. 0) then
            ref(i+nr_fftf/2) = ref(i+nr_fftf/2) * wgt
            ref1(i+nr_fftf/2) = ref1(i+nr_fftf/2) * wgt
            ref(nr_fftf/2+1-i) = ref(nr_fftf/2+1-i) * wgt
            ref1(nr_fftf/2+1-i) = ref1(nr_fftf/2+1-i) * wgt
            if(icaltone1 .gt. 6 .and. icaltone1 .lt. nr_fftf-6) then
               ref(i+icaltone1) = ref(i+icaltone1) * wgt
               ref1(i+icaltone1) = ref1(i+icaltone1) * wgt
               ref(icaltone1+1-i) = ref(icaltone1+1-i) * wgt
               ref1(icaltone1+1-i) = ref1(icaltone1+1-i) * wgt
               ref(i+nr_fftf-icaltone1) = ref(i+nr_fftf-icaltone1) * wgt
               ref1(i+nr_fftf-icaltone1) = ref1(i+nr_fftf-icaltone1)
     $              * wgt
               ref(nr_fftf-icaltone1+1-i) = ref(nr_fftf-icaltone1+1-i)
     $              * wgt
               ref1(nr_fftf-icaltone1+1-i) = ref1(nr_fftf-icaltone1+1-i)
     $              * wgt
            end if
         end if
      end do

c     scale reference for channel gain, conjugate

      gcal=1./nr_fftf
      do i=1,nr_fftf
         refn2(i)=conjg(refn2(i))*gcal
         ref(i)=conjg(ref(i))*gcal
         ref1(i)=conjg(ref1(i))*gcal
      end do

c     save spectra of reference functions for checking

         
c     
c     

c offset into valid data in a patch

      ifs = (nnn-na_valid)/2

c     load the unpacking array

      if(iflip .eq. 0) then
         do i=0,255
            unpacki1(i)=float(i)-xmi1
            unpackq1(i)=float(i)-xmq1
         end do
      else
         do i=0,255
            unpacki1(i)=float(i)-xmq1
            unpackq1(i)=float(i)-xmi1
         end do
      end if
      

      t2=seconds(t0)
      write(6,*) 'XXX elapsed time before looping over patches', t2
c     
c     begin loop to range process data
c     
      ioutrec1 = 0
      ioutrec2 = 0
      ioutrc1 = 0
      ioutrc2 = 0

      
      slc_line0 = 0
c     get offsets for the given patch
      do ipatch=1,npatches
         slpr = sloper+(ipatch-1)*dsloper
         intr = interr+(ipatch-1)*dinterr
         slpa = slopea+(ipatch-1)*dslopea
         inta = intera+(ipatch-1)*dintera
         write(6,*) 'Patch ',ipatch,' resampling inputs: ',slpr,intr,slpa
     $        ,inta

         coefa(1) = slpa
         coefa(2) = inta

c convert to function of range instead of pixel

         intr = intr - slpr*r001/delr
         slpr = slpr/delr

         inta = inta - slpa*r001/delr
         slpa = slpa/delr

         write(6,*) 'Converted Resampling inputs ', slpr,intr,slpa,inta
c     
c     compress channels
         
c do range compression for file 1
        
         
         write(6,*) ''
         write(6,*) ' range compressing channel 1 .....'
         write(6,*) ''

         irec=ifirstline+(ipatch-1)*na_valid
         ifrst= ifirst+isave-1

         if (ipatch.eq.1) then
             slc_line0 = irec + ifs +1 
         endif


         t3=seconds(t0)
         call rcpatch(ptImageRaw, dataLineGet,nnn,nlinesaz,trans,unpacki1,unpackq1,
     $        ref1,irec,ifrst,nbytes1,ngood1,nr_fftf, nr_fftr,iq,iflip, 
     $           agc,dwp,nagc,ndwp)
         
         t4=seconds(t0)
         write(6,*) 'XXX rcpatch took: ',t4-t3,'s'

            
            t1=seconds(t0)
            write(6,*) 'first channel range processing elapsed time: ',t1
     $           ,'sec'

c     transform lines for file 1
         
            call cffts(trans,nnn,1,nlinesaz,nnn,1,ier)
            t1=seconds(t0)
            write(6,*) 'transformed lines ', t1, ' sec'

c     start the range migration correction for file 1
         
         write(6,*) 'start range migration correction.'
         
         nl=nlinesaz
         r=r01
            call RMpatch(trans,0.d0,0.d0,
     .           nnn,nl,nls,r,delr,wavl,vel1,ht1,re,fd1,fdd1,fddd1,fdddd1,prf1,ideskew
     $           )
            t1=seconds(t0)
            write(6,*) 'range migrated ',t1, ' sec'

c multiply by reference and inverse transform lines for file 1
         

         nl=nlinesaz
         r = r01
         call ACpatch(trans,nnn,nl,r,delr,
     .           wavl,vel1,fd1,fdd1,fddd1,fdddd1,prf1,ht1,re,
     .           gm,r_platvel1,r_platacc1,i_lrl,
     .           npfin1,a21,a41,0.d0,0.d0,ideskew,na_valid)

         t1=seconds(t0)
         write(6,*) 'Range-Doppler done',t1,' sec'

c write out the slc for file 1

         t3=seconds(t0)
         offsetSet = 0


         do i = nnn/2-na_valid/2+1, nnn/2+na_valid/2
            do j = 1 , nlinesaz
c Giangi
                dataLineSet(j) = trans(i,j)
c               data(j) = trans(i,j)
cc               write(6,*) aimag(data(j)), real(data(j))
            end do
               
            ioutrec1 = ioutrec1 + 1
          call setLineSequential(ptImageSLC,dataLineSet)
               
         end do
         t4=seconds(t0)
         write(6,*) 'XXX Output took : ',t4-t3,'s'

         t1 = seconds(0.0)
         write(6,*) 'written up to record ',ioutrec1, ':', t1-t0, ' sec '



c     if first time through, save complex images for file 2

         

            

      end do      !end patch loop
     
      !if not destroyed there is a memory leack.  
      call cfft1d_jpl(nnn,ref,2)
      call cfft1d_jpl(nr_fftf,ref,2)
      call cfft1d_jpl(nr_fftr,ref,2)
      DEALLOCATE( trans )
      DEALLOCATE( ref )
      DEALLOCATE( ref1 )
      DEALLOCATE( ref2 )
      DEALLOCATE( refn2 )
      DEALLOCATE( spec )
      DEALLOCATE( dataLineSet )
      DEALLOCATE( dataLineGet )
      

      end
        
      subroutine moddopp(fd1, fdd1, fddd1, fdddd1, fd2, fdd2, fddd2, fdddd2,
     .      iusedopp, iradelta)

      integer*4 iusedopp, iradelta
      real*8 fd1, fdd1, fddd1, fdddd1, fd2, fdd2, fddd2, fdddd2
      real*8 temp1, temp2, temp3, temp4

      if(iusedopp .eq. 0) then
         write(6,*) 'using doppler as is '
      elseif(iusedopp .eq. 1) then
         write(6,*) 'referencing ch 2 dopp to ch 1'
         fd2 = fd1
     .        - float(iradelta)    * fdd1 
     .        + float(iradelta)**2 * fddd1
     .        - float(iradelta)**3 * fdddd1
         fdd2 = fdd1
     .         - 2.d0 * float(iradelta)    * fddd1
     .         + 3.d0 * float(iradelta)**2 * fddd1
         fddd2 = fddd1
     .          - 3.d0 * float(iradelta)    * fdddd1
         fdddd2 = fdddd1
      elseif(iusedopp .eq. 2) then
         write(6,*) 'referencing ch 1 dopp to ch 2'
         fd1 = fd2
     .        + float(iradelta)    * fdd2 
     .        + float(iradelta)**2 * fddd2
     .        + float(iradelta)**3 * fdddd2
         fdd1 = fdd2
     .         + 2.d0 * float(iradelta)    * fddd2
     .         + 3.d0 * float(iradelta)**2 * fddd2
         fddd1 = fddd2
     .          + 3.d0 * float(iradelta)    * fdddd2
         fdddd1 = fdddd2
      elseif(iusedopp .eq. 3) then
         write(6,*) 'averaging dopplers'
         temp1 = fd2
     .        + float(iradelta)    * fdd2 
     .        + float(iradelta)**2 * fddd2
     .        + float(iradelta)**3 * fdddd2
         temp2 = fdd2
     .         + 2.d0 * float(iradelta)    * fddd2
     .         + 3.d0 * float(iradelta)**2 * fddd2
         temp3 = fddd2
     .          + 3.d0 * float(iradelta)    * fdddd2
         temp4 = fdddd2
         fd1    = (fd1    + temp1)/2.d0
         fdd1   = (fdd1   + temp2)/2.d0
         fddd1  = (fddd1  + temp3)/2.d0
         fdddd1 = (fdddd1 + temp4)/2.d0
         fd2 = fd1
     .        - float(iradelta)    * fdd1 
     .        + float(iradelta)**2 * fddd1
     .        - float(iradelta)**3 * fdddd1
         fdd2 = fdd1
     .         - 2.d0 * float(iradelta)    * fddd1
     .         + 3.d0 * float(iradelta)**2 * fddd1
         fddd2 = fddd1
     .          - 3.d0 * float(iradelta)    * fdddd1
         fdddd2 = fdddd1
      end if
      return
      end

      subroutine radopp(fd, fdd, fddd, fdddd, r, del)

      real*8    fd, fdd, fddd, fdddd, r, del, temp1, temp2, temp3, temp4

      temp1 = fd
     .       - fdd   * (r/del)
     .       + fddd  * (r/del)**2
     .       - fdddd * (r/del)**3
      temp2 = fdd/del
     .       - 2.d0 * fddd  * (r/del)    /del
     .       + 3.d0 * fdddd * (r/del)**2 / del
      temp3 = fddd / del**2
     .       - 3.d0 * fdddd * (r/del) / del**2
      temp4 = fdddd / del**3

      fd    = temp1
      fdd   = temp2
      fddd  = temp3
      fdddd = temp4

      return
      end

      subroutine outdopp(fd, fdd, fddd, fdddd, r_old, r_new, del)

      implicit none
      real*8    fd, fdd, fddd, fdddd, r_old, r_new, del, temp1, temp2, temp3, temp4

      temp1 = fd
     .       + fdd   * ((r_new-r_old)/del)
     .       + fddd  * ((r_new-r_old)/del)**2
     .       + fdddd * ((r_new-r_old)/del)**3
      temp2 = fdd
     .       + 2.d0 * fddd  * ((r_new-r_old)/del)
     .       + 3.d0 * fdddd * ((r_new-r_old)/del)**2
      temp3 = fddd
     .       + 3.d0 * fdddd  * ((r_new-r_old)/del)
      temp4 = fdddd

      fd    = temp1
      fdd   = temp2
      fddd  = temp3
      fdddd = temp4

      return
      end

      real*4 function seconds(t0)
      real*4 t0
      real*8 secondo

      seconds = secondo(-1) - t0

      return
      end

      integer function lastnb(string)
      
      character*(*) string
      integer ln
      
      ln = len(string) 
      
      lastnb = 0
      do i = ln , 1 , -1
         if(string(i:i) .ne. ' ') then
            lastnb = i 
            return
         end if
      end do
      return 
      end
      Integer*4 Function NextPower(number)
      
c Function to get the next highest power of two abov
c argument.
      
      implicit none

      integer power, k, number
      
      k = 0
      power = 0
 1    if(power .lt. number) then
         k = k + 1
         power = 2 ** k
         goto 1
      end if
      NextPower = k
      return
      end
