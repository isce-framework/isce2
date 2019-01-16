!c***************************************************************

      subroutine resamp(slcAccessor1,slcAccessor2,intAccessor,ampAccessor,resampAccessor2)

!c***************************************************************
!c*     
!c*   FILE NAME: resampdb.f90  - derived from resamp_roi.F
!c*     
!c*   DATE WRITTEN: Long, long ago. (March 16, 1992)
!c*     
!c*   PROGRAMMER: Charles Werner, Paul Rosen and Scott Hensley
!c*     
!c*   FUNCTIONAL DESCRIPTION: Interferes two SLC images 
!c*   range, azimuth interpolation with a quadratic or sinc interpolator 
!c*   no circular buffer is used, rather a batch algorithm is implemented
!c*   The calculation of the range and azimuth offsets is done for
!c*   each of the data sets in the offset data file. As soon as the
!c*   current line number exceeds the range line number for one of the
!c*   data sets in the offset data file, the new lsq coefficients are
!c*   to calculate the offsets for any particular range pixel. 
!c*     
!c*   ROUTINES CALLED:
!c*     
!c*   NOTES: 
!c*     
!c*   UPDATE LOG:
!c*
!c*   Date Changed        Reason Changed 
!c*   ------------       ----------------
!c*     20-apr-92    added removal/reinsertion of range phase slope to 
!c*                  improve correlation
!c*     11-may-92    added code so that the last input block of data is processed
!c*                  even if partially full
!c*     9-jun-92     modified maximum number of range pixels
!c*     17-nov-92    added calculation of the range phase shift/pixel
!c*     29-mar-93    write out multi-look images (intensity) of the two files 
!c*     93-99        Stable with small enhancements changes
!c*     Dec 99       Modified range interpolation to interpret (correctly)
!c*                  the array indices to be those of image 2 coordinates.  
!c*                  Previous code assumed image 1, and therefore used 
!c*                  slightly wrong offsets for range resampling depending
!c*                  on the gross offset between images.  Mods involve computing
!c*                  the inverse mapping
!c*     Aug 16, 04   This version uses MPI (Message Passing Interface)
!c*                  to parallelize the resamp_roi sequential computations.
!c*                  File Name is changed to resamp_roi.F in order to use
!c*                  the Fortran compiler pre-processor to do conditional
!c*                  compiling (#ifdef etc).  This code can be compiled for
!c*                  either sequential or parallel uses. Compiler flag 
!c*                  -DMPI_PARA is needed in order to pick up the MPI code.
!c*
!c*     May 2, 09    Changed to use db as per sqlite3 processor (hz)
!c*
!c*     Nov 11 2013  KK: Added resampAccessor2 to get the resampled slc.
!c*                  If an accessor is 0, it will not be output to disk.
!c*
!c*     01-DEC-2017  Cunren Liang: avoid the blank in last block
!c*                  added variable: ndown2
!c*
!c****************************************************************

      use resampState
      use omp_lib
      use uniform_interp
      use fortranUtils
      implicit none

!c    PARAMETER STATEMENTS:

      integer*8 slcAccessor1,slcAccessor2,intAccessor,ampAccessor,resampAccessor2 !KK
      integer    NPP,MP
      parameter (NPP=10)

      real*8  pi
      integer  NP, N_OVER, NBMAX
!!      parameter (NP=30000)      !maximum number of range pixels
!!      parameter (NLINESMAX=200000) ! maximum number of SLC lines
!!      parameter (NAZMAX=16)             !number of azimuth looks
      parameter (N_OVER=2000)  !overlap between blocks
!!      parameter (NBMAX=200*NAZMAX+2*N_OVER) !number of lines in az interpol

      integer FL_LGT
      parameter (FL_LGT=8192*8)

      integer MAXDECFACTOR      ! maximum lags in interpolation kernels
      parameter(MAXDECFACTOR=8192)                        
      
      integer MAXINTKERLGH      ! maximum interpolation kernel length
      parameter (MAXINTKERLGH=8)
      
      integer MAXINTLGH         ! maximum interpolation kernel array size
      parameter (MAXINTLGH=MAXINTKERLGH*MAXDECFACTOR)

!c    LOCAL VARIABLES:

      character*20000 MESSAGE
      
      integer istats, l1, l2, lc, line
      integer nplo, i_numpnts
      integer ibs, ibe, irec, i_a1, i_r1, jrec, jrecp
      integer i, j, k, ii, ix, nb
      integer int_az_off
      integer, allocatable :: int_rd(:)
      integer, allocatable ::  int_az(:)
      integer i_na, ibfcnt       
      integer linePos,offsetCnt

      real*8  f_delay
      real*4  , allocatable :: fintp(:)
      real , allocatable :: am(:,:),amm(:)
      real , allocatable :: bm(:,:),bmm(:)
      complex , allocatable :: abmm(:)
      
      real*8 , allocatable :: fr_rd(:),fr_az(:)

      real*8 cpp, rphs, aa1, rphs1, r_ro, r_ao, rsq, asq, rmean
      real*8 amean, azsum, azoff1, rd, azs
      real*8 azmin

      complex, allocatable ::  cm(:)
      complex , allocatable ::dm(:)
      complex , allocatable ::em(:)
      real*8 , allocatable ::fd(:)
      
      complex, allocatable ::  tmp(:)
      complex, allocatable :: a(:),b(:,:)
      complex , allocatable ::cc(:),c(:,:),dddbuff(:)
      complex , allocatable ::rph(:,:)               !range phase correction

      real*8 ph1, phc, r_q
      real*8 f0,f1,f2,f3           !doppler centroid function of range poly file 1
      real*8, allocatable ::r_ranpos(:),r_azpos(:),r_sig(:),r_ranoff(:)
      real*8 , allocatable ::r_azoff(:),r_rancoef(:),r_azcoef(:)
      real*8 r_chisq
      real*8 , allocatable ::r_v(:,:),r_u(:,:),r_w(:)
      real*8 , allocatable ::r_ranpos2(:),r_azpos2(:),r_sig2(:),r_ranoff2(:)
      real*8 , allocatable ::r_azoff2(:),r_rancoef2(:),r_azcoef2(:)
      real*8 , allocatable ::r_rancoef12(:)

      real*8 r_beta,r_relfiltlen,r_pedestal
      real*8 , allocatable ::r_filter(:)
      real*8 r_delay
      integer i_decfactor,i_weight,i_intplength,i_filtercoef

      real*4 t0, t1

      real*8 r_azcorner,r_racorner

!c    COMMON BLOCKS:

      integer i_fitparam(NPP),i_coef(NPP)
      external poly_funcs    !!Needed first to avoid seg faults
      common /fred/ i_fitparam,i_coef 

      complex , allocatable :: slcLine1(:)               
      complex , allocatable :: slcLine2(:)               

      NP = max(npl,npl2) +2 !!Earlier a constant - PSA 
      MP = max(dim1_r_ranpos, dim1_r_ranpos2) +2 !!Earlier a constant - PSA
      NBMAX=200*NAZ+2*N_OVER  !!Earlier a constant - PSA
!c    ARRAY ALLOCATIONS:
      allocate(tmp(0:NP-1))
      allocate(int_rd(0:NP-1))
      allocate(int_az(0:NP-1))
      allocate(fintp(0:FL_LGT-1))
      allocate(am(0:NP-1,0:NAZ-1))
      allocate(amm(0:NP-1))
      allocate(bm(0:NP-1,0:NAZ-1))
      allocate(bmm(0:NP-1))
      allocate(fr_rd(0:NP-1))
      allocate(fr_az(0:NP-1))
      allocate(cm(0:NP-1))
      allocate(dm(0:NP-1))
      allocate(em(0:NP-1))
      allocate(fd(0:NP-1))
      allocate(a(0:NP-1))
      allocate(b(0:NP-1,0:NBMAX-1))
      allocate(c(0:NP-1,0:NAZ-1))
      allocate(dddbuff(0:NP-1))
      allocate(rph(0:NP-1,0:NAZ-1))
      allocate(r_ranpos(MP))
      allocate(r_azpos(MP))
      allocate(r_sig(MP))
      allocate(r_ranoff(MP))
      allocate(r_azoff(MP))
      allocate(r_rancoef(NPP))
      allocate(r_azcoef(NPP))
      allocate(r_v(NPP,NPP))
      allocate(r_u(MP,NPP))
      allocate(r_w(NPP))
      allocate(r_ranpos2(MP))
      allocate(r_azpos2(MP))
      allocate(r_sig2(MP))
      allocate(r_ranoff2(MP))
      allocate(r_azoff2(MP))
      allocate(r_rancoef2(NPP))
      allocate(r_azcoef2(NPP))
      allocate(r_rancoef12(NPP))
      allocate(r_filter(0:MAXINTLGH))

      pi = getPi()

!c    PROCESSING STEPS:

      istats=0
      t0 = secnds(0.0)
      nplo = npl
      allocate(slcLine1(npl))
      allocate(slcLine2(npl2))
      allocate(cc(0:nplo/NR-1))
      allocate(abmm(0:nplo/NR-1))
      write(MESSAGE,*) ' '       
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*)  ' << RTI Interpolation and Cross-correlation (quadratic) v1.0 >>'
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) ' ' 
      call write_out(ptStdWriter,MESSAGE)

      if((npl .gt. NP) .or. (npl2 .gt. NP)) then
         write(MESSAGE,*) 'ERROR:number of pixels greater than array in resampd'
         call write_out(ptStdWriter,MESSAGE)
         stop
      end if
     
      f0 = 0.0d0
      f1 = 0.0d0
      f2 = 0.0d0
      f3 = 0.0d0

      !jng set the doppler coefficients
      i_na = size(dopplerCoefficients)
      f0 = dopplerCoefficients(1)
      if (i_na.gt.1)  f1 = dopplerCoefficients(2)
      if (i_na.gt.2)  f2 = dopplerCoefficients(3)
      if (i_na.gt.3)  f3 = dopplerCoefficients(4)

!c    open offset file

      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,'(a,x,i5,x,i5)') 'Interferogram formed from lines: ',ist,ist+nl
      call write_out(ptStdWriter,MESSAGE)

      if(istats .eq. 1)then
         write(MESSAGE,*) ' '
         call write_out(ptStdWriter,MESSAGE)
         write(MESSAGE,*) ' Range    R offset     Azimuth    Az offset     SNR '
         call write_out(ptStdWriter,MESSAGE)
         write(MESSAGE,*) '++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
         call write_out(ptStdWriter,MESSAGE)
         write(MESSAGE,*) ' '
         call write_out(ptStdWriter,MESSAGE)
      endif

!c    reading offsets data file (note NS*NPM is maximal number of pixels)
      
      i_numpnts = dim1_r_ranpos
      i_na = 0
      !jng at this point the position and offset array are already set, so find th az max
      ! also convert the snr to the format used here. there my be division by zero that i guess fortran can handle (gives  +Infinity)
      do j=1,i_numpnts           !read the offset data file
            r_ranpos(j) = r_ranposV(j)
            r_azpos(j) = r_azposV(j)
            r_ranoff(j) = r_ranoffV(j)
            r_azoff(j) = r_azoffV(j)
            r_ranpos2(j) = r_ranpos2V(j)
            r_azpos2(j) = r_azpos2V(j)
            r_ranoff2(j) = r_ranoff2V(j)
            r_azoff2(j) = r_azoff2V(j)
            i_na = max(i_na,int(r_azpos(j)))
            r_sig(j) = 1.0 + 1.d0/r_sigV(j)
            r_sig2(j) = 1.0 + 1.d0/r_sig2V(j)
      end do
      write(MESSAGE,*) 'Number of points read    =  ',i_numpnts
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Number of points allowed =  ',MP
      call write_out(ptStdWriter,MESSAGE)

!c    find average int az off

      azsum = 0.
      azmin = r_azpos(1)
      do j=1,i_numpnts
         azsum = azsum + r_azoff(j)
         azmin = min(azmin,r_azpos(j))
      enddo
      azoff1 = azsum/i_numpnts
      int_az_off = nint(azoff1)
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Average azimuth offset = ',azoff1,int_az_off
      call write_out(ptStdWriter,MESSAGE)
      
      do i = 1 , i_numpnts
         r_azpos(i) = r_azpos(i) - azmin
         r_azpos2(i) = r_azpos2(i) - int_az_off - azmin
      end do

!c    make two two dimensional quadratic fits for the offset fields 
!c    one of the azimuth offsets and the other for the range offsets

      do i = 1 , NPP
         r_rancoef(i) = 0.
         r_rancoef2(i) = 0.
         r_rancoef12(i) = 0.
         r_azcoef(i) = 0.
         r_azcoef2(i) = 0.
         i_coef(i) = 0
      end do

      do i=1,i_ma
         i_coef(i) = i
      enddo

!c    azimuth offsets as a function range and azimuth

      call svdfit(r_ranpos,r_azpos,r_azoff,r_sig,i_numpnts, &
          r_azcoef,i_ma,r_u,r_v,r_w,MP,NPP,r_chisq)

      write(MESSAGE,*) 'Azimuth sigma = ',sqrt(r_chisq/i_numpnts)
      call write_out(ptStdWriter,MESSAGE)

!c    inverse mapping azimuth offsets as a function range and azimuth

      call svdfit(r_ranpos2,r_azpos2,r_azoff2,r_sig2,i_numpnts, &
          r_azcoef2,i_ma,r_u,r_v,r_w,MP,NPP,r_chisq)

      write(MESSAGE,*) 'Inverse Azimuth sigma = ',sqrt(r_chisq/i_numpnts)
      call write_out(ptStdWriter,MESSAGE)

!c    range offsets as a function of range and azimuth

      call svdfit(r_ranpos,r_azpos,r_ranoff,r_sig,i_numpnts, &
          r_rancoef,i_ma,r_u,r_v,r_w,MP,NPP,r_chisq)

      write(MESSAGE,*) 'Range sigma = ',sqrt(r_chisq/i_numpnts)
      call write_out(ptStdWriter,MESSAGE)

!c    Inverse range offsets as a function of range and azimuth

      call svdfit(r_ranpos2,r_azpos2,r_ranoff2,r_sig2,i_numpnts, &
          r_rancoef2,i_ma,r_u,r_v,r_w,MP,NPP,r_chisq)

      write(MESSAGE,*) 'Inverse Range sigma = ',sqrt(r_chisq/i_numpnts)
      call write_out(ptStdWriter,MESSAGE)

!c    Inverse range offsets as a function of range and azimuth

      call svdfit(r_ranpos,r_azpos2,r_ranoff2,r_sig2,i_numpnts, &
          r_rancoef12,i_ma,r_u,r_v,r_w,MP,NPP,r_chisq)

      write(MESSAGE,*) 'Inverse Range sigma = ',sqrt(r_chisq/i_numpnts)
      call write_out(ptStdWriter,MESSAGE)

      write(MESSAGE,*) ' ' 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range offset fit parameters'
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Constant term =            ',r_rancoef(1) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range Slope term =         ',r_rancoef(2) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth Slope =            ',r_rancoef(3) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range/Azimuth cross term = ',r_rancoef(4) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range quadratic term =     ',r_rancoef(5) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth quadratic term =   ',r_rancoef(6) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range/Azimuth^2   term =   ',r_rancoef(7) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth/Range^2 =          ',r_rancoef(8) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range cubic term =         ',r_rancoef(9) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth cubic term =       ',r_rancoef(10) 
      call write_out(ptStdWriter,MESSAGE)
       
      write(MESSAGE,*) ' ' 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth offset fit parameters'
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Constant term =            ',r_azcoef(1) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range Slope term =         ',r_azcoef(2) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth Slope =            ',r_azcoef(3) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range/Azimuth cross term = ',r_azcoef(4) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range quadratic term =     ',r_azcoef(5) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth quadratic term =   ',r_azcoef(6) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range/Azimuth^2   term =   ',r_azcoef(7) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth/Range^2 =          ',r_azcoef(8) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range cubic term =         ',r_azcoef(9) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth cubic term =       ',r_azcoef(10) 
      call write_out(ptStdWriter,MESSAGE)

      write(MESSAGE,*)
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Comparison of fit to actuals'
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) '   Ran       AZ    Ranoff    Ran fit  Rand Diff  Azoff    Az fit   Az Diff'
      call write_out(ptStdWriter,MESSAGE)
      rmean= 0.
      amean= 0.
      rsq= 0.
      asq= 0.
      offsetCnt = 0
      do i=1,i_numpnts
         r_ro = r_rancoef(1) + r_azpos(i)*(r_rancoef(3) + &
             r_azpos(i)*(r_rancoef(6) + r_azpos(i)*r_rancoef(10))) + &
             r_ranpos(i)*(r_rancoef(2) + r_ranpos(i)*(r_rancoef(5) + &
             r_ranpos(i)*r_rancoef(9))) + &
             r_ranpos(i)*r_azpos(i)*(r_rancoef(4) + r_azpos(i)*r_rancoef(7) + &
             r_ranpos(i)*r_rancoef(8)) 
         r_ao = r_azcoef(1) + r_azpos(i)*(r_azcoef(3) + &
             r_azpos(i)*(r_azcoef(6) + r_azpos(i)*r_azcoef(10))) + &
             r_ranpos(i)*(r_azcoef(2) + r_ranpos(i)*(r_azcoef(5) + &
             r_ranpos(i)*r_azcoef(9))) + &
             r_ranpos(i)*r_azpos(i)*(r_azcoef(4) + r_azpos(i)*r_azcoef(7) + &
             r_ranpos(i)*r_azcoef(8)) 
         !jng instead of saving all the information in the file, we only save r_a,ro and resturn them
         offsetCnt = offsetCnt + 1
         downOffset(offsetCnt) = r_ao
         acrossOffset(offsetCnt) = r_ro
         rmean = rmean + (r_ranoff(i)-r_ro)
         amean = amean + (r_azoff(i)-r_ao)
         rsq = rsq + (r_ranoff(i)-r_ro)**2
         asq = asq + (r_azoff(i)-r_ao)**2
         if(istats .eq. 1) write(6,150)  r_ranpos(i),r_azpos(i),r_ranoff(i), &
              r_ro,r_ranoff(i)-r_ro,r_azoff(i),r_ao,r_azoff(i)-r_ao
 150     format(2(1x,f8.1),1x,f8.3,1x,f12.4,1x,f12.4,2x,f8.3,1x,f12.4,1xf12.4,1x1x)

!         write(13,269) int(r_ranpos(i)),r_ranoff(i)-r_ro,int(r_azpos(i)),r_azoff(i)-r_ao,10.,1.,1.,0.

! 269     format(i6,1x,f10.3,1x,i6,f10.3,1x,f10.5,3(1x,f10.6))

      enddo 
      rmean = rmean / i_numpnts
      amean = amean / i_numpnts
      rsq = sqrt(rsq/i_numpnts - rmean**2)
      asq = sqrt(asq/i_numpnts - amean**2)
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,'(a,x,f15.6,x,f15.6)') 'mean, sigma range   offset residual (pixels): ',rmean, rsq
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,'(a,x,f15.6,x,f15.6)') 'mean, sigma azimuth offset residual (pixels): ',amean, asq
      call write_out(ptStdWriter,MESSAGE)
      
      write(MESSAGE,*) ' ' 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range offset fit parameters'
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Constant term =            ',r_rancoef2(1) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range Slope term =         ',r_rancoef2(2) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth Slope =            ',r_rancoef2(3) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range/Azimuth cross term = ',r_rancoef2(4) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range quadratic term =     ',r_rancoef2(5) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth quadratic term =   ',r_rancoef2(6) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range/Azimuth^2   term =   ',r_rancoef2(7) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth/Range^2 =          ',r_rancoef2(8) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range cubic term =         ',r_rancoef2(9) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth cubic term =       ',r_rancoef2(10) 
      call write_out(ptStdWriter,MESSAGE)
       
      write(MESSAGE,*) ' ' 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth offset fit parameters'
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Constant term =            ',r_azcoef2(1) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range Slope term =         ',r_azcoef2(2) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth Slope =            ',r_azcoef2(3) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range/Azimuth cross term = ',r_azcoef2(4) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range quadratic term =     ',r_azcoef2(5) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth quadratic term =   ',r_azcoef2(6) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range/Azimuth^2   term =   ',r_azcoef2(7) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth/Range^2 =          ',r_azcoef2(8) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Range cubic term =         ',r_azcoef2(9) 
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Azimuth cubic term =       ',r_azcoef2(10) 
      call write_out(ptStdWriter,MESSAGE)

      write(MESSAGE,*)
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) 'Comparison of fit to actuals'
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) '   Ran       AZ    Ranoff    Ran fit  Rand Diff  Azoff    Az fit   Az Diff'
      call write_out(ptStdWriter,MESSAGE)
      rmean= 0.
      amean= 0.
      rsq= 0.
      asq= 0.
      do i=1,i_numpnts
         r_ro = r_rancoef2(1) + r_azpos2(i)*(r_rancoef2(3) + &
             r_azpos2(i)*(r_rancoef2(6) + r_azpos2(i)*r_rancoef2(10))) + &
             r_ranpos2(i)*(r_rancoef2(2) + r_ranpos2(i)*(r_rancoef2(5) + &
             r_ranpos2(i)*r_rancoef2(9))) + &
             r_ranpos2(i)*r_azpos2(i)*(r_rancoef2(4) + r_azpos2(i)*r_rancoef2(7) + &
             r_ranpos2(i)*r_rancoef2(8)) 
         r_ao = r_azcoef2(1) + r_azpos2(i)*(r_azcoef2(3) + &
             r_azpos2(i)*(r_azcoef2(6) + r_azpos2(i)*r_azcoef2(10))) + &
             r_ranpos2(i)*(r_azcoef2(2) + r_ranpos2(i)*(r_azcoef2(5) + &
             r_ranpos2(i)*r_azcoef2(9))) + &
             r_ranpos2(i)*r_azpos2(i)*(r_azcoef2(4) + r_azpos2(i)*r_azcoef2(7) + &
             r_ranpos2(i)*r_azcoef2(8)) 
         rmean = rmean + (r_ranoff2(i)-r_ro)
         amean = amean + (r_azoff2(i)-r_ao)
         rsq = rsq + (r_ranoff2(i)-r_ro)**2
         asq = asq + (r_azoff2(i)-r_ao)**2
         if(istats .eq. 1) write(6,150)  r_ranpos2(i),r_azpos2(i), &
            r_ranoff(i),r_ro,r_ranoff2(i)-r_ro,r_azoff2(i),r_ao,r_azoff2(i)-r_ao
         !jng instead of saving all the information in the file, we only save r_a,ro and resturn them
         offsetCnt = offsetCnt + 1
         downOffset(offsetCnt) = r_ao
         acrossOffset(offsetCnt) = r_ro
!         write(13,269) int(r_ranpos2(i)),r_ranoff2(i)-r_ro, &
!            int(r_azpos2(i)),r_azoff2(i)-r_ao,10.,1.,1.,0.


       enddo 
       rmean = rmean / i_numpnts
       amean = amean / i_numpnts
       rsq = sqrt(rsq/i_numpnts - rmean**2)
       asq = sqrt(asq/i_numpnts - amean**2)
       write(MESSAGE,*) ' '
       call write_out(ptStdWriter,MESSAGE)
       write(MESSAGE,'(a,x,f15.6,x,f15.6)') 'mean, sigma range   offset residual (pixels): ',rmean, rsq
       call write_out(ptStdWriter,MESSAGE)
       write(MESSAGE,'(a,x,f15.6,x,f15.6)') 'mean, sigma azimuth offset residual (pixels): ',amean, asq
       call write_out(ptStdWriter,MESSAGE)

!c    limits of resampling offsets
      do i=1,4
         if(i.eq.1)then
            r_azcorner=ist
            r_racorner=0
         end if
         if(i.eq.2)then
            r_azcorner=ist
            r_racorner=npl-1
         end if
         if(i.eq.3)then
            r_azcorner=ist+nl
            r_racorner=0
         end if
         if(i.eq.4)then
            r_azcorner=ist+nl
            r_racorner=npl-1
         end if
         r_ro = r_rancoef2(1) + r_azcorner*(r_rancoef2(3) + &
             r_azcorner*(r_rancoef2(6) + r_azcorner*r_rancoef2(10))) + &
             r_racorner*(r_rancoef2(2) + r_racorner*(r_rancoef2(5) + &
             r_racorner*r_rancoef2(9))) + &
             r_racorner*r_azcorner*(r_rancoef2(4) + r_azcorner*r_rancoef2(7) + &
             r_racorner*r_rancoef2(8)) 
         r_ao = r_azcoef2(1) + r_azcorner*(r_azcoef2(3) + &
             r_azcorner*(r_azcoef2(6) + r_azcorner*r_azcoef2(10))) + &
             r_racorner*(r_azcoef2(2) + r_racorner*(r_azcoef2(5) + &
             r_racorner*r_azcoef2(9))) + &
             r_racorner*r_azcorner*(r_azcoef2(4) + r_azcorner*r_azcoef2(7) + &
             r_racorner*r_azcoef2(8)) 
         if(i.eq.1) then
            write(MESSAGE,*),'Upper left offsets: ',r_ro,r_ao
            call write_out(ptStdWriter,MESSAGE)
         end if
         if(i.eq.2) then
            write(MESSAGE,*),'Upper right offsets:',r_ro,r_ao
            call write_out(ptStdWriter,MESSAGE)
         end if
         if(i.eq.3) then
            write(MESSAGE,*),'Lower left offsets: ',r_ro,r_ao
            call write_out(ptStdWriter,MESSAGE)
         end if
         if(i.eq.4) then
            write(MESSAGE,*),'Lower right offsets:',r_ro,r_ao
            call write_out(ptStdWriter,MESSAGE)
         end if
       enddo 
       
!c    read in data files

      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,'(a,x,i5)') 'Number samples in interferogram: ',nplo/NR
      call write_out(ptStdWriter,MESSAGE)

      CPP=SLR/WVL

      i_a1 = i_na - int(azmin)
      i_r1 = int(npl/2.)
      rphs  = 360. * 2. * CPP * (r_rancoef(2) + i_a1*(r_rancoef(4) + &
          r_rancoef(7)*i_a1) + i_r1*(2.*r_rancoef(5) + &
          3.*r_rancoef(9)*i_r1 + 2.*r_rancoef(8)*i_a1))

      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,'(a,x,3(f15.6,x))') 'Pixel shift/pixel in range    = ',rphs/(CPP*360.),aa1,sngl(r_rancoef(2))
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,'(a,x,3(f15.6,x))') 'Degrees per pixel range shift = ',rphs,rphs1,2.*sngl(r_rancoef(2)*CPP*360.)
      call write_out(ptStdWriter,MESSAGE)

      if(f0 .eq. -99999.)then
         write(MESSAGE,*) ' '
         call write_out(ptStdWriter,MESSAGE)
         write(MESSAGE,*) 'Estimating Doppler from imagery...' 
         call write_out(ptStdWriter,MESSAGE)
         l1 = 1
         l2 = nb
         do j=l1-1,l2-1
            linePos = j
            if(mod(j,100) .eq. 0)then
               write(MESSAGE,*) 'Reading file at line = ',j
               call write_out(ptStdWriter,MESSAGE)
            endif
            call getLine(slcAccessor1,slcLine1,linePos)
            b(0:npl-1,j) = slcLine1(:)
         enddo 
         call doppler(npl,l1,l2,b,fd,dddbuff)
         do j=0,npl-1
            write(MESSAGE,*) j,fd(j)
            call write_out(ptStdWriter,MESSAGE)
         enddo
      endif

!c    compute resample coefficients 
      
      r_beta = 1.d0
      r_relfiltlen = 8.d0
      i_decfactor = 8192
      r_pedestal = 0.d0
      i_weight = 1
      
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,'(a)') 'Computing sinc coefficients...'
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      
      call sinc_coef(r_beta,r_relfiltlen,i_decfactor,r_pedestal,i_weight,i_intplength,i_filtercoef,r_filter)
      
      r_delay = i_intplength/2.d0
      f_delay = r_delay
      
      do i = 0 , i_intplength - 1
         do j = 0 , i_decfactor - 1
           fintp(i+j*i_intplength) = r_filter(j+i*i_decfactor)
         enddo
      enddo

      nb = NBMAX
      ibfcnt = (NBMAX-2*N_OVER)/NAZ
      ibfcnt = ibfcnt * NAZ
      nb = ibfcnt + 2*N_OVER

      if(nb .ne. NBMAX) then
         write(MESSAGE,*) 'Modified buffer max to provide sync-ed overlap'
         call write_out(ptStdWriter,MESSAGE)
         write(MESSAGE,*) 'Max buffer size = ',NBMAX
         call write_out(ptStdWriter,MESSAGE)
         write(MESSAGE,*) 'Set buffer size = ',nb
         call write_out(ptStdWriter,MESSAGE)
      end if

!c    begin interferogram formation

      write(MESSAGE,'(a)') 'Beginning interferogram formation...'
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) ' '
      call write_out(ptStdWriter,MESSAGE)
      
      ibfcnt = nb-2*N_OVER

!c   XXX Start of line loop
      do line=0,nl/NAZ-1
         lc = line*NAZ
         ibfcnt = ibfcnt + NAZ
         
         if(ibfcnt .ge. nb-2*N_OVER) then

            ibfcnt = 0
            ibs = ist+int_az_off-N_OVER+lc/(nb-2*N_OVER)*(nb-2*N_OVER)
            ibe = ibs+nb-1

            write(MESSAGE,'(a,x,i5,x,i5,x,i5,x,i5,x,i5)') &
                'int line, slc line, buffer #, line start, line end: ', &
                line,lc,lc/(nb-2*N_OVER)+1,ibs,ibe
            call write_out(ptStdWriter,MESSAGE)
            write(MESSAGE,'(a,i5,a)') 'Reading ',nb,' lines of data'
            call write_out(ptStdWriter,MESSAGE)

            do i=0, nb-1        !load up  buffer
               irec = i + ibs
               jrec = irec + istoff - 1  ! irec,jrec = image 2 coordinates
               jrecp = jrec - int_az_off - int(azmin) ! subtract big constant for fit

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                if(irec .gt. 0)then       !in the data?

!                   if(irec .gt. nl+ist+int_az_off)then
!                      go to 900
!                   endif
!                   linePos = irec
! !                  write(MESSAGE,*)'2b',linePos
! !                  call write_out(ptStdWriter,MESSAGE)
! !                  if(linePos.le.0 .or. linePos.gt.nl) then
! !                  endif
!                   call getLine(slcAccessor2,slcLine2,linePos)
!                   tmp(0:npl2-1) = slcLine2(:)
! !                  write(MESSAGE,*)'2a',linePos
! !                  call write_out(ptStdWriter,MESSAGE)
!                   !read(UNIT=22,REC=irec,iostat=ierr) (tmp(ii),ii=0,npl2-1) 
!                   if(linePos .lt. 0) goto 900
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

               ! I comment out the above to avoid the blank in last block
               ! Now the 900 statement is actually never used.
               ! Cunren Liang, 02-DEC-2017
               if(irec .ge. 1 .and. irec .le. ndown2)then
                  linePos = irec
                  call getLine(slcAccessor2,slcLine2,linePos)
                  tmp(0:npl2-1) = slcLine2(:)

!c*    calculate range interpolation factors, which depend on range and azimuth
!c*    looping over IMAGE 2 COORDINATES.
!$omp parallel do private(j,r_ro,rd) shared(r_rancoef12,&
!$omp &nplo,jrecp,int_rd,fr_rd,f_delay)
                  do j=0,nplo-1 
                     r_ro = r_rancoef12(1) + jrecp*(r_rancoef12(3) + &
                         jrecp*(r_rancoef12(6) + jrecp*r_rancoef12(10))) + &
                         j*(r_rancoef12(2) + j*(r_rancoef12(5) + &
                         j*r_rancoef12(9))) + &
                         j*jrecp*(r_rancoef12(4) + jrecp*r_rancoef12(7) + &
                         j*r_rancoef12(8)) 
                     rd = r_ro + j 
                     int_rd(j)=int(rd+f_delay)
                     fr_rd(j)=rd+f_delay-int_rd(j)
                  end do
!$omp end parallel do

!$omp parallel do private(j) shared(nplo,b,tmp,npl2,fintp,int_rd,fr_rd)
                  do j=0,nplo-1  !range interpolate
                     b(j,i)= sinc_eval(tmp,npl2,fintp,8192,8,int_rd(j),fr_rd(j))
!                     if( int_rd(j).lt.7 .or. int_rd(j).ge.npl2) then
!                         print *, 'Rng:',j, int_rd(j), b(j,i) 
!                     endif
                  end do
!$omp end parallel do
               else

                  do j=0,nplo-1  !fill with 0, no data yet
                     b(j,i)=(0.,0.)
                  end do

               end if  !have data in image 2 corresponding to image 1
            end do     !i loop

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!             goto 901            !jump around this code to fill

!  900        write(MESSAGE,'(a,x,i5)') 'Filling last block, line: ',i
!             call write_out(ptStdWriter,MESSAGE)

!             do ii=i,nb-1
!                do j=0,nplo-1
!                   b(j,ii)=(0.,0.)
!                end do
!             end do

!  901        continue
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

         end if
         do k=0,NAZ-1
            irec = ist + line*NAZ + k
            jrec = irec + istoff - int(azmin) - 1

!c note: this is only half the phase! Some for each channel
!$omp parallel do private(j,r_ro,r_ao,azs) shared(nplo,irec,jrec,&
!$omp &int_az,fr_az,rph,CPP,pi,r_rancoef,r_azcoef,k)
            do j=0,nplo-1
               r_ro = r_rancoef(1) + jrec*(r_rancoef(3) + &
                   jrec*(r_rancoef(6) + jrec*r_rancoef(10))) + &
                   j*(r_rancoef(2) + j*(r_rancoef(5) + &
                   j*r_rancoef(9))) + &
                   j*jrec*(r_rancoef(4) + jrec*r_rancoef(7) + &
                   j*r_rancoef(8)) 
               r_ao = r_azcoef(1) + jrec*(r_azcoef(3) + &
                   jrec*(r_azcoef(6) + jrec*r_azcoef(10))) + &
                   j*(r_azcoef(2) + j*(r_azcoef(5) + &
                   j*r_azcoef(9))) + &
                   j*jrec*(r_azcoef(4) + jrec*r_azcoef(7) + &
                   j*r_azcoef(8))

!c*    !calculate azimuth offsets

               azs = irec + r_ao 
!c              int_az(j) = nint(azs)
               if(azs .ge. 0.d0) then
                  int_az(j) = int(azs)
               else
                  int_az(j) = int(azs) - 1
               end if
               fr_az(j) = azs - int_az(j)
               rph(j,k)=cmplx(cos(sngl(2.*pi*r_ro*CPP)),-sin(sngl(2.*pi*r_ro*CPP)))
            end do !loop-j
!$omp end parallel do

            linePos = irec
!            write(MESSAGE,*)'1b',linePos
!            call write_out(ptStdWriter,MESSAGE)
!            if(linePos.le.0 .or. linePos.gt.nl) then
!            endif
            call getLine(slcAccessor1,slcLine1,linePos)
!            write(MESSAGE,*)'1a',linePos
!            call write_out(ptStdWriter,MESSAGE)
            !if(ierr .ne. 0) goto 1000
            if(linePos .lt. 0) goto 1000
            tmp(0:npl-1) = slcLine1(:)
            do j=0,npl-1
               a(j) = tmp(j)*rph(j,k)
            end do

!$omp parallel do  private(j,ix,r_q,ph1,phc,tmp,ii) shared(nplo,&
!$omp &f0,f1,f2,f3,fr_az,int_az,b,cm,rph,ibs,pi,fintp)
           do j=0,nplo-1        !azimuth interpolation
               ix = int_az(j)-ibs
               r_q = (((f3 * j + f2) * j) + f1) * j + f0
               ph1 = (r_q)*2.0*PI
               phc = fr_az(j) * ph1
               do ii = -3, 4
!                  if((ix+ii).lt.0 .or. (ix+ii).ge.NBMAX) then
!                      print *, 'Az:', j, ix, ii
!                  endif
                  tmp(ii+3) = b(j,ix+ii) * cmplx(cos(ii*ph1),-sin(ii*ph1))
               end do
               cm(j) = sinc_eval(tmp,8,fintp,8192,8,7,fr_az(j))
               cm(j) = cm(j) * cmplx(cos(phc),+sin(phc)) !KK removed conjg(rph(j,k))
            end do  !loop-j
!$omp end parallel do

            !KK: check whether to output resamp, int, amp
            if (resampAccessor2 .ne. 0) then !output resamp
               call setLine(resampAccessor2,cm,linePos)
            end if
            if (intAccessor .eq. 0) then !skip int and amp
               goto 5671
            end if
            do j=0,nplo-1
               cm(j) = cm(j) * conjg(rph(j,k))
            end do
            !KK

            dm(nplo-1) = a(nplo-1)
            dm(0) = a(0)
            em(nplo-1) = cm(nplo-1)
            em(0) = cm(0)
!$omp parallel do private(j) shared(nplo,dm,em,a,cm)
            do j = 1, nplo-2
               dm(j) = .23*a(j-1)+a(j)*.54+a(j+1)*.23
               em(j) = .23*cm(j-1)+cm(j)*.54+cm(j+1)*.23
            end do !loop-j
!$omp end parallel do

!$omp parallel do private(j) shared(nplo,k,c,dm,em,am,bm)
            do j = 0, nplo -1
               c(j,k)  =       dm(j)*   conjg(em(j))    !1-look correlation
               am(j,k) = real(dm(j))**2+aimag(dm(j))**2 !intensity of a
               bm(j,k) = real(em(j))**2+aimag(em(j))**2 !intensity of b
            end do !loop-j
!$omp end parallel do
5671        continue !KK
         end do !loop-k

!c    take looks
         if (intAccessor .eq. 0) then !KK skip again looks
            goto 5672 !KK
         end if !KK

         if(iflatten .eq. 1) then
            
            do j=0, nplo/NR-1   !sum over NR*NAZ looks
               cc(j)=(0.,0.)    !intialize sums
               amm(j)=0.
               bmm(j)=0.
               do k=0,NAZ-1
                  do i=0,NR-1
                     cc(j)=cc(j)+c(j*NR+i,k)
                     amm(j)=amm(j)+am(j*NR+i,k)
                     bmm(j)=bmm(j)+bm(j*NR+i,k)
                  end do
               end do
            end do
         else
            do j=0, nplo/NR-1   !sum over NR*NAZ looks
               cc(j)=(0.,0.)    !intialize sums
               amm(j)=0.
               bmm(j)=0.
               do k=0,NAZ-1
                  do i=0,NR-1
                     cc(j)=cc(j)+c(j*NR+i,k)
                     amm(j)=amm(j)+am(j*NR+i,k)
                     bmm(j)=bmm(j)+bm(j*NR+i,k)
                  end do
               end do
               cc(j)=cc(j)*conjg(rph(NR*j,NAZ/2)*rph(NR*j,NAZ/2)) !reinsert range phase
               abmm(j)=cmplx(sqrt(amm(j)),sqrt(bmm(j)))
            end do
         end if
         linePos = line + 1
         call setLine(intAccessor,cc,linePos)
         call setLine(ampAccessor,abmm,linePos)
5672     continue !KK
      end do
!cc XXX End of line loop

 1000 t1 = secnds(t0)
      write(MESSAGE,*) 'XXX time: ', t1-t0
      call write_out(ptStdWriter,MESSAGE)

      deallocate(tmp) 
      deallocate(slcLine1)
      deallocate(slcLine2)
      deallocate(int_rd)
      deallocate(int_az)
      deallocate(fintp)
      deallocate(am)
      deallocate(amm)
      deallocate(bm)
      deallocate(bmm)
      deallocate(abmm)
      deallocate(fr_rd)
      deallocate(fr_az)
      deallocate(cm)
      deallocate(dm)
      deallocate(em)
      deallocate(fd)
      deallocate(a)
      deallocate(b)
      deallocate(cc)
      deallocate(c)
      deallocate(dddbuff)
      deallocate(rph)
      deallocate(r_ranpos)
      deallocate(r_azpos)
      deallocate(r_sig)
      deallocate(r_ranoff)
      deallocate(r_azoff)
      deallocate(r_rancoef)
      deallocate(r_azcoef)
      deallocate(r_v)
      deallocate(r_u)
      deallocate(r_w)
      deallocate(r_ranpos2)
      deallocate(r_azpos2)
      deallocate(r_sig2)
      deallocate(r_ranoff2)
      deallocate(r_azoff2)
      deallocate(r_rancoef2)
      deallocate(r_azcoef2)
      deallocate(r_rancoef12)
      deallocate(r_filter)
      end
      


