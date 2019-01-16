!c***************************************************************
      subroutine resamp_only(ifgInAccessor, ifgOutAccessor, ampInAccessor, ampOutAccessor)

!c***************************************************************
!c*     
!c*   FILE NAME: resamp_only.f90  - derived from resamp_roi.F
!c*     
!c*   DATE WRITTEN: Long, long ago. (March 16, 1992)
!c*     
!c*   PROGRAMMER: Charles Werner, Paul Rosen and Scott Hensley
!c*     
!c*   FUNCTIONAL DESCRIPTION: Resamples one complex image to coordinates
!c*   set by offsets in rgoffset.out
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
!c*     Dec 17, 12   Changed to resample amp file as was done for the ifg file
!c*
!c****************************************************************

      use resamp_onlyState

      implicit none

!c    PARAMETER STATEMENTS:

      integer    NPP,MP
      parameter (NPP=10)

      real*8, parameter :: PI = 4.d0*atan(1.d0)
      integer  NP
      parameter (NP=30000)      !maximum number of range pixels

      integer FL_LGT
      parameter (FL_LGT=8192*8)

      integer MAXDECFACTOR      ! maximum lags in interpolation kernels
      parameter(MAXDECFACTOR=8192)                        
      
      integer MAXINTKERLGH      ! maximum interpolation kernel length
      parameter (MAXINTKERLGH=8)
      
      integer MAXINTLGH         ! maximum interpolation kernel array size
      parameter (MAXINTLGH=MAXINTKERLGH*MAXDECFACTOR)


!c    INPUT VARIABLES:

      integer*8 ifgInAccessor, ampInAccessor

!c    INPUT VARIABLES:

      integer*8 ifgOutAccessor, ampOutAccessor

!c    LOCAL VARIABLES:

      character*20000 MESSAGE

      integer istats, iflatten
      integer lineNum
      integer ist, nr, naz, i_numpnts
      integer i, j, k
      integer int_az_off
      integer i_na

      real*8 r_ro, r_ao, rsq, asq, rmean
      real*8 amean,  azsum, azoff1
      real*8 r_rt,r_at, azmin

      real*8 f0,f1,f2,f3           !doppler centroid function of range poly file 1
      real*8 r_rancoef(NPP),r_azcoef(NPP)
      real*8 r_v(NPP,NPP),r_w(NPP),r_chisq
      real*8 r_rancoef2(NPP),r_azcoef2(NPP)
      real*8 r_rancoef12(NPP)

      real*8, allocatable:: r_ranpos(:), r_azpos(:)
      real*8, allocatable:: r_ranoff(:), r_azoff(:)
      real*8, allocatable:: r_sig(:),r_u(:,:),r_sig2(:)
      real*8, allocatable:: r_ranpos2(:), r_azpos2(:)
      real*8, allocatable:: r_ranoff2(:), r_azoff2(:)

      real*4 t0, t1

      real*8 r_azcorner,r_racorner,fracr,fraca

      complex acc(32768), ctop, cbot, atop, abot
      complex, allocatable :: c1(:,:), a1(:,:), c2(:), a2(:)
      integer kk,ifrac

!c    COMMON BLOCKS:

      integer i_fitparam(NPP),i_coef(NPP)
      common /fred/ i_fitparam,i_coef 

!c    FUNCTION STATEMENTS:

      external poly_funcs

!c    PROCESSING STEPS:

      t0 = secnds(0.0)

      write(MESSAGE,*) ' '       
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*)  ' << Resample one image to another image coordinates >>'
      call write_out(ptStdWriter,MESSAGE)
      write(MESSAGE,*) ' ' 
      call write_out(ptStdWriter,MESSAGE)

      istats=0
      MP = dim1_r_ranpos
!c  allocate the big arrays
      allocate (r_ranpos(MP),r_azpos(MP))
      allocate (r_ranoff(MP),r_azoff(MP))
      allocate (r_sig(MP), r_sig2(MP))
      allocate (r_ranpos2(MP),r_azpos2(MP))
      allocate (r_ranoff2(MP),r_azoff2(MP))
      allocate (r_u(MP,NPP))
      allocate (c1(npl,nl), a1(npl,nl))
      allocate (c2(npl), a2(npl))
      NR=1
      NAZ=1
      iflatten = 0
      ist=1
!c    open offset file

      f0 = 0.0d0
      f1 = 0.0d0
      f2 = 0.0d0
      f3 = 0.0d0

      i_na = size(dopplerCoefficients)
      !jng set the doppler coefficients
      f0 = dopplerCoefficients(1)
      if(i_na.gt.1) f1 = dopplerCoefficients(2)
      if(i_na.gt.2) f2 = dopplerCoefficients(3)
      if(i_na.gt.3) f3 = dopplerCoefficients(4)

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
         rmean = rmean + (r_ranoff(i)-r_ro)
         amean = amean + (r_azoff(i)-r_ao)
         rsq = rsq + (r_ranoff(i)-r_ro)**2
         asq = asq + (r_azoff(i)-r_ao)**2
         if(istats .eq. 1) write(6,150)  r_ranpos(i),r_azpos(i),r_ranoff(i), &
              r_ro,r_ranoff(i)-r_ro,r_azoff(i),r_ao,r_azoff(i)-r_ao
 150     format(2(1x,f8.1),1x,f8.3,1x,f12.4,1x,f12.4,2x,f8.3,1x,f12.4,1xf12.4,1x1x)

!         write(13,269) int(r_ranpos(i)),r_ranoff(i)-r_ro,int(r_azpos(i)),r_azoff(i)-r_ao,10.,1.,1.,0.

 269     format(i6,1x,f10.3,1x,i6,f10.3,1x,f10.5,3(1x,f10.6))

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

!c    read in data file

      lineNum = 1
      do j = 1,nl
         call getLineSequential(ifgInAccessor,c1(:,j),lineNum)
         call getLineSequential(ampInAccessor,a1(:,j),lineNum)
      enddo
!c  estimate azimuth carrier
      do j=1,npl
         acc(j)=cmplx(0.,0.)
      end do
      
      do i=1,min(1000, nl-1)
         do j=1,npl
            acc(j)=acc(j)+c1(j,i+1)*conjg(c1(j,i))
         end do
      end do
      do j=1,npl
          azimuthCarrier(j) = atan2(aimag(acc(j)),real(acc(j)))
      end do

!c  estimate range carrier
      do j=1,npl
         acc(j)=cmplx(0.,0.)
      end do
      do i=1,min(1000, nl)
         do j=1,npl-1
            acc(j)=acc(j)+c1(j+1,i)*conjg(c1(j,i))
         end do
      end do
      do j=1,npl-1
         rangeCarrier(j) = atan2(aimag(acc(j)),real(acc(j)))
      end do
      rangeCarrier(npl) = 0.0

!c  loop over lines
       do j=1,nl
          if(mod(j,1000).eq.0) then
             write(MESSAGE,*),'At line ',j
             call write_out(ptStdWriter,MESSAGE)
          end if
          do i=1,npl
             c2(i)=cmplx(0.,0.)
             a2(i)=cmplx(0.,0.)
          end do
          do i=1,npl
             r_rt=i
             r_at=j

             r_ro = r_rancoef2(1) + r_at*(r_rancoef2(3) + &
                  r_at*(r_rancoef2(6) + r_at*r_rancoef2(10))) + &
                  r_rt*(r_rancoef2(2) + r_rt*(r_rancoef2(5) + &
                  r_rt*r_rancoef2(9))) + &
                  r_rt*r_at*(r_rancoef2(4) + r_at*r_rancoef2(7) + &
                  r_rt*r_rancoef2(8)) 
             r_ao = r_azcoef2(1) + r_at*(r_azcoef2(3) + &
                  r_at*(r_azcoef2(6) + r_at*r_azcoef2(10))) + &
                  r_rt*(r_azcoef2(2) + r_rt*(r_azcoef2(5) + &
                  r_rt*r_azcoef2(9))) + &
                  r_rt*r_at*(r_azcoef2(4) + r_at*r_azcoef2(7) + &
                  r_rt*r_azcoef2(8)) 

!!$             k=nint(i+r_ro)
!!$             if(k.lt.1)k=1
!!$             if(k.gt.npl)k=npl   !nearest neighbor in range
!!$             kk=nint(j+r_ao)
!!$             if(kk.lt.1)kk=1
!!$             if(kk.gt.nl)kk=nl   !nearest neightbor in azimuth
!!$             c2(i,j)=c1(k,kk)
             k=int(i+r_ro)   !range offset
             fracr=i+r_ro-k
             ifrac=1 !8*nint(frac*8192)
             if(k.lt.4)then
                k=4
                ifrac=0
             end if
             if(k.gt.npl-4)then
                k=npl-4
                ifrac=0
             end if                !left of point in range

             kk=int(j+r_ao)  !azimuth offset
             fraca=j+r_ao-kk
!c             ifrac=8*nint(frac*8192)
             if(kk.lt.4)then
                kk=4
                ifrac=0
             end if
             if(kk.gt.nl-4)then
                kk=nl-4
                ifrac=0
             end if                   !left of point in azimuth

!             c2(i,j)=c1(nint(k+fracr),nint(kk+fraca))   !nearest neighbor
!             a2(i)=a1(nint(k+fracr),nint(kk+fraca))   !nearest neighbor
             ctop=c1(k,kk)*(1-fracr)+c1(k+1,kk)*fracr
             cbot=c1(k,kk+1)*(1-fracr)+c1(k+1,kk+1)*fracr
             atop=a1(k,kk)*(1-fracr)+a1(k+1,kk)*fracr
             abot=a1(k,kk+1)*(1-fracr)+a1(k+1,kk+1)*fracr
             if(ifrac.eq.1)then
                c2(i)=ctop*(1-fraca)+cbot*fraca          !bilinear
                a2(i)=atop*(1-fraca)+abot*fraca            !bilinear
             else
                c2(i)=cmplx(0.,0.)
                a2(i)=cmplx(0.,0.)
             end if
          end do

          call setLineSequential(ifgOutAccessor,c2)
          call setLineSequential(ampOutAccessor,a2)

       end do


!cc XXX End of line loop

      t1 = secnds(t0)
      write(MESSAGE,*) 'Elapsed time: ', t1
      call write_out(ptStdWriter,MESSAGE)

      deallocate (c1)
      deallocate (c2)
      deallocate (a1)
      deallocate (a2)
      deallocate (r_ranpos,r_azpos)
      deallocate (r_ranoff,r_azoff)
      deallocate (r_ranpos2,r_azpos2)
      deallocate (r_ranoff2,r_azoff2)
      deallocate (r_sig,r_sig2)
      deallocate (r_u)

      end
      

