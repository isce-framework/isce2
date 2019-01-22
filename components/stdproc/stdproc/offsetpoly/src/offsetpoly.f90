!c***************************************************************

      subroutine offsetpoly()

!c***************************************************************
!c*
!c*  Estimates the offset polynomial to be used for resampling
!c*
!c*
!c****************************************************************

      use offsetpolyState
      use fortranUtils
      implicit none

!c    PARAMETER STATEMENTS:

      integer    NPP, MP
      parameter (NPP=10)


!c    LOCAL VARIABLES:

      real*8, allocatable ::r_ranpos(:),r_azpos(:),r_sig(:),r_off(:)
      real*8, allocatable :: r_coef(:), r_w(:)
      real*8, allocatable :: r_u(:,:), r_v(:,:)
      real*8 r_chisq, r_ro, rmean, rsq
      integer i,j, i_numpnts

      real*4 t0, t1


!c    COMMON BLOCKS:

      integer i_fitparam(NPP),i_coef(NPP)
      external poly_funcs
      common /fred/ i_fitparam,i_coef 


      t0 = secnds(0.0)

!c    ARRAY ALLOCATIONS:
      MP = numOffsets
      
      allocate(r_ranpos(MP))
      allocate(r_azpos(MP))
      allocate(r_sig(MP))
      allocate(r_off(MP))
      allocate(r_coef(NPP))
      allocate(r_u(MP,NPP))
      allocate(r_v(NPP,NPP))
      allocate(r_w(NPP))


!c    reading offsets data file (note NS*NPM is maximal number of pixels)
      
      i_numpnts = numOffsets
      ! also convert the snr to the format used here. there my be division by zero that i guess fortran can handle (gives  +Infinity)
      do j=1,i_numpnts           !read the offset data file
            r_ranpos(j) = r_ranposV(j)
            r_azpos(j) = r_azposV(j)
            r_off(j) = r_offV(j)
            r_sig(j) = 1.0 + 1.d0/r_sigV(j)
      end do

!c    make two two dimensional quadratic fits for the offset fields 
!c    one of the azimuth offsets and the other for the range offsets

      do i = 1 , NPP
         r_coef(i) = 0.
         i_coef(i) = 0
      end do

      do i=1,i_ma
         i_coef(i) = i
      enddo

!c    azimuth offsets as a function range and azimuth
!     do i=1,i_numpnts
!        print *,r_ranpos(i),r_azpos(i),r_sig(i), r_off(i)
!     end do
!     print *, 'Fit: ', i_fitparam
!     print *, 'Coef: ', i_coef

      call svdfit(r_ranpos,r_azpos,r_off,r_sig,i_numpnts, &
          r_coef,i_ma,r_u,r_v,r_w,MP,NPP,r_chisq)

      print *, 'Fit sigma = ',sqrt(r_chisq/i_numpnts)

      rmean= 0.
      rsq= 0.
      do i=1,i_numpnts
         r_ro = r_coef(1) + r_azpos(i)*(r_coef(3) + &
             r_azpos(i)*(r_coef(6) + r_azpos(i)*r_coef(10))) + &
             r_ranpos(i)*(r_coef(2) + r_ranpos(i)*(r_coef(5) + &
             r_ranpos(i)*r_coef(9))) + &
             r_ranpos(i)*r_azpos(i)*(r_coef(4) + r_azpos(i)*r_coef(7) + &
             r_ranpos(i)*r_coef(8)) 
         rmean = rmean + (r_off(i)-r_ro)
         rsq = rsq + (r_off(i)-r_ro)**2
      enddo

      rmean = rmean / i_numpnts
      rsq = sqrt(rsq/i_numpnts - rmean**2)
      print *,'mean, sigma offset residual (pixels): ',rmean, rsq
      
      print *, 'Constant term =            ',r_coef(1) 
      print *, 'Range Slope term =         ',r_coef(2) 
      print *, 'Azimuth Slope =            ',r_coef(3) 
      print *, 'Range/Azimuth cross term = ',r_coef(4) 
      print *, 'Range quadratic term =     ',r_coef(5) 
      print *, 'Azimuth quadratic term =   ',r_coef(6) 
      print *, 'Range/Azimuth^2   term =   ',r_coef(7) 
      print *, 'Azimuth/Range^2 =          ',r_coef(8) 
      print *, 'Range cubic term =         ',r_coef(9) 
      print *, 'Azimuth cubic term =       ',r_coef(10) 
       
       
      t1 = secnds(t0)
      print *,  'XXX time: ', t1-t0

      do i=1,i_ma
        r_polyV(i) = r_coef(i)
      end do
      
      deallocate(r_ranpos)
      deallocate(r_azpos)
      deallocate(r_sig)
      deallocate(r_off)
      deallocate(r_coef)
      deallocate(r_u)
      deallocate(r_v)
      deallocate(r_w)
      end
      


