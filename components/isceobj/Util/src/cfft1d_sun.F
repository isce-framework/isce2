c  *** JPL/Caltech Repeat Orbit Interferometry (ROI) Package ***

      subroutine cfft1d_sun(n,c,dir)

      implicit none

      integer*4  malloc, nold, n, dir, i
      complex*8  c(*)

      real*4 WORK(1)

      external cfftf, cfftb, cffti

c**** define pointer for work array

      pointer (pWORK, WORK)   
      save pWORK

c**** define flag that determines if array needs to be reinitialized

      data nold /0/
      save nold

c**** Initialize workspace and fft routine to save CPU time later.
c**** This is done when dir=0 or when the fft length has changed.

      if ((n .ne. nold) .or. (dir .eq. 0)) then

         if (nold .ne. 0) then
            call free(pWORK)
         end if

         pWORK = malloc( (4*n+30)*4 )

         if (pWORK .eq. 0) then
            write(*,*) 'cfft1d could not allocate memory'
            stop
         end if

         call cffti(n,WORK)

         nold = n

      end if

c**** forward transform with no normalization.  exp(+ikx)

      if (dir .eq. -1) then
         call cfftf(n,c,WORK)
      end if

c**** inverse transform with normalization.  exp(-ikx)

      if (dir .eq. 1) then
         call cfftb(n,c,WORK)
         do i = 1, n
            c(i) = c(i) / n
         end do
      end if

      return
      end

