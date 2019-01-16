      subroutine cffts(z,n,incn,m,incm,iopt,ier)
      integer n, incn, m, incm, iopt, ier
      complex*8 z(*)

      integer nmax, nprev
      parameter (nmax = 32768)
      complex*8 work(nmax+15)
      integer job

cc
cc SGI platform
cc

#if defined(SGI) || defined(HAVE_CFFT1D)
c     NOTE: Above condition must match same test in cfft1d_JPL.F

c     The should be updated in the future to use SCSL routines

      save nprev, work
      data nprev/-1/

      if (n.gt.nmax) then
         print *, 'cffts: transform length exceeds maximum (',n,' > ',nmax,')'
         ier = -1
         return
      endif

      if (iopt.ge.0) job = -1	! forward transform
      if (iopt.lt.0) job =  1	! scaled inverse transform

      if (n.ne.nprev) then
         call cfftm1di(n,work)
         nprev = n
      endif
      call cfftm1d(job,n,m,z,incn,incm,work)

      if (job.eq.1) then
         do i = 1, m
            call cscal1d(n,1.0/n,z(1 + (i-1)*incm),incn)
         enddo
      endif

      ier = 0

#else  ! for SUN or FFTW

      complex*8 tmp(nmax)
      save nprev, work
      data nprev/-1/

      if (n.gt.nmax) then
         print *, 'cffts: transform length exceeds maximum (',n,' > ',nmax,')'
         ier = -1
         return
      endif

      if (iopt.ge.0) job = -1   ! forward transform
      if (iopt.lt.0) job =  1   ! scaled inverse transform

      do i=1,m
        do j=1,n,incn
          tmp(j)=z((i-1)*incm+j)
        enddo
        call cfft1d_jpl(n, tmp, job)
        do j=1,n,incn
          z((i-1)*incm+j)=tmp(j)
        enddo
      enddo

#endif

      return
      end
