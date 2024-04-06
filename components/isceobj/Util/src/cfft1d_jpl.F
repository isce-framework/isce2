      subroutine cfft1d_jpl(n,c,dir)
      use, intrinsic :: iso_c_binding
      implicit none
      integer*4  n, dir, ier
      complex*8    c(*)
      integer nmax
      parameter (nmax = 131072)  !32768)
      interface
        integer(C_INT) function sfftw_import_wisdom_from_filename(filename) bind(C, name='sfftw_import_wisdom_from_filename')
        use, intrinsic :: iso_c_binding
        character(C_CHAR), dimension(*), intent(in) :: filename
        end function sfftw_import_wisdom_from_filename
      end interface

cc
cc HP platform and using its library
cc

#if defined(HP) || defined(HAVE_C1DFFT)
      real*4 work128k(5*131072/2)
      real*4 work64k(5*65536/2)
      real*4 work32k(5*32768/2)
      real*4 work16(5*16384/2),work8(5*8192/2),work4(5*4096/2)
      real*4 work2(5*2048/2),work1(5*1024/2)
      real*4 work32(5*32/2),work64(5*64/2),work128(5*128/2)
      real*4 work256(5*256/2),work512(5*512/2)
      real*4 work16nok(5*16/2),work8nok(5*8/2)
      save work1,work2,work4,work8,work16
      save work32,work64,work128,work256,work512
      save work16nok,work8nok


      if(dir .eq. 0) then
         if(n.eq.131072)call c1dfft(c,n,work128k,-3,ier)
         if(n.eq.65536)call c1dfft(c,n,work64k,-3,ier)
         if(n.eq.32768)call c1dfft(c,n,work32k,-3,ier)
         if(n.eq.16384)call c1dfft(c,n,work16,-3,ier)
         if(n.eq.8192)call c1dfft(c,n,work8,-3,ier)
         if(n.eq.4096)call c1dfft(c,n,work4,-3,ier)
         if(n.eq.2048)call c1dfft(c,n,work2,-3,ier)
         if(n.eq.1024)call c1dfft(c,n,work1,-3,ier)
         if(n.eq.512)call c1dfft(c,n,work512,-3,ier)
         if(n.eq.256)call c1dfft(c,n,work256,-3,ier)
         if(n.eq.128)call c1dfft(c,n,work128,-3,ier)
         if(n.eq.64)call c1dfft(c,n,work64,-3,ier)
         if(n.eq.32)call c1dfft(c,n,work32,-3,ier)
         if(n.eq.16)call c1dfft(c,n,work16nok,-3,ier)
         if(n.eq.8)call c1dfft(c,n,work8nok,-3,ier)
         if (ier.ne.0)then
            write(6,*) 'mlib cfft1d init error, ier= ',ier,n
            stop
         end if
         return
      end if
      if(n.eq.131072)call c1dfft(c,n,work128k,-dir,ier)
      if(n.eq.65536)call c1dfft(c,n,work64k,-dir,ier)
      if(n.eq.32768)call c1dfft(c,n,work32k,-dir,ier)
      if(n.eq.16384)call c1dfft(c,n,work16,-dir,ier)
      if(n.eq.8192)call c1dfft(c,n,work8,-dir,ier)
      if(n.eq.4096)call c1dfft(c,n,work4,-dir,ier)
      if(n.eq.2048)call c1dfft(c,n,work2,-dir,ier)
      if(n.eq.1024)call c1dfft(c,n,work1,-dir,ier)
      if(n.eq.512)call c1dfft(c,n,work512,-dir,ier)
      if(n.eq.256)call c1dfft(c,n,work256,-dir,ier)
      if(n.eq.128)call c1dfft(c,n,work128,-dir,ier)
      if(n.eq.64)call c1dfft(c,n,work64,-dir,ier)
      if(n.eq.32)call c1dfft(c,n,work32,-dir,ier)
      if(n.eq.16)call c1dfft(c,n,work16nok,-dir,ier)
      if(n.eq.8)call c1dfft(c,n,work8nok,-dir,ier)
      if(ier.ne.0)then
         write(6,*) 'mlib cfft1d exec error, ier= ',ier
         stop
      end if


cc
cc SGI platform and using its library
cc

#elif defined(SGI) || defined(HAVE_CFFT1D)
c     NOTE: if above condition changed, also need to update cffts.F

c     The should be updated in the future to use SCSL routines
      complex*8 work128k(131072+15)
      complex*8 work64k(65536+15)
      complex*8 work32k(32768+15),work16k(16384+15)
      complex*8 work8k(8192+15),work4k(4096+15)
      complex*8 work2k(2048+15),work1k(1024+15)
      complex*8 work512(512+15),work256(256+15)
      complex*8 work128(5*128/2),work64(64+15)
      complex*8 work32(32+15),work16(16+15),work8(8+15)
      common /fftwork/work128k,work64k,work32k,work16k,work8k,work4k,work2k,
     &                work1k,work512,work256,work128,work64,
     &                work32,work16,work8




      if(dir .eq. 0) then
         if (n.eq.131072) call cfft1di(n,work128k)
         if (n.eq.65536) call cfft1di(n,work64k)
         if (n.eq.32768) call cfft1di(n,work32k)
         if (n.eq.16384) call cfft1di(n,work16k)
         if (n.eq. 8192) call cfft1di(n,work8k)
         if (n.eq. 4096) call cfft1di(n,work4k)
         if (n.eq. 2048) call cfft1di(n,work2k)
         if (n.eq. 1024) call cfft1di(n,work1k)
         if (n.eq.  512) call cfft1di(n,work512)
         if (n.eq.  256) call cfft1di(n,work256)
         if (n.eq.  128) call cfft1di(n,work128)
         if (n.eq.   64) call cfft1di(n,work64)
         if (n.eq.   32) call cfft1di(n,work32)
         if (n.eq.   16) call cfft1di(n,work16)
         if (n.eq.    8) call cfft1di(n,work8)
         return
      end if

      if (n.eq.131072) call cfft1d(dir,n,c,1,work128k)
      if (n.eq.65536) call cfft1d(dir,n,c,1,work64k)
      if (n.eq.32768) call cfft1d(dir,n,c,1,work32k)
      if (n.eq.16384) call cfft1d(dir,n,c,1,work16k)
      if (n.eq. 8192) call cfft1d(dir,n,c,1,work8k)
      if (n.eq. 4096) call cfft1d(dir,n,c,1,work4k)
      if (n.eq. 2048) call cfft1d(dir,n,c,1,work2k)
      if (n.eq. 1024) call cfft1d(dir,n,c,1,work1k)
      if (n.eq.  512) call cfft1d(dir,n,c,1,work512)
      if (n.eq.  256) call cfft1d(dir,n,c,1,work256)
      if (n.eq.  128) call cfft1d(dir,n,c,1,work128)
      if (n.eq.   64) call cfft1d(dir,n,c,1,work64)
      if (n.eq.   32) call cfft1d(dir,n,c,1,work32)
      if (n.eq.   16) call cfft1d(dir,n,c,1,work16)
      if (n.eq.    8) call cfft1d(dir,n,c,1,work8)

c      if (dir.eq.1) call cscal1d(n,1.0/n,c,1)


cc
cc SUN platform and using its library
cc

#elif defined(SUN) || defined(HAVE_CFFTF)


C Sun WIPSpro Fortran 77
      complex*8 work128k(4*131072+15)
      complex*8 work64k(4*65536+15)
      complex*8 work32k(4*32768+15),work16k(4*16384+15)
      complex*8 work8k(4*8192+15),work4k(4*4096+15)
      complex*8 work2k(4*2048+15),work1k(4*1024+15)
      complex*8 work512(4*512+15),work256(4*256+15)
      complex*8 work128(4*128+15),work64(4*64+15)
      complex*8 work32(4*32+15),work16(4*16+15),work8(4*8+15)
      common /fftwork/work128k,work64k,work32k,work16k,work8k,work4k,work2k,
     &                work1k,work512,work256,work128,work64,
     &                work32,work16,work8




      if(dir .eq. 0) then
         if (n.eq.131072) call cffti(n,work128k)
         if (n.eq.65536) call cffti(n,work64k)
         if (n.eq.32768) call cffti(n,work32k)
         if (n.eq.16384) call cffti(n,work16k)
         if (n.eq. 8192) call cffti(n,work8k)
         if (n.eq. 4096) call cffti(n,work4k)
         if (n.eq. 2048) call cffti(n,work2k)
         if (n.eq. 1024) call cffti(n,work1k)
         if (n.eq.  512) call cffti(n,work512)
         if (n.eq.  256) call cffti(n,work256)
         if (n.eq.  128) call cffti(n,work128)
         if (n.eq.   64) call cffti(n,work64)
         if (n.eq.   32) call cffti(n,work32)
         if (n.eq.   16) call cffti(n,work16)
         if (n.eq.    8) call cffti(n,work8)
         return
      end if
      if (n.eq.131072) call cfftf(n,c,work128k)
      if (n.eq.65536) call cfftf(n,c,work64k)
      if (n.eq.32768) call cfftf(n,c,work32k)
      if (n.eq.16384) call cfftf(n,c,work16k)
      if (n.eq. 8192) call cfftf(n,c,work8k)
      if (n.eq. 4096) call cfftf(n,c,work4k)
      if (n.eq. 2048) call cfftf(n,c,work2k)
      if (n.eq. 1024) call cfftf(n,c,work1k)
      if (n.eq.  512) call cfftf(n,c,work512)
      if (n.eq.  256) call cfftf(n,c,work256)
      if (n.eq.  128) call cfftf(n,c,work128)
      if (n.eq.   64) call cfftf(n,c,work64)
      if (n.eq.   32) call cfftf(n,c,work32)
      if (n.eq.   16) call cfftf(n,c,work16)
      if (n.eq.    8) call cfftf(n,c,work8)

cbjs      Was
cbjs      call cfft1d_sun(n, c, dir)
cbjs      but was dumping core on free()
cbjs      so trying sun FFT from joanne.
cbjs      also lookes like cfft1d_sun is normalized, while
cbjs      calls in here are not.  Suspect bug.

cc
cc FFTW
cc

#elif defined(FFTW) || defined(HAVE_FFTW)
c     NOTE: if above condition changed, also need to update fftw3stub.c

#include <fftw3.f>

      integer*8 plani(16),planf(16),planFlagCreate(16),planFlagDestroy(16)
      complex*8 in(nmax)
      integer i, j, length
      character(len=1024) :: wisdomFile
      integer(c_int) :: ret
      logical, save :: firstTime = .true.
      save plani,planf, planFlagCreate, planFlagDestroy, in
      if(firstTime .eqv. .true.) then
          do i = 2, 16
              planFlagCreate(i) = 0
              planFlagDestroy(i) = 0
          enddo
          call get_environment_variable("WISDOM_FILE", wisdomFile,length)
          print*,"wisdomFile, length = ", wisdomFile(1:length), length
          if (length .ne. 0) then
              ret = sfftw_import_wisdom_from_filename(wisdomFile(1:length) // C_NULL_CHAR)
              write(6,*) ret

              if (ret .eq. 0) then
                  stop 'cannot import fftw wisdom file'
              endif
          endif
          firstTime = .false.
      endif

!c replace giant call block with direct call using planf(i)

      if(dir.eq.0)then
!c move i calculation to top of function
         do i=2,17
            if(2**i.eq.n)go to 1
         end do
         write(6,*) 'fftw: length unsupported:: ',n
         stop
 1       do j= 1 , n
            in(j) = cmplx(0.,0.)
         end do
         if(planFlagCreate(i) .eq. 0) then
             call sfftw_plan_dft_1d(planf(i),n,in,in,FFTW_FORWARD,FFTW_MEASURE)
             call sfftw_plan_dft_1d(plani(i),n,in,in,FFTW_BACKWARD,FFTW_MEASURE)
             planFlagCreate(i) = 1
         endif
         return
      end if

      if(dir.eq.-1)then
         if(n.eq.4)call sfftw_execute_dft(planf(2),c,c)
         if(n.eq.8)call sfftw_execute_dft(planf(3),c,c)
         if(n.eq.16)call sfftw_execute_dft(planf(4),c,c)
         if(n.eq.32)call sfftw_execute_dft(planf(5),c,c)
         if(n.eq.64)call sfftw_execute_dft(planf(6),c,c)
         if(n.eq.128)call sfftw_execute_dft(planf(7),c,c)
         if(n.eq.256)call sfftw_execute_dft(planf(8),c,c)
         if(n.eq.512)call sfftw_execute_dft(planf(9),c,c)
         if(n.eq.1024)call sfftw_execute_dft(planf(10),c,c)
         if(n.eq.2048)call sfftw_execute_dft(planf(11),c,c)
         if(n.eq.4096)call sfftw_execute_dft(planf(12),c,c)
         if(n.eq.8192)call sfftw_execute_dft(planf(13),c,c)
         if(n.eq.16384)call sfftw_execute_dft(planf(14),c,c)
         if(n.eq.32768)call sfftw_execute_dft(planf(15),c,c)
         if(n.eq.65536)call sfftw_execute_dft(planf(16),c,c)
         if(n.eq.131072)call sfftw_execute_dft(planf(17),c,c)
      end if
      if(dir.eq. 1)then
         if(n.eq.4)call sfftw_execute_dft(plani(2),c,c)
         if(n.eq.8)call sfftw_execute_dft(plani(3),c,c)
         if(n.eq.16)call sfftw_execute_dft(plani(4),c,c)
         if(n.eq.32)call sfftw_execute_dft(plani(5),c,c)
         if(n.eq.64)call sfftw_execute_dft(plani(6),c,c)
         if(n.eq.128)call sfftw_execute_dft(plani(7),c,c)
         if(n.eq.256)call sfftw_execute_dft(plani(8),c,c)
         if(n.eq.512)call sfftw_execute_dft(plani(9),c,c)
         if(n.eq.1024)call sfftw_execute_dft(plani(10),c,c)
         if(n.eq.2048)call sfftw_execute_dft(plani(11),c,c)
         if(n.eq.4096)call sfftw_execute_dft(plani(12),c,c)
         if(n.eq.8192)call sfftw_execute_dft(plani(13),c,c)
         if(n.eq.16384)call sfftw_execute_dft(plani(14),c,c)
         if(n.eq.32768)call sfftw_execute_dft(plani(15),c,c)
         if(n.eq.65536)call sfftw_execute_dft(plani(16),c,c)
         if(n.eq.131072)call sfftw_execute_dft(plani(17),c,c)
      end if
      if(dir .eq. 2) then
         do i=2,17
            if(2**i.eq.n)go to 10
         end do
         write(6,*) 'fftw: length unsupported:: ',n
         stop
         !forward and inverse were both crated
10       if(planFlagDestroy(i)  .eq. 0) then
              planFlagDestroy(i) = 1
              planFlagCreate(i) = 0
              call sfftw_destroy_plan(planf(i))
              call sfftw_destroy_plan(plani(i))
         endif
         !call sfftw_cleanup()
         return


      end if


#else
c     NO FFT routine has been specified
c     force compilation to fail, with below "ABORT" syntax error
c     rather than old behavior, of having this routine
c     silently do nothing
      ABORT NO FFT ROUTINE DEFINED
      stop  "NO FFT ROUTINE DEFINED"

#endif

      return
      end
