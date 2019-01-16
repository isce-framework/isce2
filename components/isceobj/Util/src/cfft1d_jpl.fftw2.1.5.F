      subroutine cfft1d_jpl(n,c,dir)

      integer*4  n, dir, ier
      complex*8    c(*)
      integer nmax 
      parameter (nmax = 32768)

cc
cc HP platform and using its library
cc

#ifdef HP

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

#endif


cc
cc SGI platform and using its library
cc

#ifdef SGI

      complex*8 work32k(32768+15),work16k(16384+15)
      complex*8 work8k(8192+15),work4k(4096+15)
      complex*8 work2k(2048+15),work1k(1024+15)
      complex*8 work512(512+15),work256(256+15)
      complex*8 work128(5*128/2),work64(64+15)
      complex*8 work32(32+15),work16(16+15),work8(8+15)
      common /fftwork/work32k,work16k,work8k,work4k,work2k,
     &                work1k,work512,work256,work128,work64,
     &                work32,work16,work8

      if(dir .eq. 0) then
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

#endif

cc
cc SUN platform and using its library
cc

#ifdef SUN

      call cfft1d_sun(n, c, dir)

#endif

cc
cc FFTW
cc

#ifdef FFTW

#include <fftw_f77.i>

#ifdef FFTW64
      integer*8 plani(16),planf(16)   ! for SGI, make with -64, we need integer*8
#else
      integer plani(16),planf(16)
#endif

      complex*8 out(nmax)
      integer i

      save plani,planf

      if(dir.eq.0)then
         do i=3,14
            if(2**i.eq.n)go to 1
         end do
         write(6,*) 'fftw: length unsupported:: ',n
         stop
 1       call fftw_f77_create_plan(planf(i),n,-1,FFTW_ESTIMATE)
         call fftw_f77_create_plan(plani(i),n,1,FFTW_ESTIMATE)
         return
      end if

      if(dir.eq.-1)then
         if(n.eq.8)call fftw_f77_one(planf(3),c,out)
         if(n.eq.16)call fftw_f77_one(planf(4),c,out)
         if(n.eq.32)call fftw_f77_one(planf(5),c,out)
         if(n.eq.64)call fftw_f77_one(planf(6),c,out)
         if(n.eq.128)call fftw_f77_one(planf(7),c,out)
         if(n.eq.256)call fftw_f77_one(planf(8),c,out)
         if(n.eq.512)call fftw_f77_one(planf(9),c,out)
         if(n.eq.1024)call fftw_f77_one(planf(10),c,out)
         if(n.eq.2048)call fftw_f77_one(planf(11),c,out)
         if(n.eq.4096)call fftw_f77_one(planf(12),c,out)
         if(n.eq.8192)call fftw_f77_one(planf(13),c,out)
         if(n.eq.16384)call fftw_f77_one(planf(14),c,out)
      end if
      if(dir.eq. 1)then
         if(n.eq.8)call fftw_f77_one(plani(3),c,out)
         if(n.eq.16)call fftw_f77_one(plani(4),c,out)
         if(n.eq.32)call fftw_f77_one(plani(5),c,out)
         if(n.eq.64)call fftw_f77_one(plani(6),c,out)
         if(n.eq.128)call fftw_f77_one(plani(7),c,out)
         if(n.eq.256)call fftw_f77_one(plani(8),c,out)
         if(n.eq.512)call fftw_f77_one(plani(9),c,out)
         if(n.eq.1024)call fftw_f77_one(plani(10),c,out)
         if(n.eq.2048)call fftw_f77_one(plani(11),c,out)
         if(n.eq.4096)call fftw_f77_one(plani(12),c,out)
         if(n.eq.8192)call fftw_f77_one(plani(13),c,out)
         if(n.eq.16384)call fftw_f77_one(plani(14),c,out)
      end if

      do i = 1 , n
         c(i) = out(i)
      end do

#endif

      return
      end
