      byte in(16384)
      complex a(16384),b(16384),prod(16384)
      real acc(16384)
      character*60 file

      print '(a,$)',' Input file: '
      read '(a)',file
      print *,'Line length in real samples, first, number of lines: '
      read *,len,i0,n
      print '(a,$)',' PRF ? '
      read *,prf

      call cfft1d_jpl(16384,a,0)
      call cfft1d_jpl(8192,a,0)
      open(21,file=file,access='direct',recl=len)
      do k=1,16384
         prod(k)=cmplx(0.,0.)
      end do

      do i=i0,i0+n-1
         read(21,rec=i,err=99)(in(k),k=1,len)
         do k=1,len
            kk = in(k)
            if(kk .lt. 0)  kk = kk+256
            a(k)=cmplx(kk-127.5,0.)
         end do
         do k=len+1,16384
            a(k)=cmplx(0.,0.)
         end do
         call cfft1d_jpl(16384,a,-1)
         call cfft1d_jpl(8192,a(8193),1)

c  get second line
         read(21,rec=i+1,err=99)(in(k),k=1,len)
         do k=1,len
            kk = in(k)
            if(kk .lt. 0)  kk = kk+256
            b(k)=cmplx(kk-127.5,0.)
c            b(k)=cmplx((in(k).and.255)-127.5,0.)
         end do
         do k=len+1,16384
            b(k)=cmplx(0.,0.)
         end do
         call cfft1d_jpl(16384,b,-1)
         call cfft1d_jpl(8192,b(8193),1)
         do k=1,8192
            prod(k)=prod(k)+conjg(a(k+8192))*b(k+8192)
         end do
      end do
c  convert to frequencies in cycles
 99   open(22,file='dop.out')
      do i=1,len/2
         k=i
         acc(k)=atan2(aimag(prod(k)),real(prod(k)))
         acc(i)=acc(i)/2/3.14159265
         write(22,*)i,acc(i),acc(i)*prf
      end do

      end
