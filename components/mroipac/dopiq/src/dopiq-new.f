      subroutine dopiq(rawAccessor)

      use dopiqState
      implicit none
      integer*8 rawAccessor
      integer*1, dimension(:), allocatable :: in
      integer cnt,i,k
      complex*8, dimension(:), allocatable ::  a, b, prod
      real*8, parameter :: pi = 4.d0*atan(1.d0)

      allocate(  in( last ))
      allocate(   a( ((last-hdr)/2)+1 ))
      allocate(   b( ((last-hdr)/2)+1 ))
      allocate(prod( ((last-hdr)/2)+1 ))

      do k=1, ((last-hdr)/2)+1
         prod(k)=cmplx(0.,0.)
      end do
      cnt = 0
      do i=i0,i0+n-1
          cnt = i
         call getLine(rawAccessor,in,cnt)
         if (cnt.eq.-1) goto 99
         do k=hdr+1,last, 2
            a((k-hdr)/2+1)=cmplx(iand(int(in(k)),255)-xmn,iand(int(in(k+1)),255)-xmn)
         end do
c     get second line
        cnt = i+1 
        call getLine(rawAccessor,in,cnt)
         if(cnt.eq.-1) goto 99
         do k=hdr+1,last,2
            b((k-hdr)/2+1)=cmplx(iand(int(in(k)),255)-xmn,iand(int(in(k+1)),255)-xmn)
         end do
         do k=1, (len-hdr)/2
            prod(k)=prod(k)+conjg(a(k))*b(k)
         end do
      end do

c     convert to frequencies in cycles
 99   do k=1, (last-hdr)/2
         acc(k)=atan2(aimag(prod(k)),real(prod(k)))
         acc(k)=acc(k)/2/pi
      end do

      deallocate(in)
      deallocate(a)
      deallocate(b)
      deallocate(prod)

      end

