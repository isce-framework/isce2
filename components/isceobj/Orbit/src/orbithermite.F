c  orbithermite - hermite polynomial interpolation of alos orbits

      subroutine orbithermite(x,v,t,time,xx,vv)

c  inputs
c   x - 3x4 matrix of positions at four times
c   v - 3x4 matrix of velocities
c   t - 4-vector of times for each of the above data points
c   time - time to interpolate orbit to

c  outputs
c   xx - position at time time
c   vv - velocity at time time

      implicit none
      integer n
      parameter (n=4)
      integer i,j,k
      double precision x(3,n),v(3,n),t(n),time,xx(3),vv(3)
      double precision h(n),hdot(n),f0(n),f1(n),g0(n),g1(n)
      double precision sum,product

c  equations from nec memo
       
      do i=1,n
         f1(i)=time-t(i)
         sum=0.0d0
         do j=1,n
            if(j.ne.i)sum=sum+1.0d0/(t(i)-t(j))
         end do
         f0(i)=1.0d0-2.0d0*(time-t(i))*sum
      end do

      do i=1,n
         product=1.0d0
         do k=1,n
            if(k.ne.i)product=product*(time-t(k))/(t(i)-t(k))
         end do
         h(i)=product
         sum=0.0d0
         do j=1,n
            product=1.0d0
            do k=1,n
               if(k.ne.i.and.k.ne.j)product=product*(time-t(k))/(t(i)-t(k))
            end do
            if(j.ne.i)sum=sum+1.0d0/(t(i)-t(j))*product
         end do
         hdot(i)=sum
      end do

      do i=1,n
         g1(i)=h(i)+2.0d0*(time-t(i))*hdot(i)
         sum=0.0d0
         do j=1,n
            if(i.ne.j)sum=sum+1./(t(i)-t(j))
         end do
         g0(i)=2.0d0*(f0(i)*hdot(i)-h(i)*sum)
      end do

      do k=1,3
         sum=0.0d0
         do i=1,n
            sum=sum+(x(k,i)*f0(i)+v(k,i)*f1(i))*h(i)*h(i)
         end do
         xx(k)=sum

         sum=0.0d0
         do i=1,n
            sum=sum+(x(k,i)*g0(i)+v(k,i)*g1(i))*h(i)  !*h(i) extra in pdf
         end do
         vv(k)=sum
      end do

c$$$      print *,'f0',f0
c$$$      print *,'f1',f1
c$$$      print *,'g0',g0
c$$$      print *,'g1',g1
c$$$      print *,'h ',h
c$$$      print *,'hd',hdot
c$$$      print *
c$$$
c$$$      write(10,*)f0,f1,g0,g1,h,hdot,xx,vv

      return
      end

