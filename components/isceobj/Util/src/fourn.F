      subroutine fourn(data,nn,ndim,isign)
      complex data(*), d(16384)
      integer nn(2),n,ndim,is

      is = -isign
      n = nn(1)
      do i = 1 , nn(1)
         call cfft1d_jpl(nn(2),data(1+nn(2)*(i-1)),is)
      end do
      do i = 1 , nn(2)
         do j = 1 , nn(1)
            d(j) = data(i+nn(2)*(j-1))
         end do
         call cfft1d_jpl(nn(1),d,is)
         do j = 1 , nn(1)
            if(is .eq. 1)then
               d(j) = d(j)*nn(1)*nn(2)
           endif
            data(i+nn(2)*(j-1)) = d(j)
         end do
      end do

      return
      end

