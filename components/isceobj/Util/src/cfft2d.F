      subroutine cfft2d(n1,n2,arr,m,dir)

      integer*4 n1, n2, dir, m
      complex*8 arr(m,n2), dum(16384)

      do i = 1 , n2
         call cfft1d_jpl(n1,arr(1,i),dir)
      end do
      do i = 1 , n1
         do j = 1 , n2
            dum(j) = arr(i,j)
         end do
         call cfft1d_jpl(n2,dum,dir)
         do j = 1 , n2
             arr(i,j) = dum(j)
         end do
      end do
      return
      end

