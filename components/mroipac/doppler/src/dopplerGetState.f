      subroutine get_r_fd(array1d,dim1)
       use dopplerState
       implicit none
       integer dim1,i
       real*8, dimension(dim1) :: array1d

       do i=1,dim1
        array1d(i) = r_fd(i)
       enddo

      end
