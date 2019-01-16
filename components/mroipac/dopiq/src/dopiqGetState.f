      subroutine get_acc(array1d,dim1)
       use dopiqState
       implicit none
       integer dim1,i
       real*8, dimension(dim1) :: array1d
       do i = 1,dim1
        array1d(i) = acc(i)
       enddo
      end

