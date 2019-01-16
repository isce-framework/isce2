      subroutine allocate_r_fd(dim1)
       use dopplerState
       implicit none
       integer dim1
       dim1_r_fd = dim1
       allocate(r_fd(dim1_r_fd))
      end

      subroutine deallocate_r_fd
       use dopplerState
       implicit none
       deallocate(r_fd)
      end
