      subroutine allocate_acc(dim1)
       use dopiqState
       implicit none
       integer dim1
       dim1_acc = dim1
       allocate(acc(dim1))
      end

      subroutine deallocate_acc()
       use dopiqstate
       deallocate(acc)
      end
