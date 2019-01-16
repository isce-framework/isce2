!c get number of rows int the output table
       subroutine getNumRows(numRowT)
          use ampcorState
          implicit none
          integer numRowT
          numRowT = numRowTable
       end

       subroutine getLocationAcross(array1d, dim1)
           use ampcorState
           implicit none
           integer dim1,i
           integer, dimension(dim1):: array1d
           do i=1, dim1
                array1d(i) = i_centerxiArr(i)
           enddo
       end


       subroutine getLocationAcrossOffset(array1d,dim1)
           use ampcorState
           implicit none
           integer dim1, i
           real*4, dimension(dim1):: array1d
           do i=1, dim1
                array1d(i) = r_shftxoscArr(i)
           enddo
       end

       subroutine getLocationDown(array1d,dim1)
           use ampcorState
           implicit none
           integer dim1,i
           integer, dimension(dim1):: array1d
           do i=1, dim1
                array1d(i) = i_centeryiArr(i)
           enddo
       end

       subroutine getLocationDownOffset(array1d, dim1)
           use ampcorState
           implicit none
           integer dim1, i
           real*4, dimension(dim1):: array1d
           do i=1, dim1
                array1d(i) = r_shftyoscArr(i)
           enddo
       end

       subroutine getSNR(array1d, dim1)
           use ampcorState
           implicit none
           integer dim1, i
           real*4, dimension(dim1):: array1d
           do i=1, dim1
                array1d(i) = r_snrArr(i)
           enddo
       end

       subroutine getCov1(array1d, dim1)
           use ampcorState
           implicit none
           integer dim1, i
           real*4, dimension(dim1):: array1d
           do i=1, dim1
                array1d(i) = r_cov1Arr(i)
           enddo
       end

       subroutine getCov2(array1d, dim1)
           use ampcorState
           implicit none
           integer dim1, i
           real*4, dimension(dim1):: array1d
           do i=1, dim1
                array1d(i) = r_cov2Arr(i)
           enddo
       end

       subroutine getCov3(array1d, dim1)
           use ampcorState
           implicit none
           integer dim1, i
           real*4, dimension(dim1):: array1d
           do i=1, dim1
                array1d(i) = r_cov3Arr(i)
           enddo
       end

