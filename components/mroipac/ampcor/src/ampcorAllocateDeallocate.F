        subroutine allocate_locationAcross(dim1)
            use ampcorState
            implicit none 
            integer dim1
            allocate(i_centerxiArr(dim1))
        end

        subroutine deallocate_locationAcross()
            use ampcorState
            deallocate(i_centerxiArr)
        end

        subroutine allocate_locationAcrossOffset(dim1)
            use ampcorState
            implicit none 
            integer dim1
            allocate(r_shftxoscArr(dim1))
        end

        subroutine deallocate_locationAcrossOffset()
            use ampcorState
            deallocate(r_shftxoscArr)
        end

        subroutine allocate_locationDown(dim1)
            use ampcorState
            implicit none 
            integer dim1
            allocate(i_centeryiArr(dim1))
        end

        subroutine deallocate_locationDown()
            use ampcorState
            deallocate(i_centeryiArr)
        end

        subroutine allocate_locationDownOffset(dim1)
            use ampcorState
            implicit none 
            integer dim1
            allocate(r_shftyoscArr(dim1))
        end

        subroutine deallocate_locationDownOffset()
            use ampcorState
            deallocate(r_shftyoscArr)
        end

        subroutine allocate_snrRet(dim1)
            use ampcorState
            implicit none 
            integer dim1
            allocate(r_snrArr(dim1))
        end

        subroutine deallocate_snrRet()
            use ampcorState
            deallocate(r_snrArr)
        end

        subroutine allocate_cov1Ret(dim1)
            use ampcorState
            implicit none 
            integer dim1
            allocate(r_cov1Arr(dim1))
        end

        subroutine deallocate_cov1Ret()
            use ampcorState
            deallocate(r_cov1Arr)
        end

        subroutine allocate_cov2Ret(dim1)
            use ampcorState
            implicit none 
            integer dim1
            allocate(r_cov2Arr(dim1))
        end

        subroutine deallocate_cov2Ret()
            use ampcorState
            deallocate(r_cov2Arr)
        end

        subroutine allocate_cov3Ret(dim1)
            use ampcorState
            implicit none 
            integer dim1
            allocate(r_cov3Arr(dim1))
        end

        subroutine deallocate_cov3Ret()
            use ampcorState
            deallocate(r_cov3Arr)
        end
 
