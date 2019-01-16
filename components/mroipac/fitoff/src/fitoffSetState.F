        subroutine setStdWriter(varInt)
                use fitoffState
                implicit none
                integer*8 varInt
                ptStdWriter = varInt
        end
        subroutine setNumberLines(varInt)
                use fitoffState
                implicit none
                integer varInt
                imax = varInt
        end

        subroutine setMinPoint(varInt)
                use fitoffState
                implicit none
                integer varInt
                minpoint = varInt
        end

        subroutine setNSig(varDouble)
                use fitoffState
                implicit none
                real*8 varDouble
                nsig = varDouble
        end

        subroutine setMaxRms(varDouble)
                use fitoffState
                implicit none
                real*8 varDouble
                maxrms = varDouble
        end

        subroutine setMinIter(varInt)
                use fitoffState
                implicit none
                integer varInt
                miniter = varInt
        end

        subroutine setMaxIter(varInt)
                use fitoffState
                implicit none
                integer varInt
                maxiter = varInt
        end

        subroutine setL1normFlag(varInt)
                use fitoffState
                implicit none
                integer varInt
                if (varInt.gt.0) then
                    l1norm = .true.
                else
                    l1norm = .false.
                endif
         end

        subroutine setLocationAcross(array1d, dim1)
            use fitoffState
            implicit none
            integer dim1, i
            double precision, dimension(dim1) :: array1d
            do i = 1, dim1
                x1o(i) = array1d(i)
            enddo
        end

        subroutine setLocationDown(array1d, dim1)
            use fitoffState
            implicit none
            integer dim1, i
            double precision, dimension(dim1) :: array1d
            do i=1,dim1
                y1o(i) = array1d(i)
            enddo
        end

        subroutine setLocationAcrossOffset(array1d, dim1)
            use fitoffState
            implicit none
            integer dim1, i
            double precision, dimension(dim1) :: array1d
            do i=1,dim1
                dx(i) = array1d(i)
            enddo
        end

        subroutine setLocationDownOffset(array1d, dim1)
            use fitoffState
            implicit none
            integer dim1, i
            double precision, dimension(dim1) :: array1d
            do i=1,dim1
                dy(i) = array1d(i)
            enddo
        end

        subroutine setSNR(array1d, dim1)
            use fitoffState
            implicit none
            integer dim1, i
            double precision, dimension(dim1) :: array1d
            do i=1,dim1
                snr(i) = array1d(i)
            enddo
        end

        subroutine setCovAcross(array1d, dim1)
            use fitoffState
            implicit none
            integer dim1, i
            double precision, dimension(dim1) :: array1d
            do i=1,dim1
                r_covac(i) = array1d(i)
            enddo
        end

        subroutine setCovDown(array1d, dim1)
            use fitoffState
            implicit none
            integer dim1, i
            double precision, dimension(dim1) :: array1d
            do i=1,dim1
                r_covdn(i) = array1d(i)
            enddo
        end

        subroutine setCovCross(array1d, dim1)
            use fitoffState
            implicit none
            integer dim1, i
            double precision, dimension(dim1) :: array1d
            do i=1,dim1
                r_covx(i) = array1d(i)
            enddo
        end




