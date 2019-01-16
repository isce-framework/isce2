        subroutine getAffineVector(affineVec)
                use fitoffState
                implicit none
                double precision, dimension(*)::affineVec
                
                affineVec(1:6) = x(1:6)
        end

        subroutine getNumberOfRefinedOffsets(varInt)
                use fitoffState
                implicit none
                integer varInt
                varInt = imax
        end


        subroutine  getRefinedLocationAcross(loc)
                use fitoffState
                implicit none 
                double precision, dimension(*)::loc
                integer i
                do i=1,imax
                    loc(i) = x1o(i)
                enddo
        end

        subroutine getRefinedLocationDown(loc)
                use fitoffState
                implicit none
                double precision, dimension(*)::loc
                integer i
                do i=1,imax
                    loc(i)=y1o(i)
                enddo
        end

        subroutine getRefinedLocationAcrossOffset(loc)
                use fitoffState
                implicit none
                double precision, dimension(*)::loc
                integer i
                do i=1,imax
                    loc(i)=x2o(i) - x1o(i)
                enddo
        end

        subroutine getRefinedLocationDownOffset(loc)
                use fitoffState
                implicit none
                double precision, dimension(*)::loc
                integer i
                do i=1,imax
                    loc(i)=y2o(i) - y1o(i)
                enddo
        end


        subroutine getRefinedSNR(loc)
                use fitoffState
                implicit none
                double precision, dimension(*)::loc
                integer i
                do i=1,imax
                    loc(i) = snr(i)
                enddo
        end

        subroutine getRefinedCovAcross(loc)
                use fitoffState
                implicit none
                double precision, dimension(*)::loc
                integer i
                do i=1,imax
                    loc(i) = r_covac(i)
                enddo
        end

        subroutine getRefinedCovDown(loc)
                use fitoffState
                implicit none
                double precision, dimension(*)::loc
                integer i
                do i=1,imax
                    loc(i) = r_covdn(i)
                enddo
        end

        subroutine getRefinedCovCross(loc)
                use fitoffState
                implicit none
                double precision, dimension(*)::loc
                integer i
                do i=1,imax
                    loc(i) = r_covx(i)
                enddo
        end

        
