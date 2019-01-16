!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!#
!#
!# Author: Piyush Agram
!# Copyright 2013, by the California Institute of Technology. ALL RIGHTS RESERVED.
!# United States Government Sponsorship acknowledged.
!# Any commercial use must be negotiated with the Office of Technology Transfer at
!# the California Institute of Technology.
!# This software may be subject to U.S. export control laws.
!# By accepting this software, the user agrees to comply with all applicable U.S.
!# export laws and regulations. User has the responsibility to obtain export licenses,
!# or other export authority as may be required before exporting such information to
!# foreign countries or providing access to foreign persons.
!#
!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine allocate_LocationAcross()
            use fitoffState
            allocate(x1o(imax))
            allocate(x2o(imax))
            x1o = 0
            x2o = 0
        end

        subroutine deallocate_LocationAcross()
            use fitoffState
            deallocate(x1o)
            deallocate(x2o)
        end

        subroutine allocate_LocationDown()
            use fitoffState
            allocate(y1o(imax))
            allocate(y2o(imax))
            y1o = 0
            y2o = 0
        end

        subroutine deallocate_LocationDown()
            use fitoffState
            deallocate(y1o)
            deallocate(y2o)
        end

        subroutine allocate_LocationAcrossOffset()
            use fitoffState
            allocate(dx(imax))
            dx = 0
        end

        subroutine deallocate_LocationAcrossOffset()
            use fitoffState
            deallocate(dx)
        end

        subroutine allocate_LocationDownOffset()
            use fitoffState
            allocate(dy(imax))
            dy = 0
        end

        subroutine deallocate_LocationDownOffset()
            use fitoffState
            deallocate(dy)
        end

        subroutine allocate_SNR()
            use fitoffState
            allocate(snr(imax))
            snr = 0
        end

        subroutine deallocate_SNR()
            use fitoffState
            deallocate(snr)
        end

        subroutine allocate_Covariance()
            use fitoffState
            allocate(r_covac(imax))
            allocate(r_covdn(imax))
            allocate(r_covx(imax))
            r_covac = 0
            r_covdn = 0
            r_covx = 0
        end

        subroutine deallocate_Covariance()
            use fitoffState
            deallocate(r_covac)
            deallocate(r_covdn)
            deallocate(r_covx)
        end
