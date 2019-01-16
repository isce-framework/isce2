!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine setWidth(varInt)
            use aikimaState
            implicit none

            integer varInt
            nac = varInt
        end

        subroutine setLength(varInt)
            use aikimaState
            implicit none
            integer varInt
            ndn = varInt
        end

        subroutine setFirstPixelAcross(varInt)
            use aikimaState
            implicit none
            integer varInt
            i_xmin = varInt
        end

        subroutine setLastPixelAcross(varInt)
            use aikimaState
            implicit none
            integer varInt
            i_xmax = varInt
        end

        subroutine setFirstLineDown(varInt)
            use aikimaState
            implicit none
            integer varInt
            i_ymin = varInt
        end

        subroutine setLastLineDown(varInt)
            use aikimaState
            implicit none
            integer varInt
            i_ymax = varInt
        end

        subroutine setBlockSize(varInt)
            use aikimaState
            implicit none
            integer varInt
            i_skip = varInt
        end

        subroutine setPadSize(varInt)
            use aikimaState
            implicit none
            integer varInt
            i_padn = varInt
        end

        subroutine setNumberPtsPartial(varInt)
            use aikimaState
            implicit none
            integer varInt
            i_ncp = varInt
        end

        subroutine setPrintFlag(varInt)
            use aikimaState
            implicit none
            integer varInt
            i_pflag = varInt
        end

        subroutine setThreshold(varInt)
            use aikimaState
            implicit none
            real*4 varInt
            r_thres = varInt
        end
            

