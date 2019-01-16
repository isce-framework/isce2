!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
! 
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
! 
! http://www.apache.org/licenses/LICENSE-2.0
! 
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
! 
! United States Government Sponsorship acknowledged. This software is subject to
! U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
! (No [Export] License Required except when exporting to an embargoed country,
! end user, or in support of a prohibited end use). By downloading this software,
! the user agrees to comply with all applicable U.S. export laws and regulations.
! The user has the responsibility to obtain export licenses, or other export
! authority as may be required before exporting this software to any 'EAR99'
! embargoed foreign country or citizen of those countries.
!
! Author: Giangi Sacco
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





        subroutine setStdWriter(varInt)
            use resamp_onlyState
            implicit none
            integer*8 varInt
            ptStdWriter = varInt
        end

        subroutine setNumberFitCoefficients(varInt)
            use resamp_onlyState
            implicit none
            integer varInt
            i_ma = varInt
        end

        subroutine setNumberRangeBin(varInt)
            use resamp_onlyState
            implicit none
            integer varInt
            npl = varInt
        end

        subroutine setNumberLines(varInt)
            use resamp_onlyState
            implicit none
            integer varInt
            nl = varInt
        end

        subroutine setFirstLineOffset(varInt)
            use resamp_onlyState
            implicit none
            integer varInt
            istoff = varInt
        end

        subroutine setRadarWavelength(varInt)
            use resamp_onlyState
            implicit none
            real*4 varInt
            WVL = varInt
        end

        subroutine setSlantRangePixelSpacing(varInt)
            use resamp_onlyState
            implicit none
            real*4 varInt
            SLR = varInt
        end

        subroutine setDopplerCentroidCoefficients(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                dopplerCoefficients(i) = array1d(i)
            enddo
        end

        subroutine setLocationAcross1(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                r_ranposV(i) = array1d(i)
            enddo
        end

        subroutine setLocationAcrossOffset1(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                r_ranoffV(i) = array1d(i)
            enddo
        end

        subroutine setLocationDown1(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                r_azposV(i) = array1d(i)
            enddo
        end

        subroutine setLocationDownOffset1(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                r_azoffV(i) = array1d(i)
            enddo
        end

        subroutine setSNR1(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                r_sigV(i) = array1d(i)
            enddo
        end

        subroutine setLocationAcross2(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                r_ranpos2V(i) = array1d(i)
            enddo
        end

        subroutine setLocationAcrossOffset2(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                r_ranoff2V(i) = array1d(i)
            enddo
        end

        subroutine setLocationDown2(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                r_azpos2V(i) = array1d(i)
            enddo
        end

        subroutine setLocationDownOffset2(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                r_azoff2V(i) = array1d(i)
            enddo
        end

        subroutine setSNR2(array1d,dim1)
            use resamp_onlyState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                r_sig2V(i) = array1d(i)
            enddo
        end

