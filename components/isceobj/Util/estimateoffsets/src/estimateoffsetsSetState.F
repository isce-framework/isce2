!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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




        subroutine setWindowSize(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            NPTS = varInt
        end

        subroutine setSearchWindowSize(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            NOFF = varInt
        end

        subroutine setZoomWindowSize(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            NDISP = varInt
        end

        subroutine setOversamplingFactor(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            NOVS = varInt
        end

        subroutine setLineLength1(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            len1 = varInt
        end

        subroutine setLineLength2(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            len2 = varInt
        end


        subroutine setFileLength1(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            lines1 = varInt
        end

        subroutine setFileLength2(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            lines2 = varInt
        end

        subroutine setFirstSampleAcross(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            isamp_s = varInt
        end

        subroutine setLastSampleAcross(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            isamp_f = varInt
        end

        subroutine setNumberLocationAcross(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            nloc = varInt
        end

        subroutine setFirstSampleDown(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            isamp_sdn = varInt
        end

        subroutine setLastSampleDown(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            isamp_fdn = varInt
        end

        subroutine setNumberLocationDown(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            nlocdn = varInt
        end

        subroutine setAcrossGrossOffset(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            ioffac = varInt
        end

        subroutine setDownGrossOffset(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            ioffdn = varInt
        end

        subroutine setFirstPRF(varInt)
            use estimateoffsetsState
            implicit none
            real*4 varInt
            prf1 = varInt
        end

        subroutine setSecondPRF(varInt)
            use estimateoffsetsState
            implicit none
            real*4 varInt
            prf2 = varInt
        end

        subroutine setIsComplex1(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            iscpx1 = varInt
        end

        subroutine setIsComplex2(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            iscpx2 = varInt
        end

        subroutine setBand1(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            band1 = varInt
        end
        
        subroutine setBand2(varInt)
            use estimateoffsetsState
            implicit none
            integer varInt
            band2 = varInt
        end

        subroutine setDebugFlag(varString, varInt)
            use estimateoffsetsState
            implicit none
            character*1 varString
            integer*4 varInt
            talk  = ''
            talk(1:varInt) = varString
        end

