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




        subroutine setWindowSizeWidth(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            NPTSW = varInt
        end

        subroutine setWindowSizeHeight(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            NPTSH = varInt
        end

        subroutine setSearchWindowSizeWidth(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            NOFFW = varInt
        end

        subroutine setSearchWindowSizeHeight(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            NOFFH = varInt
        end

        subroutine setZoomWindowSize(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            NDISP = varInt
        end

        subroutine setOversamplingFactor(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            NOVS = varInt
        end

        subroutine setLineLength1(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            len1 = varInt
        end

        subroutine setLineLength2(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            len2 = varInt
        end


        subroutine setFileLength1(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            lines1 = varInt
        end

        subroutine setFileLength2(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            lines2 = varInt
        end

        subroutine setFirstSampleAcross(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            isamp_s = varInt+1
        end

        subroutine setLastSampleAcross(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            isamp_f = varInt+1
        end

        subroutine setSkipSampleAcross(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            iskipac = varInt
        end

        subroutine setFirstSampleDown(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            isamp_sdn = varInt+1
        end

        subroutine setLastSampleDown(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            isamp_fdn = varInt+1
        end

        subroutine setSkipSampleDown(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            iskipdn = varInt
        end

        subroutine setAcrossGrossOffset(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            ioffac = varInt
        end

        subroutine setDownGrossOffset(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            ioffdn = varInt
        end

        subroutine setScaleFactorX(varInt)
            use denseoffsetsState
            implicit none
            real*4 varInt
            scalex = varInt
        end

        subroutine setScaleFactorY(varInt)
            use denseoffsetsState
            implicit none
            real*4 varInt
            scaley = varInt
        end

        subroutine setIsComplex1(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            iscpx1 = varInt
        end

        subroutine setIsComplex2(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            iscpx2 = varInt
        end

        subroutine setBand1(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            band1 = varInt
        end
        
        subroutine setBand2(varInt)
            use denseoffsetsState
            implicit none
            integer varInt
            band2 = varInt
        end

        subroutine setDebugFlag(varString, varInt)
            use denseoffsetsState
            implicit none
            character*1 varString
            integer*4 varInt
            talk  = ''
            talk(1:varInt) = varString
        end

        subroutine setNormalizeFlag(varInt)
            use denseoffsetsState
            implicit none 
            integer*4 varInt
            if(varInt.ne.0) then
                normalize = .True.
            else
                normalize = .False.
            endif
        end subroutine setNormalizeFlag

