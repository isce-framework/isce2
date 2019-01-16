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





        subroutine setWidth(varInt)
            use readOrbitPulseERSState
            implicit none
            integer varInt
            width = varInt
        end

        subroutine setICUoffset(varInt)
            use readOrbitPulseERSState
            implicit none
            integer varInt
            ICUoffset = varInt
        end

        subroutine setNumberLines(varInt)
            use readOrbitPulseERSState
            implicit none
            integer varInt
            lines = varInt
        end

        subroutine setSatelliteUTC(varInt)
            use readOrbitPulseERSState
            implicit none
            double precision varInt
            sat_utc = varInt
        end

        subroutine setPRF(varInt)
            use readOrbitPulseERSState
            implicit none
            double precision varInt
            prf = varInt
        end

        subroutine setDeltaClock(varInt)
            use readOrbitPulseERSState
            implicit none
            double precision varInt
            dICU = varInt
        end

        subroutine setEncodedBinaryTimeCode(varInt)
            use readOrbitPulseERSState
            implicit none
            integer*8 varInt
            ICUb = varInt
        end
