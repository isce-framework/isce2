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





        subroutine setEllipsoidMajorSemiAxis(varInt)
            use geo2rdrState
            implicit none
            double precision varInt
            majorSemiAxis = varInt
        end subroutine

        subroutine setEllipsoidEccentricitySquared(varInt)
            use geo2rdrState
            implicit none
            double precision varInt
            eccentricitySquared = varInt
        end subroutine

        subroutine setRangePixelSpacing(varInt)
            use geo2rdrState
            implicit none
            real*8 varInt
            drho = varInt
        end subroutine

        subroutine setLookSide(varInt)
            use geo2rdrState
            implicit none
            integer varInt
            ilrl = varInt
        end subroutine

        subroutine setRangeFirstSample(varInt)
            use geo2rdrState
            implicit none
            double precision varInt
            rho0 = varInt
        end subroutine

        subroutine setDopplerAccessor(varInt)
            use geo2rdrState
            implicit none
            type(poly1dType) ::  varInt
            dopAcc = varInt
        end subroutine

        subroutine setPRF(varInt)
            use geo2rdrState
            implicit none
            real*8 varInt
            prf = varInt
        end subroutine

        subroutine setRadarWavelength(varInt)
            use geo2rdrState
            implicit none
            real*8 varInt
            wvl = varInt
        end subroutine

        subroutine setSensingStart(varInt)
            use geo2rdrState
            implicit none
            double precision varInt
            t0 = varInt
        end subroutine

        subroutine setLength(varInt)
            use geo2rdrState
            implicit none
            integer varInt
            length = varInt
        end subroutine

        subroutine setWidth(varInt)
            use geo2rdrState
            implicit none
            integer varInt
            width = varInt
        end subroutine

        subroutine setDemWidth(varInt)
            use geo2rdrState
            implicit none
            integer varInt
            demwidth = varInt
        end subroutine

        subroutine setDemLength(varInt)
            use geo2rdrState
            implicit none
            integer varInt
            demlength = varInt
        end subroutine

        subroutine setOrbit(corb)
            use geo2rdrState
            implicit none
            type(orbitType):: corb
            orbit = corb
        end subroutine

        subroutine setNumberRangeLooks(varInt)
            use geo2rdrState
            implicit none
            integer :: varInt
            Nrnglooks = varInt
        end subroutine

        subroutine setNumberAzimuthLooks(varInt)
            use geo2rdrState
            implicit none
            integer :: varInt
            Nazlooks = varInt
        end subroutine

        subroutine setBistaticFlag(varInt)
            use geo2rdrState
            implicit none
            integer :: varInt

            if (varInt .eq. 0) then
                bistatic = .False.
            else
                bistatic = .True.
            endif
        end subroutine

        subroutine setOrbitMethod(varInt)
            use geo2rdrState
            implicit none
            integer :: varInt

            orbitMethod = varInt
        end subroutine
