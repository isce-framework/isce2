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





        subroutine setNumberIterations(var)
            use topozeroState
            implicit none
            integer var
            numiter = var
        end

        subroutine setDemWidth(var)
            use topozeroState
            implicit none
            integer var
            idemwidth = var
        end

        subroutine setDemLength(var)
            use topozeroState
            implicit none
            integer var
            idemlength = var
        end

        subroutine setOrbit(corb)
            use topozeroState
            type(orbitType) :: corb
            orbit = corb
        end

        subroutine setFirstLatitude(var)
            use topozeroState
            implicit none
            double precision var
            firstlat = var
        end

        subroutine setFirstLongitude(var)
            use topozeroState
            implicit none
            double precision var
            firstlon = var
        end

        subroutine setDeltaLatitude(var)
            use topozeroState
            implicit none
            double precision var
            deltalat = var
        end

        subroutine setDeltaLongitude(var)
            use topozeroState
            implicit none
            double precision var
            deltalon = var
        end


        subroutine setEllipsoidMajorSemiAxis(var)
            use topozeroState
            implicit none
            double precision var
            major = var
        end

        subroutine setEllipsoidEccentricitySquared(var)
            use topozeroState
            implicit none
            double precision var
            eccentricitySquared = var
        end

        subroutine setLength(var)
            use topozeroState
            implicit none
            integer var
            length = var
        end

        subroutine setWidth(var)
            use topozeroState
            implicit none
            integer var
            width = var
        end

        subroutine setRangePixelSpacing(var)
            use topozeroState
            implicit none
            double precision var
            rspace = var
        end

        subroutine setRangeFirstSample(var)
            use topozeroState
            implicit none
            double precision var
            r0 = var
        end

        subroutine setNumberRangeLooks(var)
            use topozeroState
            implicit none
            integer var
            Nrnglooks = var
        end

        subroutine setNumberAzimuthLooks(var)
            use topozeroState
            implicit none
            integer var
            Nazlooks = var
        end

        subroutine setPegHeading(var)
            use topozeroState
            implicit none
            double precision var
            peghdg = var
        end

        subroutine setPRF(var)
            use topozeroState
            implicit none
            double precision var
            prf = var
        end

        subroutine setSensingStart(var)
            use topozeroState
            implicit none
            double precision var
            t0 = var
        end

        subroutine setRadarWavelength(var)
            use topozeroState
            implicit none
            double precision var
            wvl = var
        end

        subroutine setLatitudePointer(var)
            use topozeroState
            implicit none
            integer*8 var
            latAccessor = var
        end

        subroutine setLongitudePointer(var)
            use topozeroState
            implicit none
            integer*8 var
            lonAccessor = var
        end

        subroutine setHeightPointer(var)
            use topozeroState
            implicit none
            integer*8 var
            heightAccessor = var
        end

        subroutine setLosPointer(var)
            use topozeroState
            implicit none
            integer*8 var
            losAccessor=var
        end

        subroutine setIncPointer(var)
            use topozeroState
            implicit none
            integer*8 var
            incAccessor = var
        end

        subroutine setMaskPointer(var)
            use topozeroState
            implicit none
            integer*8 var
            maskAccessor = var
        end 

        subroutine setLookSide(var)
            use topozeroState
            implicit none
            integer var
            ilrl = var
        end

        subroutine setSecondaryIterations(var)
            use topozeroState
            implicit none
            integer var
            extraiter = var
        end

        subroutine setThreshold(var)
            use topozeroState
            implicit none 
            double precision var
            thresh = var
        end 

        subroutine setMethod(var)
            use topozeroState
            implicit none
            integer var
            method = var
        end

        subroutine setOrbitMethod(var)
            use topozeroState
            implicit none
            integer var
            orbitMethod = var
        end
