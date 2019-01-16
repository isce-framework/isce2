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
            use geocodeState
            implicit none
            integer*8 varInt
            ptStdWriter = varInt
        end

        subroutine setEllipsoidMajorSemiAxis(varInt)
            use geocodeState
            implicit none
            double precision varInt
            majorSemiAxis = varInt
        end

        subroutine setEllipsoidEccentricitySquared(varInt)
            use geocodeState
            implicit none
            double precision varInt
            eccentricitySquared = varInt
        end

        subroutine setMinimumLatitude(varInt)
            use geocodeState
            implicit none
            double precision varInt
            min_lat = varInt
        end

        subroutine setMinimumLongitude(varInt)
            use geocodeState
            implicit none
            double precision varInt
            min_lon = varInt
        end

        subroutine setMaximumLatitude(varInt)
            use geocodeState
            implicit none
            double precision varInt
            max_lat = varInt
        end

        subroutine setMaximumLongitude(varInt)
            use geocodeState
            implicit none
            double precision varInt
            max_lon = varInt
        end

        subroutine setPegLatitude(varInt)
            use geocodeState
            implicit none
            double precision varInt
            peglat = varInt
        end

        subroutine setPegLongitude(varInt)
            use geocodeState
            implicit none
            double precision varInt
            peglon = varInt
        end

        subroutine setPegHeading(varInt)
            use geocodeState
            implicit none
            double precision varInt
            peghdg = varInt
        end

        subroutine setRangePixelSpacing(varInt)
            use geocodeState
            implicit none
            real*4 varInt
            drho = varInt
        end

        subroutine setLookSide(varInt)
            use geocodeState
            implicit none
            integer varInt
            ilrl = varInt
        end

        subroutine setRangeFirstSample(varInt)
            use geocodeState
            implicit none
            double precision varInt
            rho0 = varInt
        end

        subroutine setHeight(varInt)
            use geocodeState
            implicit none
            real*4 varInt
            h = varInt
        end

        subroutine setPlanetLocalRadius(varInt)
            use geocodeState
            implicit none
            double precision varInt
            ra = varInt
        end

        subroutine setVelocity(varInt)
            use geocodeState
            implicit none
            real*4 varInt
            vel = varInt
        end

        subroutine setDopplerAccessor(varInt)
            use geocodeState
            implicit none
            type(poly1dType) ::  varInt
            dopAcc = varInt
        end

        subroutine setPRF(varInt)
            use geocodeState
            implicit none
            real*4 varInt
            prf = varInt
        end

        subroutine setRadarWavelength(varInt)
            use geocodeState
            implicit none
            real*4 varInt
            wvl = varInt
        end

        subroutine setSCoordinateFirstLine(varInt)
            use geocodeState
            implicit none
            double precision varInt
            s0 = varInt
        end

        subroutine setFirstLatitude(varInt)
            use geocodeState
            implicit none
            double precision varInt
            lat_first = varInt
        end

        subroutine setFirstLongitude(varInt)
            use geocodeState
            implicit none
            double precision varInt
            lon_first = varInt
        end

        subroutine setDeltaLatitude(varInt)
            use geocodeState
            implicit none
            double precision varInt
            dlat = varInt
        end

        subroutine setDeltaLongitude(varInt)
            use geocodeState
            implicit none
            double precision varInt
            dlon = varInt
        end

        subroutine setLength(varInt)
            use geocodeState
            implicit none
            integer varInt
            length = varInt
        end

        subroutine setWidth(varInt)
            use geocodeState
            implicit none
            integer varInt
            width = varInt
        end

        subroutine setNumberRangeLooks(varInt)
            use geocodeState
            implicit none
            integer varInt
            nrnglooks = varInt
        end

        subroutine setNumberAzimuthLooks(varInt)
            use geocodeState
            implicit none
            integer varInt
            nazlooks = varInt
        end

        subroutine setNumberPointsPerDemPost(varInt)
            use geocodeState
            implicit none
            integer varInt
            ipts = varInt
        end

        subroutine setISMocomp(varInt)
            use geocodeState
            implicit none
            integer varInt
            is_mocomp = varInt
        end

        subroutine setDemWidth(varInt)
            use geocodeState
            implicit none
            integer varInt
            demwidth = varInt
        end

        subroutine setDemLength(varInt)
            use geocodeState
            implicit none
            integer varInt
            demlength = varInt
        end

        subroutine setReferenceOrbit(array1d,dim1)
            use geocodeState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                s_mocomp(i) = array1d(i)
            enddo
        end

