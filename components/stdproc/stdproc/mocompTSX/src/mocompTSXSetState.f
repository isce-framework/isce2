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





        subroutine setStdWriter(var)
            use mocompTSXState
            implicit none
            integer var
            stdWriter = var
        end

        subroutine setNumberRangeBins(var)
            use mocompTSXState
            implicit none
            integer var
            nr = var
        end

        subroutine setNumberAzLines(var)
            use mocompTSXState
            implicit none
            integer var
            naz = var
        end

        subroutine setDopplerCentroidCoefficients(array1d,dim1)
            use mocompTSXState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                dopplerCentroidCoefficients(i) = array1d(i)
            enddo
        end

        subroutine setTime(array1d,dim1)
            use mocompTSXState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                time(i) = array1d(i)
            enddo
        end

        subroutine setPosition(array2dT,dim1,dim2)
            use mocompTSXState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    sch(i,j) = array2dT(i,j)
                enddo
            enddo
        end


        subroutine setPlanetLocalRadius(var)
            use mocompTSXState
            implicit none
            double precision var
            rcurv = var
        end

        subroutine setBodyFixedVelocity(var)
            use mocompTSXState
            implicit none
            double precision var
            vel = var
        end

        subroutine setSpacecraftHeight(var)
            use mocompTSXState
            implicit none
            double precision var
            ht = var
        end

        subroutine setPRF(var)
            use mocompTSXState
            implicit none
            double precision var
            prf = var
        end

        subroutine setRangeSamplingRate(var)
            use mocompTSXState
            implicit none
            double precision var
            fs = var
        end

        subroutine setRadarWavelength(var)
            use mocompTSXState
            implicit none
            double precision var
            wvl = var
        end

        subroutine setRangeFisrtSample(var)
            use mocompTSXState
            implicit none
            double precision var
            r0 = var
        end

        subroutine setLookSide(var)
            use mocompTSXState
            implicit none
            integer var
            ilrl = var
        end

        subroutine setEllipsoid(a,e2)
            use mocompTSXState
            implicit none
            double precision :: a, e2
            elp%r_a = a
            elp%r_e2 = e2
        end subroutine setEllipsoid


        subroutine setPegPoint(lat,lon,hdg)
            use mocompTSXState
            implicit none
            double precision :: lat,lon,hdg
            peg%r_lat = lat
            peg%r_lon = lon
            peg%r_hdg = hdg
        end subroutine setPegPoint

        subroutine setOrbit(corb)
            use mocompTSXState
            implicit none

            type(orbitType) :: corb
            orbit = corb
        end subroutine

        subroutine setMocompOrbit(corb)
            use mocompTSXState
            implicit none

            type(orbitType) :: corb
            mocompOrbit = corb
        end subroutine

        subroutine setPlanet(spin,gm)
            use mocompTSXState
            implicit none
            double precision :: spin,gm
            pln%r_spindot = spin
            pln%r_gm = gm
        end subroutine setPlanet

        subroutine setSensingStart(varDbl)
            use mocompTSXState
            implicit none
            double precision varDbl
            sensingStart = varDbl
        end subroutine
