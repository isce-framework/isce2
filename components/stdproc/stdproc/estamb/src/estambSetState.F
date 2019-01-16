c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
c 
c Licensed under the Apache License, Version 2.0 (the "License");
c you may not use this file except in compliance with the License.
c You may obtain a copy of the License at
c 
c http://www.apache.org/licenses/LICENSE-2.0
c 
c Unless required by applicable law or agreed to in writing, software
c distributed under the License is distributed on an "AS IS" BASIS,
c WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
c See the License for the specific language governing permissions and
c limitations under the License.
c 
c United States Government Sponsorship acknowledged. This software is subject to
c U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
c (No [Export] License Required except when exporting to an embargoed country,
c end user, or in support of a prohibited end use). By downloading this software,
c the user agrees to comply with all applicable U.S. export laws and regulations.
c The user has the responsibility to obtain export licenses, or other export
c authority as may be required before exporting this software to any 'EAR99'
c embargoed foreign country or citizen of those countries.
c
c Author: Giangi Sacco
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




        subroutine setStdWriter(varInt)
            use estambStateSoi
            implicit none
            integer*8 varInt
            ptStdWriter = varInt
        end

        subroutine setNumberGoodBytes(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            ngood = varInt
        end

        subroutine setNumberBytesPerLine(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            nbytes = varInt
        end

        subroutine setFirstLine(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            ifirstline = varInt
        end

        subroutine setNumberValidPulses(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            na_valid = varInt
        end

        subroutine setFirstSample(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            ifirst = varInt
        end

        subroutine setNumberPatches(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            npatches = varInt
        end

        subroutine setStartRangeBin(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            isave = varInt
        end

        subroutine setNumberRangeBin(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            nlinesaz = varInt
        end

        subroutine setRangeChirpExtensionPoints(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            nextend = varInt
        end

        subroutine setAzimuthPatchSize(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            nnn = varInt
        end

        subroutine setLookSide(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            ilrl = varInt
        end

        subroutine setOverlap(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            overlap = varInt
        end

        subroutine setRanfftov(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            ranfftov = varInt
        end

        subroutine setRanfftiq(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            ranfftiq = varInt
        end

        subroutine setDebugFlag(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            iflag = varInt
        end

        subroutine setCaltoneLocation(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            caltone1 = varInt
        end

        subroutine setPlanetLocalRadius(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            rcurv = varInt
        end

        subroutine setBodyFixedVelocity(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            vel1 = varInt
        end

        subroutine setSpacecraftHeight(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            ht1 = varInt
        end

        subroutine setPRF(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            prf1 = varInt
        end

        subroutine setInPhaseValue(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            xmi1 = varInt
        end

        subroutine setQuadratureValue(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            xmq1 = varInt
        end

        subroutine setAzimuthResolution(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            azres = varInt
        end

        subroutine setRangeSamplingRate(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            fs = varInt
        end

        subroutine setChirpSlope(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            slope = varInt
        end

        subroutine setRangePulseDuration(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            pulsedur = varInt
        end

        subroutine setRadarWavelength(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            wavl = varInt
        end

        subroutine setRangeFirstSample(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            r001 = varInt
        end

        subroutine setRangeSpectralWeighting(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            rhww = varInt
        end

        subroutine setSpectralShiftFraction(varInt)
            use estambStateSoi
            implicit none
            double precision varInt
            pctbw = varInt
        end

        subroutine setIMRC1(varInt)
            use estambStateSoi
            implicit none
            integer*8 varInt
            imrc1Accessor = varInt
        end

        subroutine setIMMocomp(varInt)
            use estambStateSoi
            implicit none
            integer*8 varInt
            immocompAccessor = varInt
        end

        subroutine setIMRCAS1(varInt)
            use estambStateSoi
            implicit none
            integer*8 varInt
            imrcas1Accessor = varInt
        end

        subroutine setIMRCRM1(varInt)
            use estambStateSoi
            implicit none
            integer*8 varInt
            imrcrm1Accessor = varInt
        end

        subroutine setTransDat(varInt)
            use estambStateSoi
            implicit none
            integer*8 varInt
            transAccessor = varInt
        end

        subroutine setIQFlip(varString, varInt)
            use estambStateSoi
            implicit none
            character*1 varString
            integer*4 varInt
            iqflip  = ''
            iqflip(1:varInt) = varString
        end

        subroutine setDeskewFlag(varString, varInt)
            use estambStateSoi
            implicit none
            character*1 varString
            integer*4 varInt
            deskew  = ''
            deskew(1:varInt) = varString
        end

        subroutine setSecondaryRangeMigrationFlag(varString, varInt)
            use estambStateSoi
            implicit none
            character*1 varString
            integer*4 varInt
            srm  = ''
            srm(1:varInt) = varString
        end

        subroutine setPosition(array2dT,dim1,dim2)
            use estambStateSoi
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    sch(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setVelocity(array2dT,dim1,dim2)
            use estambStateSoi
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    vsch(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setTime(array1d,dim1)
            use estambStateSoi
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                time(i) = array1d(i)
            enddo
        end

        subroutine setDopplerCentroidCoefficients(array1d,dim1)
            use estambStateSoi
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                dopplerCoefficients(i) = array1d(i)
            enddo
        end

        subroutine setPegPoint(lat,lon,hdg)
            use estambStateSoi
            implicit none
            double precision :: lat,lon,hdg
            peg%r_lat = lat
            peg%r_lon = lon
            peg%r_hdg = hdg
        end subroutine setPegPoint

        subroutine setEllipsoid(a,e2)
            use estambStateSoi
            implicit none
            double precision :: a, e2
            elp%r_a = a
            elp%r_e2 = e2
        end subroutine setEllipsoid

        subroutine setPlanet(spin,gm)
            use estambStateSoi
            implicit none
            double precision :: spin,gm
            pln%r_spindot = spin
            pln%r_gm = gm
        end subroutine setPlanet

        subroutine setSlcWidth(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            nlinesazout = varInt
        end

        subroutine setStartingRange(varDbl)
            use estambStateSoi
            implicit none
            double precision varDbl
            r01 = varDbl
        end

        !KK,ML 2013-07-15
        subroutine setShift(varDbl)
            use estambStateSoi
            implicit none
            double precision varDbl
            shift = varDbl
        end
        !KK,ML

        subroutine setMinAmb(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            minamb = varInt
        end

        subroutine setMaxAmb(varInt)
            use estambStateSoi
            implicit none
            integer varInt
            maxamb = varInt
        end

