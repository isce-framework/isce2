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





        subroutine setReferenceOrbit(array1d,dim1)
            use correctState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                s_mocomp(i) = array1d(i)
            enddo
        end

        subroutine setMocompBaseline(array2dT,dim1,dim2)
            use correctState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    mocbase(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setISMocomp(var)
            use correctState
            implicit none
            integer var
            is_mocomp = var
        end

        subroutine setEllipsoidMajorSemiAxis(var)
            use correctState
            implicit none
            double precision var
            major = var
        end

        subroutine setEllipsoidEccentricitySquared(var)
            use correctState
            implicit none
            double precision var
            eccentricitySquared = var
        end

        subroutine setLength(var)
            use correctState
            implicit none
            integer var
            length = var
        end

        subroutine setWidth(var)
            use correctState
            implicit none
            integer var
            width = var
        end

        subroutine setRangePixelSpacing(var)
            use correctState
            implicit none
            double precision var
            rspace = var
        end

        subroutine setLookSide(var)
            use correctState
            implicit none
            integer var
            ilrl = var
        end

        subroutine setRangeFirstSample(var)
            use correctState
            implicit none
            double precision var
            r0 = var
        end

        subroutine setSpacecraftHeight(var)
            use correctState
            implicit none
            double precision var
            height = var
        end

        subroutine setPlanetLocalRadius(var)
            use correctState
            implicit none
            double precision var
            rcurv = var
        end

        subroutine setBodyFixedVelocity(var)
            use correctState
            implicit none
            real*4 var
            vel = var
        end

        subroutine setNumberRangeLooks(var)
            use correctState
            implicit none
            integer var
            Nrnglooks = var
        end

        subroutine setNumberAzimuthLooks(var)
            use correctState
            implicit none
            integer var
            Nazlooks = var
        end

        subroutine setPegLatitude(var)
            use correctState
            implicit none
            double precision var
            peglat = var
        end

        subroutine setPegLongitude(var)
            use correctState
            implicit none
            double precision var
            peglon = var
        end

        subroutine setPegHeading(var)
            use correctState
            implicit none
            double precision var
            peghdg = var
        end

        subroutine setPRF(var)
            use correctState
            implicit none
            double precision var
            prf = var
        end

        subroutine setRadarWavelength(var)
            use correctState
            implicit none
            double precision var
            wvl = var
        end

        subroutine setMidpoint(array2dT,dim1,dim2)
            use correctState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    midpoint(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setSch1(array2dT,dim1,dim2)
            use correctState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    s1sch(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setSch2(array2dT,dim1,dim2)
            use correctState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    s2sch(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setSc(array2dT,dim1,dim2)
            use correctState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    smsch(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setDopCoeff(var)
            use correctState
            implicit none
            integer*8 var
            dopAcc = var
        end subroutine setDopCoeff

