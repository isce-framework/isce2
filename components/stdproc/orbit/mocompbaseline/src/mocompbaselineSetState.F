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
            use mocompbaselineState
            implicit none
            integer*8 varInt
            ptStdWriter = varInt
        end

        subroutine setSchPosition1(array2dT,dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    sch1(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setSchPosition2(array2dT,dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    sch2(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setMocompPosition1(array1d,dim1)
            use mocompbaselineState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                s1(i) = array1d(i)
            enddo
        end

        subroutine setMocompPositionIndex1(array1d,dim1)
            use mocompbaselineState
            implicit none
            integer dim1,i
            integer, dimension(dim1):: array1d
            do i = 1, dim1
                is1(i) = array1d(i)
            enddo
        end

        subroutine setMocompPosition2(array1d,dim1)
            use mocompbaselineState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                s2(i) = array1d(i)
            enddo
        end

        subroutine setMocompPositionIndex2(array1d,dim1)
            use mocompbaselineState
            implicit none
            integer dim1,i
            integer, dimension(dim1):: array1d
            do i = 1, dim1
                is2(i) = array1d(i)
            enddo
        end

        subroutine setEllipsoidMajorSemiAxis(varInt)
            use mocompbaselineState
            implicit none
            double precision varInt
            major = varInt
        end

        subroutine setEllipsoidEccentricitySquared(varInt)
            use mocompbaselineState
            implicit none
            double precision varInt
            eccentricitySquared = varInt
        end

        subroutine setPlanetLocalRadius(var)
            use mocompbaselineState
            implicit none
            double precision var
            rcurv = var
        end

        subroutine setPegLatitude(varInt)
            use mocompbaselineState
            implicit none
            double precision varInt
            peglat = varInt
        end

        subroutine setHeight(varInt)
            use mocompbaselineState
            implicit none
            double precision varInt
            height = varInt
        end
        subroutine setPegLongitude(varInt)
            use mocompbaselineState
            implicit none
            double precision varInt
            peglon = varInt
        end

        subroutine setPegHeading(varInt)
            use mocompbaselineState
            implicit none
            double precision varInt
            peghdg = varInt
        end

