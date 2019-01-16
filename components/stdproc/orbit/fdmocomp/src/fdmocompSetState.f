!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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





        subroutine setStartingRange(var)
            use fdmocompState
            implicit none
            double precision var
            r001 = var
        end

        subroutine setPRF(var)
            use fdmocompState
            implicit none
            double precision var
            prf = var
        end

        subroutine setRadarWavelength(var)
            use fdmocompState
            implicit none
            double precision var
            wavl = var
        end

        subroutine setWidth(var)
            use fdmocompState
            implicit none
            integer var
            nlinesaz = var
        end

        subroutine setHeigth(var)
            use fdmocompState
            implicit none
            integer var
            nlines = var
        end

        subroutine setPlatformHeigth(var)
            use fdmocompState
            implicit none
            integer var
            ht1 = var
        end

        subroutine setLookSide(var)
            use fdmocompState
            implicit none
            integer var
            ilrl = var
        end

        subroutine setRangeSamplingRate(var)
            use fdmocompState
            implicit none
            double precision var
            fs = var
        end

        subroutine setRadiusOfCurvature(var)
            use fdmocompState
            implicit none
            double precision var
            rcurv = var
        end

        subroutine setDopplerCoefficients(array1d,dim1)
            use fdmocompState
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                fdArray(i) = array1d(i)
            enddo
        end

        subroutine setSchVelocity(array2dT,dim1,dim2)
            use fdmocompState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    vsch(i,j) = array2dT(i,j)
                enddo
            enddo
        end

