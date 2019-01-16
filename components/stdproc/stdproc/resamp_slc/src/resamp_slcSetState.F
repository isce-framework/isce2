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




        subroutine setInputWidth(varInt)
            use resamp_slcState
            implicit none
            integer varInt
            inwidth = varInt
        end subroutine setInputWidth

        subroutine setInputLines(varInt)
            use resamp_slcState
            implicit none
            integer varInt
            inlength = varInt
        end subroutine setInputLines

        subroutine setOutputLines(varInt)
            use resamp_slcState
            implicit none
            integer varInt
            outlength = varInt
        end subroutine setOutputLines

        subroutine setOutputWidth(varInt)
            use resamp_slcState
            implicit none
            integer varInt
            outwidth = varInt
        end subroutine setOutputWidth

        subroutine setRadarWavelength(varInt)
            use resamp_slcState
            implicit none
            real*8 varInt
            WVL = varInt
        end subroutine setRadarWavelength

        subroutine setReferenceWavelength(varInt)
            use resamp_slcState
            implicit none
            real*8 varInt
            REFWVL = varInt
        end subroutine setReferenceWavelength

        subroutine setSlantRangePixelSpacing(varInt)
            use resamp_slcState
            implicit none
            real*8 varInt
            SLR = varInt
        end subroutine setSlantRangePixelSpacing

        subroutine setReferenceSlantRangePixelSpacing(varInt)
            use resamp_slcState
            implicit none
            real*8 varInt
            REFSLR = varInt
        end subroutine setReferenceSlantRangePixelSpacing

        subroutine setStartingRange(varInt)
            use resamp_slcState
            implicit none
            real*8 varInt
            R0 = varInt
        end subroutine setStartingRange

        subroutine setReferenceStartingRange(varInt)
            use resamp_slcState
            implicit none
            real*8 varInt
            REFR0 = varInt
        end subroutine setReferenceStartingRange

        subroutine setRangeCarrier(varPoly)
            use resamp_slcState
            implicit none
            type(poly2dType) :: varPoly
            rgCarrier = varPoly
        end subroutine setRangeCarrier

        subroutine setAzimuthCarrier(varPoly)
            use resamp_slcState
            implicit none
            type(poly2dType) :: varPoly
            azCarrier = varPoly
        end subroutine setAzimuthCarrier

        subroutine setAzimuthOffsetsPoly(varPoly)
            use resamp_slcState
            implicit none
            type(poly2dType) :: varPoly
            azOffsetsPoly = varPoly
        end subroutine setAzimuthOffsetsPoly

        subroutine setRangeOffsetsPoly(varPoly)
            use resamp_slcState
            implicit none
            type(poly2dType) :: varPoly
            rgOffsetsPoly = varPoly
        end subroutine setRangeOffsetsPoly

        subroutine setDopplerPoly(varPoly)
            use resamp_slcState
            implicit none
            type(poly2dType) :: varPoly
            dopplerPoly = varPoly
        end subroutine setDopplerPoly

        subroutine setIsComplex(varInt)
            use resamp_slcState
            implicit none
            integer :: varInt
            iscomplex = varInt
        end subroutine setIsComplex

        subroutine setMethod(varInt)
            use resamp_slcState
            implicit none
            integer :: varInt
            method = varInt
        end subroutine setMethod

        subroutine setFlatten(varInt)
            use resamp_slcState
            implicit none
            integer :: varInt
            flatten = varInt
        end subroutine setFlatten
