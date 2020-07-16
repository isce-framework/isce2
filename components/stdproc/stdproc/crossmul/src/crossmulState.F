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





        module crossmulState
        use, intrinsic:: iso_c_binding

            type, bind(C) :: crossmulType
                integer(C_INT) :: na    !c Number across in range
                integer(C_INT) :: nd    !c Number down in azimuth
                real(C_DOUBLE) ::  scale  !c Scaling the numbers numerically
                integer (C_INT):: looksac          !c Looks across
                integer (C_INT):: looksdn          !c Looks down
                integer (C_INT):: blocksize        !c Number of azimuth lines in a block
                real(C_DOUBLE) :: wvl1  !c Reference wavelength
                real(C_DOUBLE) :: wvl2  !c Secondary wavelength
                real(C_DOUBLE) :: drg1  !c Reference slant range spacing
                real(C_DOUBLE) :: drg2  !c Secondary slant range spacing
                integer(C_INT) :: flatten  !c Flatten flag
                real(C_DOUBLE) :: wgt  !c Range filter weight
            end type crossmulType
        end module crossmulState
