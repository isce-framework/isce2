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





        module mocompbaselineState
            integer*8 ptStdWriter
            double precision, allocatable, dimension(:,:) ::  sch1
            integer dim1_sch1, dim2_sch1
            double precision, allocatable, dimension(:,:) ::  sch2
            integer dim1_sch2, dim2_sch2
            double precision, allocatable, dimension(:) ::  s1
            integer dim1_s1
            integer, allocatable, dimension(:) ::  is1
            integer dim1_is1
            double precision, allocatable, dimension(:) ::  s2
            integer dim1_s2
            integer, allocatable, dimension(:) ::  is2
            integer dim1_is2
            double precision height
            double precision major
            double precision eccentricitySquared
            double precision rcurv
            double precision peglat
            double precision peglon
            double precision peghdg
            double precision, allocatable, dimension(:,:) ::  baselineArray
            integer dim1_baselineArray, dim2_baselineArray
            double precision, allocatable, dimension(:,:) ::  midPointArray
            integer dim1_midPointArray, dim2_midPointArray
            double precision, allocatable, dimension(:,:) ::  midPointArray1
            integer dim1_midPointArray1, dim2_midPointArray1
            double precision, allocatable, dimension(:,:) ::  midPointArray2
            integer dim1_midPointArray2, dim2_midPointArray2
            double precision, allocatable, dimension(:,:) ::  baselineArray1
            integer dim1_baselineArray1, dim2_baselineArray1
            double precision, allocatable, dimension(:,:) ::  baselineArray2
            integer dim1_baselineArray2, dim2_baselineArray2
            double precision, allocatable, dimension(:,:) ::  schArray
            integer dim1_schArray, dim2_schArray
            double precision, allocatable, dimension(:,:) ::  scArray
            integer dim1_scArray, dim2_scArray
        end module
