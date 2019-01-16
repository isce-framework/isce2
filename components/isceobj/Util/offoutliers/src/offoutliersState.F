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





        module offoutliersState
            integer*8 ptStdWriter
            integer, allocatable, dimension(:) ::  indexArray
            integer dim1_indexArray
            integer indexArraySize
            real*4 cdn
            real*4 cac
            integer MP
            double precision, allocatable, dimension(:) ::  xd
            integer dim1_xd
            double precision, allocatable, dimension(:) ::  acshift
            integer dim1_acshift
            double precision, allocatable, dimension(:) ::  yd
            integer dim1_yd
            double precision, allocatable, dimension(:) ::  dnshift
            integer dim1_dnshift
            real*4 distance
            double precision, allocatable, dimension(:) ::  sig
            integer dim1_sig
            double precision, allocatable, dimension(:) ::  s
            integer dim1_s
        end module 
