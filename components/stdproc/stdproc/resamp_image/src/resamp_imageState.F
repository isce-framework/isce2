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





        module resamp_imageState
            integer*8 ptStdWriter
            integer i_ma
            integer npl
            integer nl
            integer looks
            integer istoff
            real*4 WVL
            real*4 SLR
            double precision, allocatable, dimension(:) ::  dopplerCoefficients
            integer dim1_dopplerCoefficients
            double precision, allocatable, dimension(:) ::  r_ranposV
            integer dim1_r_ranpos
            double precision, allocatable, dimension(:) ::  r_ranoffV
            integer dim1_r_ranoff
            double precision, allocatable, dimension(:) ::  r_azposV
            integer dim1_r_azpos
            double precision, allocatable, dimension(:) ::  r_azoffV
            integer dim1_r_azoff
            double precision, allocatable, dimension(:) ::  r_sigV
            integer dim1_r_sig
            double precision, allocatable, dimension(:) ::  r_ranpos2V
            integer dim1_r_ranpos2
            double precision, allocatable, dimension(:) ::  r_ranoff2V
            integer dim1_r_ranoff2
            double precision, allocatable, dimension(:) ::  r_azpos2V
            integer dim1_r_azpos2
            double precision, allocatable, dimension(:) ::  r_azoff2V
            integer dim1_r_azoff2
            double precision, allocatable, dimension(:) ::  r_sig2V
            integer dim1_r_sig2
        end module 
