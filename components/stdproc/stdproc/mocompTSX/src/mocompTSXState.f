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





        module mocompTSXState
            use orbitModule
            use geometryModule

            integer stdWriter
            integer nr
            integer naz
            double precision, allocatable, dimension(:) ::  dopplerCentroidCoefficients
            integer dim1_dopplerCentroidCoefficients
            double precision, allocatable, dimension(:) ::  time
            integer dim1_time
            double precision, allocatable, dimension(:,:) ::  sch
            integer dim1_sch, dim2_sch
            double precision rcurv
            double precision vel
            double precision ht
            double precision prf
            double precision fs
            double precision wvl
            double precision r0
            integer dim1_i_mocomp
            integer mocompPositionSize
            integer ilrl
            double precision adjustr0

            type planet_type
              double precision :: r_spindot !< Planet spin rate
              double precision :: r_gm !< Planet GM
            end type planet_type
            type(orbitType) :: orbit  !Input short orbit
            type(orbitType) :: mocompOrbit !Output short orbit
            double precision :: sensingStart !UTC time corresponding to first raw line
            double precision :: slcSensingStart !UTC time corresponding to first slc line
            double precision :: rho_mocomp   !Range used for motion compensation
            type(pegtransType) :: ptm  !For WGS84 to SCH
            type(pegType) :: peg
            type(ellipsoidType) :: elp
            type(planet_type) :: pln

        end module
