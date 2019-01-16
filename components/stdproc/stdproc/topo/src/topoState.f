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





        module topoState
            use geometryModule
            use orbitModule
            integer numiter
            integer idemwidth
            integer idemlength
            double precision, allocatable, dimension(:) ::  s_mocomp
            integer dim1_s_mocompArray
            double precision firstlat
            double precision firstlon
            double precision deltalat
            double precision deltalon
            integer is_mocomp
            double precision major
            double precision eccentricitySquared
            integer length
            integer width
            double precision rspace
            double precision r0
            double precision height
            double precision rcurv
            real*4 vel
            integer Nrnglooks
            integer Nazlooks
            double precision peglat
            double precision peglon
            double precision peghdg
            double precision prf
            double precision wvl
            integer*8 latAccessor
            integer*8 lonAccessor
            integer*8 heightRAccessor
            integer*8 heightSchAccessor
            integer*8 losAccessor
            integer*8 incAccessor
            double precision azspace
            double precision re
            double precision s0
            double precision send
            double precision min_lat
            double precision min_lon
            double precision max_lat
            double precision max_lon
            double precision, allocatable, dimension(:) ::  squintshift
            integer dim1_squintshift
            integer ilrl
            integer method
            type(orbitType):: orbit
            double precision sensingStart
        end module topoState 
