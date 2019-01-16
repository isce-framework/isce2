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





        module topozeroState
            use orbitModule
            integer numiter
            integer idemwidth
            integer idemlength
            type(orbitType) :: orbit
            double precision firstlat
            double precision firstlon
            double precision deltalat
            double precision deltalon
            double precision major
            double precision eccentricitySquared
            integer length
            integer width
            double precision rspace
            double precision r0
            integer Nrnglooks
            integer Nazlooks
            double precision peghdg
            double precision prf
            double precision t0
            double precision wvl
            integer*8 latAccessor
            integer*8 lonAccessor
            integer*8 heightAccessor
            integer*8 losAccessor
            integer*8 incAccessor
            integer*8 maskAccessor
            double precision min_lat
            double precision min_lon
            double precision max_lat
            double precision max_lon
            double precision thresh
            integer ilrl
            integer extraiter
            integer method
            integer orbitMethod

            !!!For cropping DEM
            !!!Min global height
            !!!Max global height
            !!!Margin around bbox in degrees
            double precision MIN_H, MAX_H, MARGIN
            parameter(MIN_H=-500.0d0, MAX_H=9000.0d0, MARGIN=0.15d0)

            integer HERMITE_METHOD, LEGENDRE_METHOD, SCH_METHOD
            parameter(HERMITE_METHOD=0,SCH_METHOD=1,LEGENDRE_METHOD=2)
        end module topozeroState 
