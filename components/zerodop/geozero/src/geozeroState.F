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





        module geozeroState
            use poly1dModule
            use orbitModule

            !!Planet description
            double precision majorSemiAxis
            double precision eccentricitySquared

            !!Bounding box input
            double precision min_lat
            double precision min_lon
            double precision max_lat
            double precision max_lon

            !!Range spacing
            double precision drho
            double precision rho0

            !!Radar Geometry
            type(poly1dType) ::  dopAcc
            type(orbitType) :: orbit

            !!Acquisition properties
            double precision wvl
            double precision t0
            double precision prf
            integer length
            integer width
            integer nrnglooks
            integer nazlooks
            integer ilrl

            !!DEM information
            double precision lat_first
            double precision lon_first
            double precision dlat
            double precision dlon
            integer demwidth
            integer demlength
            
            !!Output
            integer geowidth
            integer geolength
            double precision geomin_lat
            double precision geomax_lat
            double precision geomin_lon
            double precision geomax_lon
        end module geozeroState 
