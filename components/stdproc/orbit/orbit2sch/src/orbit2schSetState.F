c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
c 
c Licensed under the Apache License, Version 2.0 (the "License");
c you may not use this file except in compliance with the License.
c You may obtain a copy of the License at
c 
c http://www.apache.org/licenses/LICENSE-2.0
c 
c Unless required by applicable law or agreed to in writing, software
c distributed under the License is distributed on an "AS IS" BASIS,
c WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
c See the License for the specific language governing permissions and
c limitations under the License.
c 
c United States Government Sponsorship acknowledged. This software is subject to
c U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
c (No [Export] License Required except when exporting to an embargoed country,
c end user, or in support of a prohibited end use). By downloading this software,
c the user agrees to comply with all applicable U.S. export laws and regulations.
c The user has the responsibility to obtain export licenses, or other export
c authority as may be required before exporting this software to any 'EAR99'
c embargoed foreign country or citizen of those countries.
c
c Author: Giangi Sacco
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




        subroutine setStdWriter(varInt)
            use orbit2schstate
            implicit none
            integer*8 varInt
            ptStdWriter = varInt
        end

        subroutine setOrbitPosition(array2dT,dim1,dim2)
            use orbit2schState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    xyz(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setOrbitVelocity(array2dT,dim1,dim2)
            use orbit2schState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    vxyz(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setPlanetGM(varInt)
            use orbit2schState
            implicit none
            double precision varInt
            GM = varInt
        end

        subroutine setEllipsoidMajorSemiAxis(varInt)
            use orbit2schState
            implicit none
            double precision varInt
            major = varInt
        end

        subroutine setEllipsoidEccentricitySquared(varInt)
            use orbit2schState
            implicit none
            double precision varInt
            eccentricitySquared = varInt
        end

        subroutine setComputePegInfoFlag(varInt)
            use orbit2schState
            implicit none
            integer varInt
            computePegInfoFlag = varInt
        end

        subroutine setPegLatitude(varInt)
            use orbit2schState
            implicit none
            double precision varInt
            pegLatitude = varInt
        end

        subroutine setPegLongitude(varInt)
            use orbit2schState
            implicit none
            double precision varInt
            pegLongitude = varInt
        end

        subroutine setPegHeading(varInt)
            use orbit2schState
            implicit none
            double precision varInt
            pegHeading = varInt
        end

        subroutine setAverageHeight(varInt)
            use orbit2schState
            implicit none
            double precision varInt
            have = varInt
        end

