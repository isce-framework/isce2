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





        subroutine allocate_locationAcross(dim1)
            use estimateoffsetsState
            implicit none
            integer dim1
            dim1_locationAcross = dim1
            allocate(locationAcross(dim1)) 
        end

        subroutine deallocate_locationAcross()
            use estimateoffsetsState
            deallocate(locationAcross) 
        end

        subroutine allocate_locationAcrossOffset(dim1)
            use estimateoffsetsState
            implicit none
            integer dim1
            dim1_locationAcrossOffset = dim1
            allocate(locationAcrossOffset(dim1)) 
        end

        subroutine deallocate_locationAcrossOffset()
            use estimateoffsetsState
            deallocate(locationAcrossOffset) 
        end

        subroutine allocate_locationDown(dim1)
            use estimateoffsetsState
            implicit none
            integer dim1
            dim1_locationDown = dim1
            allocate(locationDown(dim1)) 
        end

        subroutine deallocate_locationDown()
            use estimateoffsetsState
            deallocate(locationDown) 
        end

        subroutine allocate_locationDownOffset(dim1)
            use estimateoffsetsState
            implicit none
            integer dim1
            dim1_locationDownOffset = dim1
            allocate(locationDownOffset(dim1)) 
        end

        subroutine deallocate_locationDownOffset()
            use estimateoffsetsState
            deallocate(locationDownOffset) 
        end

        subroutine allocate_snrRet(dim1)
            use estimateoffsetsState
            implicit none
            integer dim1
            dim1_snrRet = dim1
            allocate(snrRet(dim1)) 
        end

        subroutine deallocate_snrRet()
            use estimateoffsetsState
            deallocate(snrRet) 
        end

