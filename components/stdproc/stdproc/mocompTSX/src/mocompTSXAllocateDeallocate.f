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





        subroutine allocate_dopplerCentroidCoefficients(dim1)
            use mocompTSXState
            implicit none
            integer dim1
            dim1_dopplerCentroidCoefficients = dim1
            allocate(dopplerCentroidCoefficients(dim1)) 
        end

        subroutine deallocate_dopplerCentroidCoefficients()
            use mocompTSXState
            deallocate(dopplerCentroidCoefficients) 
        end

        subroutine allocate_time(dim1)
            use mocompTSXState
            implicit none
            integer dim1
            dim1_time = dim1
            allocate(time(dim1)) 
        end

        subroutine deallocate_time()
            use mocompTSXState
            deallocate(time) 
        end

        subroutine allocate_sch(dim1,dim2)
            use mocompTSXState
            implicit none
            integer dim1,dim2
            dim1_sch = dim2
            dim2_sch = dim1
            allocate(sch(dim2,dim1)) 
        end

        subroutine deallocate_sch()
            use mocompTSXState
            deallocate(sch) 
        end


