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





        subroutine allocate_sch1(dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2
            dim1_sch1 = dim2
            dim2_sch1 = dim1
            allocate(sch1(dim2,dim1))
        end

        subroutine deallocate_sch1()
            use mocompbaselineState
            deallocate(sch1)
        end

        subroutine allocate_sch2(dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2
            dim1_sch2 = dim2
            dim2_sch2 = dim1
            allocate(sch2(dim2,dim1))
        end

        subroutine deallocate_sch2()
            use mocompbaselineState
            deallocate(sch2)
        end

        subroutine allocate_s1(dim1)
            use mocompbaselineState
            implicit none
            integer dim1
            dim1_s1 = dim1
            allocate(s1(dim1))
        end

        subroutine deallocate_s1()
            use mocompbaselineState
            deallocate(s1)
        end

        subroutine allocate_is1(dim1)
            use mocompbaselineState
            implicit none
            integer dim1
            dim1_is1 = dim1
            allocate(is1(dim1))
        end

        subroutine deallocate_is1()
            use mocompbaselineState
            deallocate(is1)
        end

        subroutine allocate_s2(dim1)
            use mocompbaselineState
            implicit none
            integer dim1
            dim1_s2 = dim1
            allocate(s2(dim1))
        end

        subroutine deallocate_s2()
            use mocompbaselineState
            deallocate(s2)
        end

        subroutine allocate_is2(dim1)
            use mocompbaselineState
            implicit none
            integer dim1
            dim1_is2 = dim1
            allocate(is2(dim1))
        end

        subroutine deallocate_is2()
            use mocompbaselineState
            deallocate(is2)
        end

        subroutine allocate_baselineArray(dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2
            dim1_baselineArray = dim2
            dim2_baselineArray = dim1
            allocate(baselineArray(dim2,dim1))
        end

        subroutine deallocate_baselineArray()
            use mocompbaselineState
            deallocate(baselineArray)
        end
        subroutine allocate_midPointArray(dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2
            dim1_midPointArray = dim2
            dim2_midPointArray = dim1
            allocate(midPointArray(dim2,dim1))
        end

        subroutine deallocate_midPointArray()
            use mocompbaselineState
            deallocate(midPointArray)
        end

        subroutine allocate_midPointArray1(dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2
            dim1_midPointArray1 = dim2
            dim2_midPointArray1 = dim1
            allocate(midPointArray1(dim2,dim1))
        end

        subroutine deallocate_midPointArray1()
            use mocompbaselineState
            deallocate(midPointArray1)
        end

        subroutine allocate_midPointArray2(dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2
            dim1_midPointArray2 = dim2
            dim2_midPointArray2 = dim1
            allocate(midPointArray2(dim2,dim1))
        end

        subroutine deallocate_midPointArray2()
            use mocompbaselineState
            deallocate(midPointArray2)
        end

        subroutine allocate_baselineArray1(dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2
            dim1_baselineArray1 = dim2
            dim2_baselineArray1 = dim1
            allocate(baselineArray1(dim2,dim1))
        end

        subroutine deallocate_baselineArray1()
            use mocompbaselineState
            deallocate(baselineArray1)
        end

        subroutine allocate_baselineArray2(dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2
            dim1_baselineArray2 = dim2
            dim2_baselineArray2 = dim1
            allocate(baselineArray2(dim2,dim1))
        end

        subroutine deallocate_baselineArray2()
            use mocompbaselineState
            deallocate(baselineArray2)
        end

        subroutine allocate_schArray(dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2
            dim1_schArray = dim2
            dim2_schArray = dim1
            allocate(schArray(dim2,dim1))
        end

        subroutine deallocate_schArray()
            use mocompbaselineState
            deallocate(schArray)
        end

        subroutine allocate_scArray(dim1,dim2)
            use mocompbaselineState
            implicit none
            integer dim1,dim2
            dim1_scArray = dim2
            dim2_scArray = dim1
            allocate(scArray(dim2,dim1))
        end

        subroutine deallocate_scArray()
            use mocompbaselineState
            deallocate(scArray)
        end
