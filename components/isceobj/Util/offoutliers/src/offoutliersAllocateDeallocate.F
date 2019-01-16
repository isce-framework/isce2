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





        subroutine allocate_indexArray(dim1)
            use offoutliersState
            implicit none
            integer dim1
            dim1_indexArray = dim1
            allocate(indexArray(dim1)) 
            indexArray=0
        end

        subroutine deallocate_indexArray()
            use offoutliersState
            deallocate(indexArray) 
        end

        subroutine allocate_xd(dim1)
            use offoutliersState
            implicit none
            integer dim1
            dim1_xd = dim1
            allocate(xd(dim1))
            xd=0
        end

        subroutine deallocate_xd()
            use offoutliersState
            deallocate(xd) 
        end

        subroutine allocate_acshift(dim1)
            use offoutliersState
            implicit none
            integer dim1
            dim1_acshift = dim1
            allocate(acshift(dim1))
            acshift=0 
        end

        subroutine deallocate_acshift()
            use offoutliersState
            deallocate(acshift) 
        end

        subroutine allocate_yd(dim1)
            use offoutliersState
            implicit none
            integer dim1
            dim1_yd = dim1
            allocate(yd(dim1)) 
            yd=0
        end

        subroutine deallocate_yd()
            use offoutliersState
            deallocate(yd) 
        end

        subroutine allocate_dnshift(dim1)
            use offoutliersState
            implicit none
            integer dim1
            dim1_dnshift = dim1
            allocate(dnshift(dim1))
            dnshift=0
        end

        subroutine deallocate_dnshift()
            use offoutliersState
            deallocate(dnshift) 
        end

        subroutine allocate_sig(dim1)
            use offoutliersState
            implicit none
            integer dim1
            dim1_sig = dim1
            allocate(sig(dim1)) 
            sig=0
        end

        subroutine deallocate_sig()
            use offoutliersState
            deallocate(sig) 
        end

        subroutine allocate_s(dim1)
            use offoutliersState
            implicit none
            integer dim1
            dim1_s = dim1
            allocate(s(dim1)) 
            s = 0
        end

        subroutine deallocate_s()
            use offoutliersState
            deallocate(s) 
        end

