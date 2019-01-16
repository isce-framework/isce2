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





        subroutine allocate_xyz(dim1,dim2)
            use orbit2schState
            implicit none
            integer dim1,dim2
            dim1_xyz = dim2
            dim2_xyz = dim1
            allocate(xyz(dim2,dim1)) 
        end

        subroutine deallocate_xyz()
            use orbit2schState
            deallocate(xyz) 
        end

        subroutine allocate_vxyz(dim1,dim2)
            use orbit2schState
            implicit none
            integer dim1,dim2
            dim1_vxyz = dim2
            dim2_vxyz = dim1
            allocate(vxyz(dim2,dim1)) 
        end

        subroutine deallocate_vxyz()
            use orbit2schState
            deallocate(vxyz) 
        end

        subroutine allocate_sch(dim1,dim2)
            use orbit2schState
            implicit none
            integer dim1,dim2
            dim1_sch = dim2
            dim2_sch = dim1
            allocate(sch(dim2,dim1)) 
        end

        subroutine deallocate_sch()
            use orbit2schState
            deallocate(sch) 
        end

        subroutine allocate_vsch(dim1,dim2)
            use orbit2schState
            implicit none
            integer dim1,dim2
            dim1_vsch = dim2
            dim2_vsch = dim1
            allocate(vsch(dim2,dim1)) 
        end

        subroutine deallocate_vsch()
            use orbit2schState
            deallocate(vsch) 
        end

        subroutine allocate_asch(dim1,dim2)
            use orbit2schState
            implicit none
            integer dim1,dim2
            dim1_asch = dim2
            dim2_asch = dim1
            allocate(asch(dim2,dim1)) 
        end

        subroutine deallocate_asch()
            use orbit2schState
            deallocate(asch) 
        end

