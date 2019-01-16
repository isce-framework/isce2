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





        subroutine getMocompPosition(array2d,dim1,dim2)
            use arraymodule
            use formslcStateSoi
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2d
                do j = 1, dim2
                    array2d(j,1) = t_ mocomp(j)
                    array2d(j,2) = s_ mocomp(j)
                enddo
                deallocate(t_mocomp)
                deallocate(s_mocomp)
        end subroutine

        subroutine getMocompIndex(array1d,dim1)
            use arraymodule
            use formslcStateSoi
            implicit none
            integer dim1,i
            integer, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = i_mocomp(i)
            enddo
            deallocate(i_mocomp)
        end subroutine

        subroutine getMocompPositionSize(varInt)
            use formslcStateSoi
            implicit none
            integer varInt
            varInt = mocompPositionSize
        end subroutine
        
        subroutine getStartingRange(varDbl)
            use formslcStateSoi
            implicit none
            double precision varDbl
            varDbl = slcr01
        end subroutine

        subroutine getSlcSensingStart(varDbl)
            use formslcStateSoi
            implicit none
            double precision varDbl
            varDbl = slcSensingStart
        end subroutine

        subroutine getMocompRange(varDbl)
            use formSlcStateSoi
            implicit none
            double precision varDbl
            varDbl = rho_mocomp
        end subroutine
