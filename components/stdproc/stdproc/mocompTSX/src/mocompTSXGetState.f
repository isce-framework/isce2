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





        subroutine getMocompIndex(array1d,dim1)
            use mocompTSXState
            use arraymodule
            implicit none
            integer dim1,i
            double precision, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = i_mocomp(i)
            enddo
            deallocate(i_mocomp)
        end subroutine getMocompIndex

        subroutine getMocompPositionSize(var)
            use mocompTSXState
            implicit none
            integer var
            var = mocompPositionSize
        end subroutine getMocompPositionSize

        subroutine getMocompPosition(array2d,dim1,dim2)
            use arraymodule
            use mocompTSXState
            implicit none
            integer dim1,dim2,i,j
            double precision, dimension(dim2,dim1):: array2d
                do j = 1, dim2
                    array2d(j,1) = t_mocomp(j)
                    array2d(j,2) = s_mocomp(j)
                enddo
                deallocate(t_mocomp)
                deallocate(s_mocomp)
        end subroutine getMocompPosition

        subroutine getStartingRange(vardbl)
            use arraymodule
            use mocompTSXState
            implicit none
            double precision vardbl
            vardbl = adjustr0
        end subroutine getStartingRange

        subroutine getSlcSensingStart(varDbl)
            use mocompTSXState
            implicit none
            double precision varDbl
            varDbl = slcSensingStart
        end subroutine

        subroutine getMocompRange(varDbl)
            use mocompTSXState
            implicit none
            double precision varDbl
            varDbl = rho_mocomp
        end subroutine
