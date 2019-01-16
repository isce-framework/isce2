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





        subroutine allocate_dopplerCoefficients(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_dopplerCoefficients = dim1
            allocate(dopplerCoefficients(dim1)) 
        end

        subroutine deallocate_dopplerCoefficients()
            use resamp_ampsState
            deallocate(dopplerCoefficients) 
        end

        subroutine allocate_r_ranpos(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_r_ranpos = dim1
            allocate(r_ranposV(dim1)) 
        end

        subroutine deallocate_r_ranpos()
            use resamp_ampsState
            deallocate(r_ranposV) 
        end

        subroutine allocate_r_ranoff(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_r_ranoff = dim1
            allocate(r_ranoffV(dim1)) 
        end

        subroutine deallocate_r_ranoff()
            use resamp_ampsState
            deallocate(r_ranoffV) 
        end

        subroutine allocate_r_azpos(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_r_azpos = dim1
            allocate(r_azposV(dim1)) 
        end

        subroutine deallocate_r_azpos()
            use resamp_ampsState
            deallocate(r_azposV) 
        end

        subroutine allocate_r_azoff(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_r_azoff = dim1
            allocate(r_azoffV(dim1)) 
        end

        subroutine deallocate_r_azoff()
            use resamp_ampsState
            deallocate(r_azoffV) 
        end

        subroutine allocate_r_sig(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_r_sig = dim1
            allocate(r_sigV(dim1)) 
        end

        subroutine deallocate_r_sig()
            use resamp_ampsState
            deallocate(r_sigV) 
        end

        subroutine allocate_r_ranpos2(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_r_ranpos2 = dim1
            allocate(r_ranpos2V(dim1)) 
        end

        subroutine deallocate_r_ranpos2()
            use resamp_ampsState
            deallocate(r_ranpos2V) 
        end

        subroutine allocate_r_ranoff2(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_r_ranoff2 = dim1
            allocate(r_ranoff2V(dim1)) 
        end

        subroutine deallocate_r_ranoff2()
            use resamp_ampsState
            deallocate(r_ranoff2V) 
        end

        subroutine allocate_r_azpos2(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_r_azpos2 = dim1
            allocate(r_azpos2V(dim1)) 
        end

        subroutine deallocate_r_azpos2()
            use resamp_ampsState
            deallocate(r_azpos2V) 
        end

        subroutine allocate_r_azoff2(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_r_azoff2 = dim1
            allocate(r_azoff2V(dim1)) 
        end

        subroutine deallocate_r_azoff2()
            use resamp_ampsState
            deallocate(r_azoff2V) 
        end

        subroutine allocate_r_sig2(dim1)
            use resamp_ampsState
            implicit none
            integer dim1
            dim1_r_sig2 = dim1
            allocate(r_sig2V(dim1)) 
        end

        subroutine deallocate_r_sig2()
            use resamp_ampsState
            deallocate(r_sig2V) 
        end

