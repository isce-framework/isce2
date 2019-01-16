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




      module fortranUtils
        integer :: UNIT_STDERR = 0
        integer :: UNIT_STDOUT = 6
        integer :: UNIT_LOG = 1
        character*16 :: FILE_LOG = "isce_fortran.log"

        contains
            function getPI()
                double precision ::getPI
                 getPI = 4.d0*atan(1.d0)
            end function getPI

            function getSpeedOfLight()
                double precision:: getSpeedOfLight
                getSpeedOfLight = 299792458.0d0
            end function getSpeedOfLight

            subroutine set_stdoel_units()
                implicit none
                logical UNITOK, UNITOP
                inquire (unit=UNIT_LOG, exist=UNITOK, opened=UNITOP)
                if (UNITOK .and. .not. UNITOP) then
                   open(unit=UNIT_LOG, file=FILE_LOG, form="formatted", access="append", status="unknown")
                end if
            end subroutine

            subroutine c_to_f_string(pName,cString, cStringLen, fString, fStringLen)
                use iso_c_binding, only: c_char, c_null_char
                implicit none
                integer*4 fStringLen
                integer*4 cStringLen, i
                character*(*), intent(in) :: pName
                character*(*), intent(out) :: fString
                character(kind=c_char, len=1),dimension(cStringLen),intent(in)::  cString
                !Check to amke sure the fString is large enough to hold the cString
                if( cStringLen-1 .gt. fStringLen ) then
                    write(UNIT_STDOUT,*) "*** Error in fortranUtils::c_to_f_string ", &
                      " called from program, ", pName
                    write(UNIT_STDOUT,*) "variable fString of length, ", fStringLen, &
                      "is not large enough to hold variable cString = ", &
                      cString(1:cStringLen), " of length, ", cStringLen
                    stop
                end if

                fString  = ''
                do i = 1, cStringLen
                    if(cString(i) .eq. c_null_char) then
                        fStringLen = i-1
                        exit
                    else
                        fString(i:i) = cString(i)
                    end if
                end do
            end subroutine c_to_f_string

      end module
