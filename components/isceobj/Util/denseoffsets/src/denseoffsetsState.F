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





        module denseoffsetsState
            !File widths in range pixels
            integer len1, len2

            !If input data is complex
            integer iscpx1, iscpx2

            !Band numbers for input data
            integer band1, band2

            !File lengths in azimuth lines
            integer lines1, lines2
            integer isamp_s
            integer isamp_f
            integer iskipac
            integer iskipdn
            integer isamp_sdn
            integer isamp_fdn
            integer nlocdn
            integer ioffac
            integer ioffdn
            real*4 scalex
            real*4 scaley
            character*1 talk

            integer NPTSW   !Window width
            integer NPTSH   !Window Height
            integer NOFFW   !Search width
            integer NOFFH   !Search height
            integer NDISP  !Search window around maximum
            integer NOVS   !Oversampling factor 
            logical normalize !False = Acchen vs True = Ampcor-like
        end module denseoffsetsState 
