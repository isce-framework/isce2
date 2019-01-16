!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!#
!#
!# Author: Piyush Agram
!# Copyright 2013, by the California Institute of Technology. ALL RIGHTS RESERVED.
!# United States Government Sponsorship acknowledged.
!# Any commercial use must be negotiated with the Office of Technology Transfer at
!# the California Institute of Technology.
!# This software may be subject to U.S. export control laws.
!# By accepting this software, the user agrees to comply with all applicable U.S.
!# export laws and regulations. User has the responsibility to obtain export licenses,
!# or other export authority as may be required before exporting such information to
!# foreign countries or providing access to foreign persons.
!#
!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            module geocodeReadWrite
                implicit none

                real*4, allocatable, dimension(:) :: rarr
                interface
                    subroutine readTemplate(acc,carr,band,irow,n)
                        integer*8 :: acc
                        complex, dimension(:) :: carr
                        integer:: irow,band,n
                    end subroutine readTemplate

                    subroutine writeTemplate(acc,carr,band,n)
                        integer*8 :: acc
                        complex, dimension(:) :: carr
                        integer:: band,n
                    end subroutine writeTemplate

                end interface

                contains
                    subroutine init_RW(width, iscomplex)
                         integer :: width, iscomplex
                         if(iscomplex.ne.1) allocate(rarr(width))
                    end subroutine init_RW

                    subroutine finalize_RW(iscomplex)
                        integer :: iscomplex
                        if(iscomplex.ne.1) deallocate(rarr)
                    end subroutine finalize_RW

                    subroutine readCpxLine(acc,carr,band,irow,n)
                        complex, dimension(:) :: carr
                        integer*8 :: acc
                        integer :: irow,band,n,i

                        call getLineSequentialBand(acc,carr,band,irow)
                    end subroutine readCpxLine

                    subroutine readRealLine(acc,carr,band,irow,n)
                        complex, dimension(:) :: carr
                        integer*8 :: acc
                        integer :: irow,band,n,i

                        call getLineSequentialBand(acc,rarr,band,irow)
                        do i=1,n
                            carr(i) = cmplx(rarr(i), 0.)
                        end do
                    end subroutine readRealLine

                    subroutine writeCpxLine(acc,carr,band,n)
                        complex, dimension(:) :: carr
                        integer*8 :: acc
                        integer :: band,n,i

                        call setLineSequentialBand(acc,carr,band)
                    end subroutine writeCpxLine

                    subroutine writeRealLine(acc,carr,band,n)
                        complex, dimension(:) :: carr
                        integer*8 :: acc
                        integer :: band,n,i

                        do i=1,n
                            rarr(i) = real(carr(i))
                        enddo
                        
                        call setLineSequentialBand(acc,rarr,band)
                    end subroutine writeRealLine

                end module geocodeReadWrite
