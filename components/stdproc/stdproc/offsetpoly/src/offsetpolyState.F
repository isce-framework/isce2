!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
! Author: Piyush Agram
! Copyright 2013, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
! Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
!
! This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S.
! export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before 
! exporting such information to foreign countries or providing access to foreign persons.
!
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        module offsetpolyState
            integer i_ma             !c Number of fit Coefficients
            integer numOffsets       !c Number of offsets

            !c Location across array 1
            double precision, allocatable, dimension(:) ::  r_ranposV

            !c Location across offset array 1
            double precision, allocatable, dimension(:) ::  r_offV

            !c Location down array 1
            double precision, allocatable, dimension(:) ::  r_azposV

            !c SNR array 1
            double precision, allocatable, dimension(:) ::  r_sigV

            !Polynomial arrays
            double precision, allocatable, dimension(:) :: r_polyV
        end module offsetpolyState 
