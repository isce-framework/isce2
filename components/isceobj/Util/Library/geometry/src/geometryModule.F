!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!# Author: Piyush Agram
!# Copyright 2014, by the California Institute of Technology. ALL RIGHTS RESERVED.
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

        module geometryModule
            use, intrinsic :: iso_c_binding
            implicit none

            type, bind(C) :: pegType
                real(C_DOUBLE) :: r_lat
                real(C_DOUBLE) :: r_lon
                real(C_DOUBLE) :: r_hdg
            end type pegType

            type, bind(C) :: ellipsoidType
                real(C_DOUBLE) :: r_a
                real(C_DOUBLE) :: r_e2
            end type ellipsoidType

            type, bind(C) :: pegtransType
                real(C_DOUBLE), dimension(3,3) :: r_mat
                real(C_DOUBLE), dimension(3,3) :: r_matinv
                real(C_DOUBLE), dimension(3) :: r_ov
                real(C_DOUBLE) :: r_radcur
            end type pegtransType

            type, bind(C) :: positionType
                real(C_DOUBLE), dimension(3) :: r_j
                real(C_DOUBLE), dimension(3) :: r_jdot
                real(C_DOUBLE), dimension(3) :: r_jddt
            end type positionType

            integer, parameter :: SCH_2_XYZ = 0
            integer, parameter :: XYZ_2_SCH = 1
            integer, parameter :: LLH_2_UTM = 1
            integer, parameter :: UTM_2_LLH = 2
            integer, parameter :: LLH_2_XYZ = 1
            integer, parameter :: XYZ_2_LLH = 2
            integer, parameter :: XYZ_2_LLH_OLD = 3
        
        contains

            include 'convert_sch_to_xyz.F'
            include 'convert_schdot_to_xyzdot.F'
            include 'cosineC.F'
            include 'curvature.F'
            include 'enubasis.F'
            include 'schbasis.F'
            include 'getangs.F'
            include 'gettcn_tcvec.F'
            include 'latlon.F'
            include 'lookvec.F'
            include 'radar_to_xyz.F'
            include 'tcnbasis.F'

        end module geometryModule
