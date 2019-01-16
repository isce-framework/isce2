c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
c                                  Giangi Sacco
c                        NASA Jet Propulsion Laboratory
c                      California Institute of Technology
c                        (C) 2009-2010  All Rights Reserved
c
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        module get_peg_infoState
            integer i_numobs
            integer i_startline
            integer i_numlines
            integer i_slclines
            integer i_looksaz
            real*8 r_prf
            real*8 r_timeslc
            real*8, allocatable, dimension(:) ::  r_time
            integer dim1_r_time
            real*8, allocatable, dimension(:,:) ::  r_xyz1
            integer dim1_r_xyz1, dim2_r_xyz1
            real*8, allocatable, dimension(:,:) ::  r_vxyz1
            integer dim1_r_vxyz1, dim2_r_vxyz1
            real*8, allocatable, dimension(:,:) ::  r_axyz1
            integer dim1_r_axyz1, dim2_r_axyz1
            real*8 r_gm
            real*8 r_spindot
            real*8 r_pegLat
            real*8 r_pegLon
            real*8 r_pegHgt
            real*8 r_pegHead
            real*8, allocatable, dimension(:) ::  r_af
            integer dim1_r_af
            real*8, allocatable, dimension(:) ::  r_cf
            integer dim1_r_cf
            real*8, allocatable, dimension(:) ::  r_afdot
            integer dim1_r_afdot
            real*8, allocatable, dimension(:) ::  r_cfdot
            integer dim1_r_cfdot
            real*8, allocatable, dimension(:) ::  r_sfdot
            integer dim1_r_sfdot
            real*8 r_pegRadius
            real*8 r_grndSpace
            real*8, allocatable, dimension(:) ::  r_transVect
            integer dim1_r_transVect
            real*8, allocatable, dimension(:,:) ::  r_transfMat
            integer dim1_r_transfMat, dim2_r_transfMat
            real*8, allocatable, dimension(:,:) ::  r_intPos
            integer dim1_r_intPos, dim2_r_intPos
            real*8, allocatable, dimension(:,:) ::  r_intVel
            integer dim1_r_intVel, dim2_r_intVel
            real*8, allocatable, dimension(:) ::  r_vxyzpeg
            integer dim1_r_vxyzpeg
            real*8, allocatable, dimension(:) ::  r_platvel
            integer dim1_r_platvel
            real*8, allocatable, dimension(:) ::  r_platacc
            integer dim1_r_platacc
            real*8 r_time_first_line
        end module 
