c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
c                                  Giangi Sacco
c                        NASA Jet Propulsion Laboratory
c                      California Institute of Technology
c                        (C) 2009  All Rights Reserved
c
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        module formslcState
                integer ngood1
                integer nbytes1
                integer iflag
                integer ideskew
                integer iflip
                integer isrm
                integer ifirstline
                integer npatches
                integer ifirst
                integer nnn
                integer na_valid
                real*4 caltone1
                integer isave
                integer nlinesaz
                real*8, dimension(4) ::  dopplerCoefficients
                integer dim1_dopplerCoefficients
                real*8 re
                real*4 vel1
                real*8 ht1
                real*8 gm
                integer i_lrl
                real*8, dimension(3) ::  r_platvel1
                integer dim1_r_platvel1
                real*8, dimension(3) ::  r_platacc1
                integer dim1_r_platacc1
                real*8 r001
                real*4 prf1
                real*4 xmi1
                real*4 xmq1
                real*4 azres
                integer nlooks
                real*4 fs
                real*4 slope
                real*4 pulsedur
                integer nextend
                real*8 wavl
                real*4 rhww
                real*4, dimension(2) ::  spectralShiftFrac
                real*8, dimension(4) ::  linearResampCoeff
                real*8, dimension(4) ::  linearResampDeltas
                real*8  slc_r0
                integer*4  slc_line0
        end module 
