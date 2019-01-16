c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
c                                  Giangi Sacco
c                        NASA Jet Propulsion Laboratory
c                      California Institute of Technology
c                        (C) 2009-2010  All Rights Reserved
c
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        subroutine setNumObservations(varInt)
            use get_peg_infoState
            implicit none
            integer varInt
            i_numobs = varInt
        end

        subroutine setStartLineSlc(varInt)
            use get_peg_infoState
            implicit none
            integer varInt
            i_startline = varInt
        end

        subroutine setNumLinesInt(varInt)
            use get_peg_infoState
            implicit none
            integer varInt
            i_numlines = varInt
        end

        subroutine setNumLinesSlc(varInt)
            use get_peg_infoState
            implicit none
            integer varInt
            i_slclines = varInt
        end

        subroutine setNumAzimuthLooksInt(varInt)
            use get_peg_infoState
            implicit none
            integer varInt
            i_looksaz = varInt
        end

        subroutine setPrfSlc(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            r_prf = varInt
        end

        subroutine setTimeSlc(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            r_timeslc = varInt
        end

        subroutine setTime(array1d,dim1)
            use get_peg_infoState
            implicit none
            integer dim1,i
            real*8, dimension(dim1):: array1d
            do i = 1, dim1
                r_time(i) = array1d(i)
            enddo
        end

        subroutine setPositionVector(array2dT,dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2,i,j
            real*8, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    r_xyz1(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setVelocityVector(array2dT,dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2,i,j
            real*8, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    r_vxyz1(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setAccelerationVector(array2dT,dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2,i,j
            real*8, dimension(dim2,dim1):: array2dT
            do i = 1, dim2
                do j = 1, dim1
                    r_axyz1(i,j) = array2dT(i,j)
                enddo
            enddo
        end

        subroutine setPlanetGM(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            r_gm = varInt
        end

        subroutine setPlanetSpinRate(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            r_spindot = varInt
        end

