c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
c                                  Giangi Sacco
c                        NASA Jet Propulsion Laboratory
c                      California Institute of Technology
c                        (C) 2009-2010  All Rights Reserved
c
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        subroutine getPegLat(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            varInt = r_pegLat
        end

        subroutine getPegLon(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            varInt = r_pegLon
        end

        subroutine getPegHeight(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            varInt = r_pegHgt
        end

        subroutine getPegHeading(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            varInt = r_pegHead
        end

        subroutine getVerticalFit(array1d,dim1)
            use get_peg_infoState
            implicit none
            integer dim1,i
            real*8, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = r_af(i)
            enddo
        end

        subroutine getHorizontalFit(array1d,dim1)
            use get_peg_infoState
            implicit none
            integer dim1,i
            real*8, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = r_cf(i)
            enddo
        end

        subroutine getVerticalVelocityFit(array1d,dim1)
            use get_peg_infoState
            implicit none
            integer dim1,i
            real*8, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = r_afdot(i)
            enddo
        end

        subroutine getCrossTrackVelocityFit(array1d,dim1)
            use get_peg_infoState
            implicit none
            integer dim1,i
            real*8, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = r_cfdot(i)
            enddo
        end

        subroutine getAlongTrackVelocityFit(array1d,dim1)
            use get_peg_infoState
            implicit none
            integer dim1,i
            real*8, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = r_sfdot(i)
            enddo
        end

        subroutine getPegRadius(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            varInt = r_pegRadius
        end

        subroutine getGroundSpacing(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            varInt = r_grndSpace
        end

        subroutine getTranslationVector(array1d,dim1)
            use get_peg_infoState
            implicit none
            integer dim1,i
            real*8, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = r_transVect(i)
            enddo
        end

        subroutine getTransformationMatrix(array2d,dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2,i,j
            real*8, dimension(dim2,dim1):: array2d
            do i = 1, dim1
                do j = 1, dim2
                    array2d(j,i) = r_transfMat(i,j)
                enddo
            enddo
        end

        subroutine getIntPosition(array2d,dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2,i,j
            real*8, dimension(dim2,dim1):: array2d
            do i = 1, dim1
                do j = 1, dim2
                    array2d(j,i) = r_intPos(i,j)
                enddo
            enddo
        end

        subroutine getIntVelocity(array2d,dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2,i,j
            real*8, dimension(dim2,dim1):: array2d
            do i = 1, dim1
                do j = 1, dim2
                    array2d(j,i) = r_intVel(i,j)
                enddo
            enddo
        end

        subroutine getPegVelocity(array1d,dim1)
            use get_peg_infoState
            implicit none
            integer dim1,i
            real*8, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = r_vxyzpeg(i)
            enddo
        end

        subroutine getPlatformSCHVelocity(array1d,dim1)
            use get_peg_infoState
            implicit none
            integer dim1,i
            real*8, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = r_platvel(i)
            enddo
        end

        subroutine getPlatformSCHAcceleration(array1d,dim1)
            use get_peg_infoState
            implicit none
            integer dim1,i
            real*8, dimension(dim1):: array1d
            do i = 1, dim1
                array1d(i) = r_platacc(i)
            enddo
        end

        subroutine getTimeFirstScene(varInt)
            use get_peg_infoState
            implicit none
            real*8 varInt
            varInt = r_time_first_line
        end

