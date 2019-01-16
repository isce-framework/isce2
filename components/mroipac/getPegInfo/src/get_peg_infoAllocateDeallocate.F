c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
c                                  Giangi Sacco
c                        NASA Jet Propulsion Laboratory
c                      California Institute of Technology
c                        (C) 2009-2010  All Rights Reserved
c
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        subroutine allocate_r_time(dim1)
            use get_peg_infoState
            implicit none
            integer dim1
            dim1_r_time = dim1
            allocate(r_time(dim1)) 
        end

        subroutine deallocate_r_time()
            use get_peg_infoState
            deallocate(r_time) 
        end

        subroutine allocate_r_xyz1(dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2
            dim1_r_xyz1 = dim2
            dim2_r_xyz1 = dim1
            allocate(r_xyz1(dim2,dim1)) 
        end

        subroutine deallocate_r_xyz1()
            use get_peg_infoState
            deallocate(r_xyz1) 
        end

        subroutine allocate_r_vxyz1(dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2
            dim1_r_vxyz1 = dim2
            dim2_r_vxyz1 = dim1
            allocate(r_vxyz1(dim2,dim1)) 
        end

        subroutine deallocate_r_vxyz1()
            use get_peg_infoState
            deallocate(r_vxyz1) 
        end

        subroutine allocate_r_axyz1(dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2
            dim1_r_axyz1 = dim2
            dim2_r_axyz1 = dim1
            allocate(r_axyz1(dim2,dim1)) 
        end

        subroutine deallocate_r_axyz1()
            use get_peg_infoState
            deallocate(r_axyz1) 
        end

        subroutine allocate_r_af(dim1)
            use get_peg_infoState
            implicit none
            integer dim1
            dim1_r_af = dim1
            allocate(r_af(dim1)) 
        end

        subroutine deallocate_r_af()
            use get_peg_infoState
            deallocate(r_af) 
        end

        subroutine allocate_r_cf(dim1)
            use get_peg_infoState
            implicit none
            integer dim1
            dim1_r_cf = dim1
            allocate(r_cf(dim1)) 
        end

        subroutine deallocate_r_cf()
            use get_peg_infoState
            deallocate(r_cf) 
        end

        subroutine allocate_r_afdot(dim1)
            use get_peg_infoState
            implicit none
            integer dim1
            dim1_r_afdot = dim1
            allocate(r_afdot(dim1)) 
        end

        subroutine deallocate_r_afdot()
            use get_peg_infoState
            deallocate(r_afdot) 
        end

        subroutine allocate_r_cfdot(dim1)
            use get_peg_infoState
            implicit none
            integer dim1
            dim1_r_cfdot = dim1
            allocate(r_cfdot(dim1)) 
        end

        subroutine deallocate_r_cfdot()
            use get_peg_infoState
            deallocate(r_cfdot) 
        end

        subroutine allocate_r_sfdot(dim1)
            use get_peg_infoState
            implicit none
            integer dim1
            dim1_r_sfdot = dim1
            allocate(r_sfdot(dim1)) 
        end

        subroutine deallocate_r_sfdot()
            use get_peg_infoState
            deallocate(r_sfdot) 
        end

        subroutine allocate_r_transVect(dim1)
            use get_peg_infoState
            implicit none
            integer dim1
            dim1_r_transVect = dim1
            allocate(r_transVect(dim1)) 
        end

        subroutine deallocate_r_transVect()
            use get_peg_infoState
            deallocate(r_transVect) 
        end

        subroutine allocate_r_transfMat(dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2
            dim1_r_transfMat = dim1
            dim2_r_transfMat = dim2
            allocate(r_transfMat(dim1,dim2)) 
        end

        subroutine deallocate_r_transfMat()
            use get_peg_infoState
            deallocate(r_transfMat) 
        end

        subroutine allocate_r_intPos(dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2
            dim1_r_intPos = dim1
            dim2_r_intPos = dim2
            allocate(r_intPos(dim1,dim2)) 
        end

        subroutine deallocate_r_intPos()
            use get_peg_infoState
            deallocate(r_intPos) 
        end

        subroutine allocate_r_intVel(dim1,dim2)
            use get_peg_infoState
            implicit none
            integer dim1,dim2
            dim1_r_intVel = dim1
            dim2_r_intVel = dim2
            allocate(r_intVel(dim1,dim2)) 
        end

        subroutine deallocate_r_intVel()
            use get_peg_infoState
            deallocate(r_intVel) 
        end

        subroutine allocate_r_vxyzpeg(dim1)
            use get_peg_infoState
            implicit none
            integer dim1
            dim1_r_vxyzpeg = dim1
            allocate(r_vxyzpeg(dim1)) 
        end

        subroutine deallocate_r_vxyzpeg()
            use get_peg_infoState
            deallocate(r_vxyzpeg) 
        end

        subroutine allocate_r_platvel(dim1)
            use get_peg_infoState
            implicit none
            integer dim1
            dim1_r_platvel = dim1
            allocate(r_platvel(dim1)) 
        end

        subroutine deallocate_r_platvel()
            use get_peg_infoState
            deallocate(r_platvel) 
        end

        subroutine allocate_r_platacc(dim1)
            use get_peg_infoState
            implicit none
            integer dim1
            dim1_r_platacc = dim1
            allocate(r_platacc(dim1)) 
        end

        subroutine deallocate_r_platacc()
            use get_peg_infoState
            deallocate(r_platacc) 
        end

