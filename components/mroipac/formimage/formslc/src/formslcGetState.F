c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
c                                  Giangi Sacco
c                        NASA Jet Propulsion Laboratory
c                      California Institute of Technology
c                        (C) 2009  All Rights Reserved
c
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        subroutine getSLCStartingRange(varDbl)
                use formslcState
                implicit none
                double precision varDbl
                varDbl = slc_r0
        end subroutine

        subroutine getSLCStartingLine(varInt)
                use formslcState
                implicit none
                integer varInt
                varInt = slc_line0
        end subroutine

