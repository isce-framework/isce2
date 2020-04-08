!****************************************************************
!** Fortran interfaces for C binding
!****************************************************************

    subroutine c_fitoff(infile,outfile,nsig,maxrms,minpoint) bind(c, name="c_fitoff")
        use iso_c_binding, only : c_double, c_char, c_int
        implicit none
        external fitoff

        ! input parameters
        character(kind=c_char), dimension(*), intent(in) :: infile, outfile
        real(kind=c_double), intent(in) :: nsig, maxrms
        integer(kind=c_int), intent(in) :: minpoint

        ! call
        call fitoff(infile,outfile,nsig,maxrms,minpoint)
    end subroutine

    subroutine c_rect(infile,outfile,ndac,nddn,nrac,nrdn,a,b,c,d,e,f,filetype,intstyle) bind(c, name="c_rect")
        use iso_c_binding, only : c_double, c_char, c_int
        implicit none
        external rect

        character(kind=c_char), dimension(*), intent(in) :: infile, outfile, intstyle, filetype
        integer(kind=c_int), intent(in) :: ndac, nddn, nrac, nrdn
        real(kind=c_double), intent(in) :: a,b,c,d,e,f

        call rect(infile,outfile,ndac,nddn,nrac,nrdn,a,b,c,d,e,f,filetype,intstyle)
    end subroutine

    subroutine c_rect_with_looks(infile,outfile,ndac,nddn,nrac,nrdn,a,b,c,d,e,f,lac,ldn,lac0,ldn0,filetype,intstyle) &
            bind(c, name="c_rect_with_looks")
        use iso_c_binding, only : c_double, c_char, c_int
        implicit none
        external rect_with_looks

        character(kind=c_char), dimension(*), intent(in) :: infile, outfile, intstyle, filetype
        integer(kind=c_int), intent(in) :: ndac, nddn, nrac, nrdn
        real(kind=c_double), intent(in) :: a,b,c,d,e,f
        integer(kind=c_int), intent(in) :: lac, ldn, lac0, ldn0

        call rect_with_looks(infile,outfile,ndac,nddn,nrac,nrdn,a,b,c,d,e,f,lac,ldn,lac0,ldn0,filetype,intstyle)
    end subroutine
