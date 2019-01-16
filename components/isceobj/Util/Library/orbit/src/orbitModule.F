        module orbitModule
            use, intrinsic:: iso_c_binding
            implicit none

            type, bind(C) :: orbitType
                integer(C_INT) :: nVectors 
                character(C_CHAR), dimension(256) :: yyyymmdd
                type(C_PTR) :: pos
                type(C_PTR) :: vel
                integer(C_INT) :: basis
            end type orbitType

            type, bind(C) :: stateVectorType
                real(C_DOUBLE) :: time
                real(C_DOUBLE), dimension(3) :: pos
                real(C_DOUBLE), dimension(3) :: vel
            end type stateVectorType

            integer, parameter :: WGS84_ORBIT = 1
            integer, parameter :: SCH_ORBIT = 2


            interface

                subroutine initOrbit_f(orb,nvec,basis)BIND(C,NAME='initOrbit')
                    use, intrinsic :: iso_c_binding
                    import :: orbitType
                    type(orbitType) :: orb
                    integer(C_INT), value :: nvec
                    integer(C_INT), value :: basis
                end subroutine initOrbit_f

                subroutine cleanOrbit_f(orb)BIND(C,NAME='cleanOrbit')
                    use, intrinsic :: iso_c_binding
                    import :: orbitType
                    type(orbitType) :: orb
                end subroutine cleanOrbit_f

                subroutine printOrbit_f(orb)BIND(C,NAME='printOrbit')
                    use, intrinsic :: iso_c_binding
                    import :: orbitType
                    type(orbitType) :: orb
                end subroutine printOrbit_f

                subroutine getStateVector_f(orb,ind,t,pos,vel)BIND(C,NAME='getStateVector')
                    use, intrinsic :: iso_c_binding
                    import :: orbitType
                    type(orbitType) :: orb
                    integer(C_INT), value :: ind
                    real(C_DOUBLE) :: t
                    real(C_DOUBLE), dimension(3) :: pos
                    real(C_DOUBLE), dimension(3) :: vel
                end subroutine getStateVector_f

                subroutine setStateVector_f(orb,ind,t,pos,vel)BIND(C,NAME='setStateVector')
                    use, intrinsic :: iso_c_binding
                    import :: orbitType
                    type(orbitType) :: orb
                    integer(C_INT), value :: ind
                    real(C_DOUBLE), value :: t
                    real(C_DOUBLE), dimension(3) :: pos
                    real(C_DOUBLE), dimension(3) :: vel
                end subroutine setStateVector_f

                function interpolateWGS84Orbit_f(orb,t,pos,vel)BIND(C,NAME='interpolateWGS84Orbit')
                    use, intrinsic :: iso_c_binding
                    import :: orbitType
                    type(orbitType) :: orb
                    real(C_DOUBLE), value :: t
                    real(C_DOUBLE), dimension(3) :: pos
                    real(C_DOUBLE), dimension(3) :: vel
                    integer(C_INT) :: interpolateWGS84Orbit_f
                end function interpolateWGS84Orbit_f

                function interpolateLegendreOrbit_f(orb,t,pos,vel)BIND(C,NAME='interpolateLegendreOrbit')
                    use, intrinsic :: iso_c_binding
                    import :: orbitType
                    type(orbitType) :: orb
                    real(C_DOUBLE), value :: t
                    real(C_DOUBLE), dimension(3) :: pos
                    real(C_DOUBLE), dimension(3) :: vel
                    integer(C_INT) :: interpolateLegendreOrbit_f
                end function interpolateLegendreOrbit_f

                function interpolateSCHOrbit_f(orb,t,pos,vel)BIND(C,NAME='interpolateSCHOrbit')
                    use, intrinsic :: iso_c_binding
                    import :: orbitType
                    type(orbitType) :: orb
                    real(C_DOUBLE), value :: t
                    real(C_DOUBLE), dimension(3) :: pos
                    real(C_DOUBLE), dimension(3) :: vel
                    integer(C_INT) :: interpoateSCHOrbit_f
                end function interpolateSCHOrbit_f

                function computeAcceleration_f(orb,t,acc)BIND(C,NAME='computeAcceleration')
                    use, intrinsic :: iso_c_binding
                    import :: orbitType
                    type(orbitType) :: orb
                    real(C_DOUBLE), value :: t
                    real(C_DOUBLE), dimension(3) :: acc
                    integer(C_INT) :: computeAcceleration_f
                end function computeAcceleration_f

                integer function interpolateOrbit_f(orb,t,pos,vel)
                    use, intrinsic :: iso_c_binding
                    import :: orbitType
                    type(orbitType) :: orb
                    real(C_DOUBLE), value :: t
                    real(C_DOUBLE), dimension(3) :: pos
                    real(C_DOUBLE), dimension(3) :: vel
                end function interpolateOrbit_f

            end interface

            contains
                function loadFromHDR_f(filename, basis)
                    character(len=*) filename
                    character(len=256) line
                    integer :: basis
                    type(orbitType) :: loadFromHDR_f, orb
                    double precision, dimension(3) :: pos, vel
                    double precision :: t
                    integer :: nLines, i, unit

                    unit = 11
                    open(unit, file=filename)
                    nLines = 0
                    do
                        read(unit,'(a)',end=20)line
                        nLines = nLines+1
                    enddo

20                  close(unit)
                    
                    unit = 11
                    i=0
                    call initOrbit_f(orb, nLines, i)
                    unit = 11
                    open(unit, file=filename)
                    do i=1,nLines
                        read(unit,'(a)') line
                        read(line,*) t,pos(1),pos(2),pos(3),vel(1),vel(2),vel(3)
                        call setStateVector_f(orb, i-1, t, pos, vel)
                    end do
                    close(unit)
                    loadFromHDR_f = orb
                end function loadFromHDR_f

        end module orbitModule



