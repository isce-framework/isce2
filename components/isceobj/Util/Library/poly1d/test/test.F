        program test
            use, intrinsic :: iso_c_binding
            use poly2dState
            type(poly2dType) :: res 
            integer :: i,j
            double precision ::cnt 
            integer :: rgOrder, azOrder

            azOrder = 2
            rgOrder = 3
            call initPoly2d_f(res, azOrder,rgOrder)

            print *, 'Setting coeffs'
            cnt = 0.0 
            do i=0,azOrder
                do j=0,rgOrder
                    print *, i, j, cnt
                    call setCoeff_f(res, i, j, cnt)
                    cnt = cnt+1
                end do
            end do


            print *, 'Print out coeffs: '
            do i=0, azOrder
                do j=0,rgOrder
                    print *, i, j, getCoeff_f(res, i, j)
                end do
            end do

            call printPoly2d_f(res)

            print *, 'Testing evaluation of polynomials'
            cnt =  evalPoly2d_f(res, 0.0d0, 0.0d0)
            print *, 0, 0, cnt

            cnt = evalPoly2d_f(res, 0.0d0, 1.0d0)
            print *, 0, 1, cnt

            cnt = evalPoly2d_f(res, 1.0d0, 0.0d0)
            print *, 1, 0, cnt

            cnt = evalPoly2d_f(res, 1.0d0, 1.0d0)
            print *, 1, 1, cnt
            call cleanPoly2d_f(res)

        end program

