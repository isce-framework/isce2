            function cosineC(a,b,c)BIND(C,NAME='cosineC_C')
                use, intrinsic :: iso_c_binding
                implicit none

                real(C_DOUBLE), value :: a,b,c
                real(C_DOUBLE) :: cosineC
                real*8 val
                val = ((a*a+b*b-c*c)/(2*a*b))
                cosineC = acos(val)
            end function cosineC

