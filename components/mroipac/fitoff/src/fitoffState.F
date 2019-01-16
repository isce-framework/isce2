        module fitoffState
            integer*8 ptStdWriter
            integer   miniter, maxiter 
            logical   l1norm
            integer   minpoint,nmax, imax
            real*8 nsig,maxrms

            !!Arrays for offsets
            double precision, allocatable, dimension(:) :: x1o, y1o
            double precision, allocatable, dimension(:) :: dx, dy
            double precision, allocatable, dimension(:) :: x2o, y2o
            double precision, allocatable, dimension(:) :: r_covac, r_covdn
            double precision, allocatable, dimension(:) :: r_covx, snr

            parameter(nmax=8)
            real*8 x (nmax)
        end
