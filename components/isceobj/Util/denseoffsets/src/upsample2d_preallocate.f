        module upsample2d_preallocate

            implicit none
            include "fftw3.f"
        ! 2d upsampling, using Fourier Transform

        contains

        subroutine upsample2d_complex(a,aup,Atrans,Auptrans,plan,plani,m,n,novr)
  ! upsample matrix a (size mxn) by a factor of novr
  ! output is in aa
  ! aup must be size(m*novr x n*novr)
  
  ! Atrans is workspace, must be size mxn
  ! Auptrans is workspace, must be size m*novr x n*novr

  ! plan and plani must be created thus:
  !  call sfftw_plan_dft_2d(plan,m,n,a,Atrans,FFTW_FORWARD,FFTW_ESTIMATE)
  !  call sfftw_plan_dft_2d(plani,m*novr,n*novr,Auptrans,aup,FFTW_BACKWARD,FFTW_ESTIMATE)

  ! input
            complex, dimension(:,:), intent(in) :: a
            integer, intent(in) :: m, n, novr
            integer*8, intent(in) :: plan, plani

  ! output
            complex, dimension(:,:), intent(out) :: aup
            complex, dimension(:,:), intent(out) :: Atrans,Auptrans

  ! computation variables
            integer :: nyqst_m, nyqst_n

  ! 2d fft
            call sfftw_execute(plan)

  ! Nyquist frequencies
            nyqst_m = ceiling((m+1)/2.0)
            nyqst_n = ceiling((n+1)/2.0)

  ! zero out spectra
            Auptrans(1:(m*novr),1:(n*novr)) = cmplx(0.0,0.0)

  ! copy spectra
            Auptrans(1:nyqst_m,1:nyqst_n) = Atrans(1:nyqst_m,1:nyqst_n);
            Auptrans(m*novr-nyqst_m+3:m*novr,1:nyqst_n) = Atrans(nyqst_m+1:m,1:nyqst_n);
            Auptrans(1:nyqst_m,n*novr-nyqst_n+3:n*novr) = Atrans(1:nyqst_m,nyqst_n+1:n);
            Auptrans(m*novr-nyqst_m+3:m*novr,n*novr-nyqst_n+3:n*novr) = Atrans(nyqst_m+1:m,nyqst_n+1:n);

            if(mod(m,2).eq.0)then
                Auptrans(nyqst_m,1:(n*novr)) = Auptrans(nyqst_m,1:(n*novr))/cmplx(2.0,0.0)
                Auptrans(m*novr-nyqst_m+2,1:(n*novr)) = Auptrans(nyqst_m,1:(n*novr))
            end if

            if(mod(n,2).eq.0) then
                Auptrans(1:(m*novr),nyqst_n) = Auptrans(1:(m*novr),nyqst_n)/cmplx(2.0,0.0)
                Auptrans(1:(m*novr),n*novr-nyqst_n+2) = Auptrans(1:(m*novr),nyqst_n)
            end if

  ! 2d inverse fft
            call sfftw_execute(plani)
  
  ! normalize
            aup = aup / cmplx(real(m*n),0.0)

        end subroutine upsample2d_complex

        end module upsample2d_preallocate

