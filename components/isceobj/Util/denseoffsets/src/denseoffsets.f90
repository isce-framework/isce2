! offset
! A.C.Chen, January, 2010 
      
subroutine denseoffsets(img1,img2,offset,snr)

use omp_lib
use denseoffsetsState
use denseoffsetsRead
use upsample2d_preallocate

implicit none

!!!  DECLARATIONS

! parallelization using OpenMP
! NOTE: If your computer supports more than 16 threads,
!  change Nth below to the maximum number of threads.
!integer, parameter :: Nth = 16   ! number of threads
integer :: ith   ! thread number
integer :: Nth   ! Number of threads

integer*8 img1,img2,offset, snr
integer :: NPTSW2, NPTSH2, NPTSWby2
integer :: NPTSW4, NPTSH4, NPTSHby2
integer :: NDISPby2,NLARGE

!Computation variables
complex, dimension(:,:,:), allocatable :: a, b, aa, bb
complex, dimension(:,:,:), allocatable :: pa, pb, cpa, cpb
complex, dimension(:,:,:), allocatable :: c, ctrans, cpiece, corr
complex, dimension(:,:,:), allocatable :: Atrans, Auptrans
complex, dimension(:,:,:), allocatable :: Btrans, Buptrans
complex, dimension(:,:,:), allocatable :: CPIECEtrans, CPIECEuptrans
complex, dimension(:,:,:), allocatable :: fsum, fsum2

complex :: prod4, prod2, prodpiece
complex :: paave, pbave, esum, e2sum
integer :: ii,jj,kk

!!FFT plans
integer*8, dimension(:), allocatable :: plan_pa, plan_pb, plan_ctrans, plan_a
integer*8, dimension(:), allocatable :: plan_ai, plan_b, plan_bi
integer*8, dimension(:), allocatable :: plan_cpiece, plan_cpiecei

real :: cmax
integer :: imax,jmax

! file i/o
complex, dimension(:,:), allocatable :: s1, s2
real, dimension(:), allocatable :: rdata

! locations of offset estimates
integer, dimension(:), allocatable :: az_loc, rg_loc
real*4, dimension(:), allocatable :: az_off, rg_off, snr_off
integer :: iazloc, irgloc
integer :: az_num, ilineno
integer :: numrg, numaz

! offset estimates
integer :: rough_az_off, rough_rg_off
integer :: gross_az_off, gross_rg_off

! runtime
real :: seconds

interface
   subroutine fftshift2d(a,m,n)
     complex, dimension(:,:) :: a
     integer :: m,n
   end subroutine fftshift2d

   subroutine derampc(a,m,n)
       complex, dimension(:,:) :: a
       integer :: m,n
    end subroutine derampc

    subroutine readTemplate(acc,arr,iraw,band,n,carr)
        integer*8 :: acc
        complex, dimension(:) :: carr
        real, dimension(:) :: arr
        integer :: irow,band,n
    end subroutine readTemplate
end interface

procedure(readTemplate), pointer :: readBand1 => null()
procedure(readTemplate), pointer :: readBand2 => null()


! runtime
seconds = omp_get_wtime()

! display number of threads
! Capping number of threads at 16
!$omp parallel shared(Nth)
!$omp master
Nth = omp_get_num_threads()
Nth = min(16,Nth)
write(*,*) 'Max threads used: ', Nth
!$omp end master
!$omp end parallel

call omp_set_num_threads(Nth)
numaz = ((isamp_fdn - isamp_sdn)/(iskipdn)) 
numrg = ((isamp_f - isamp_s)/(iskipac))

NPTSW2 = 2*NPTSW
NPTSH2 = 2*NPTSH
NPTSW4 = 4*NPTSW
NPTSH4 = 4*NPTSH
NPTSWby2 = NPTSW/2
NPTSHby2 = NPTSH/2
NDISPby2 = NDISP/2
NLARGE = NDISP*NOVS
prod4 = cmplx(real(NPTSH4*NPTSW4),0.)
prod2 = cmplx(real(NPTSH2*NPTSW2),0.)
prodpiece = cmplx(real(NDISP*NDISP), 0.)

! allocate memory
allocate( s1(NPTSH,len1), s2(NPTSH2,len2) )
allocate( rdata(max(len1,len2)))
allocate( az_loc(numaz), rg_loc(numrg) )
allocate( az_off(numrg), rg_off(numrg), snr_off(numrg))


!!Allocate memory for plans
allocate(plan_pa(Nth), plan_pb(Nth), plan_ctrans(Nth), plan_a(Nth))
allocate(plan_ai(Nth), plan_b(Nth), plan_bi(Nth))
allocate(plan_cpiece(Nth), plan_cpiecei(Nth))

!!Allocate memory for arrays
allocate( a(NPTSH,NPTSW,Nth), b(NPTSH2,NPTSW2,Nth))
allocate(aa(NPTSH2,NPTSW2,Nth), bb(NPTSH4,NPTSW4,Nth))
allocate(pa(NPTSH4,NPTSW4,Nth), pb(NPTSH4,NPTSW4,Nth))
allocate(cpa(NPTSH4,NPTSW4,Nth), cpb(NPTSH4,NPTSW4,Nth))
allocate(c(NPTSH4,NPTSW4,Nth), ctrans(NPTSH4,NPTSW4,Nth))
allocate(cpiece(NDISP,NDISP,Nth), corr(NLARGE,NLARGE,Nth)) 
allocate(Atrans(NPTSH,NPTSW,Nth), Auptrans(NPTSH2,NPTSW2,Nth))
allocate(Btrans(NPTSH2,NPTSW2,Nth), Buptrans(NPTSH4,NPTSW4,Nth))
allocate(CPIECEtrans(NDISP,NDISP,Nth), CPIECEuptrans(NLARGE,NLARGE,Nth))

if(normalize) then
    allocate( fsum(NPTSH4,NPTSW4,Nth), fsum2(NPTSH4,NPTSW4,Nth))
endif


! make FFT plans
do ii=1,Nth
   call sfftw_plan_dft_2d(plan_pa(ii),NPTSH4,NPTSW4,pa(1,1,ii),cpa(1,1,ii),FFTW_FORWARD,FFTW_ESTIMATE)
   call sfftw_plan_dft_2d(plan_pb(ii),NPTSH4,NPTSW4,pb(1,1,ii),cpb(1,1,ii),FFTW_FORWARD,FFTW_ESTIMATE)
   call sfftw_plan_dft_2d(plan_ctrans(ii),NPTSH4,NPTSW4,ctrans(1,1,ii),c(1,1,ii),FFTW_BACKWARD,FFTW_ESTIMATE)

   call sfftw_plan_dft_2d(plan_a(ii),NPTSH,NPTSW,a(1,1,ii),Atrans(1,1,ii),FFTW_FORWARD,FFTW_ESTIMATE)
   call sfftw_plan_dft_2d(plan_ai(ii),NPTSH2,NPTSW2,Auptrans(1,1,ii),aa(1,1,ii),FFTW_BACKWARD,FFTW_ESTIMATE)

   call sfftw_plan_dft_2d(plan_b(ii),NPTSH2,NPTSW2,b(1,1,ii),Btrans(1,1,ii),FFTW_FORWARD,FFTW_ESTIMATE)
   call sfftw_plan_dft_2d(plan_bi(ii),NPTSH4,NPTSW4,Buptrans(1,1,ii),bb(1,1,ii),FFTW_BACKWARD,FFTW_ESTIMATE)

   call sfftw_plan_dft_2d(plan_cpiece(ii),NDISP,NDISP,cpiece(1,1,ii),CPIECEtrans(1,1,ii), &
        FFTW_FORWARD,FFTW_ESTIMATE)
   call sfftw_plan_dft_2d(plan_cpiecei(ii),NLARGE,NLARGE,CPIECEuptrans(1,1,ii),corr(1,1,ii), &
        FFTW_BACKWARD,FFTW_ESTIMATE)
end do

! calculate locations
do ii=1,numaz
   az_loc(ii) = isamp_sdn + (ii-1)*iskipdn
end do
print *, 'Azimuth start, end, skip, num.', az_loc(1), az_loc(numaz), iskipdn, numaz
print *, 'Lines: ', lines1, lines2


do ii=1,numrg
   rg_loc(ii) = isamp_s + (ii-1)*iskipac
end do
print *, 'Range start, end, skip, num.', rg_loc(1), rg_loc(numrg), iskipac, numrg
print *, 'Widths: ', len1, len2

print *, 'Gross offset at top left: ', ioffdn, ioffac
print *, 'Scale Factors: ', scaley, scalex
if (normalize) then
    print *, 'Using ampcor hybrid algorithm'
else
    print *, 'Using unnormalize covariance algorithm'
endif

! +++++++++ LOOP OVER LOCATIONS +++++++++

if(iscpx1.eq.1) then
    readBand1 => readCpxAmp
    print *, 'Band1 is complex'
else
    readBand1 => readAmp
    print *, 'Band1 is real'
endif

if(iscpx2.eq.1) then
    readBand2 => readCpxAmp
    print *, 'Band2 is complex'
else
    readBand2 => readAmp
    print *, 'Band2 is real'
endif


! loop over azimuth locations
az_num = 0
az_loc_loop : do iazloc=1,numaz

   az_num = az_num+1
   if (mod(az_num,100)==0) then
      print *,'az_loc: ',az_loc(iazloc)
   end if
 
   gross_az_off = nint((scaley-1)*az_loc(iazloc))+ioffdn
!!   print *, 'gross az: ', iazloc, gross_az_off

   !!Read channel 1 data
   do ii=1,NPTSH
      ilineno = az_loc(iazloc)- NPTSHby2 + ii
!      print *, 'Image 1: ', ilineno, ii
      if ((ilineno.ge.1).and.(ilineno.le.lines1)) then
          call readBand1(img1, rdata, ilineno, band1, len1, s1(ii,:))
      else
          s1(ii,:) = cmplx(0., 0.)
      endif
   end do 

   ! read channel 2 data
   do ii=1,NPTSH2
      ilineno = az_loc(iazloc) + gross_az_off-NPTSH+ii
!      print *, 'Image 2: ', ilineno, ii
      if((ilineno.ge.1).and.(ilineno.le.lines2)) then
          call readBand2(img2, rdata, ilineno, band2, len2, s2(ii,:))
      else
          s2(ii,:) = cmplx(0., 0.)
      endif
    end do

   ! loop over range locations

   !$omp parallel do default(private) shared(s1,s2,pa,cpa,pb,cpb,&
   !$omp &ctrans,c,a,Atrans,Auptrans,aa,b,Btrans,Buptrans,bb,cpiece,&
   !$omp &CPIECEtrans,CPIECEuptrans,corr,plan_pa,plan_pb,plan_ctrans,&
   !$omp &plan_a,plan_ai,plan_b,plan_bi,plan_cpiece,plan_cpiecei,&
   !$omp &az_loc,iazloc,rg_loc,gross_az_off,az_off,rg_off,snr_off,&
   !$omp &NPTSH,NPTSW,NPTSH2,NPTSW2,NPTSH4,NPTSW4,NPTSHby2,NPTSWby2,&
   !$omp &NDISP,NOVS,NOFFH,NOFFW,NDISPby2,NLARGE,scalex,ioffac,&
   !$omp &len1,len2,prod2,prod4,fsum,fsum2,normalize,prodpiece)
   rg_loc_loop : do irgloc=1,numrg

      ! get thread number
      ith = omp_get_thread_num() + 1

      gross_rg_off = nint((scalex-1)*rg_loc(irgloc))+ioffac

!!      print *, az_loc(iazloc), rg_loc(irgloc), gross_az_off, gross_rg_off
      ! put data into buffers a and b
      do ii=1,NPTSH
         do jj=1,NPTSW
            kk = rg_loc(irgloc)-NPTSWby2+jj
            kk = max(1, min(kk,len1))
            a(ii,jj,ith) = s1(ii,kk)
         end do
      end do

      call derampc(a(1:NPTSH,1:NPTSW,ith), NPTSH, NPTSW)

      do ii=1,NPTSH2
         do jj=1,NPTSW2
            kk = rg_loc(irgloc) + gross_rg_off-NPTSW+jj
            kk = max(1, min(kk,len2))
            b(ii,jj,ith) = s2(ii,kk)
         end do
      end do

      call derampc(b(1:NPTSH2,1:NPTSW2,ith), NPTSH2, NPTSW2)

      ! upsample by 2
      call upsample2d_complex(a(1:NPTSH,1:NPTSW,ith),aa(1:NPTSH2,1:NPTSW2,ith),Atrans(1:NPTSH,1:NPTSW,ith), Auptrans(1:NPTSH2,1:NPTSW2,ith),plan_a(ith),plan_ai(ith),NPTSH,NPTSW,2)
      call upsample2d_complex(b(1:NPTSH2,1:NPTSW2,ith),bb(1:NPTSH4,1:NPTSW4,ith),Btrans(1:NPTSH2,1:NPTSW2,ith), Buptrans(1:NPTSH4,1:NPTSW4,ith),plan_b(ith),plan_bi(ith),NPTSH2,NPTSW2,2)


      ! pb magnitudes
      pbave = cmplx(0.,0.)

      do ii=1,NPTSH4
         do jj=1,NPTSW4
            pb(ii,jj,ith) = cmplx(abs(bb(ii,jj,ith)),0.0)
            pbave = pbave + pb(ii,jj,ith)/prod4
         end do
      end do

      do ii=1,NPTSH4
         do jj=1,NPTSW4
            pb(ii,jj,ith) = pb(ii,jj,ith) - pbave
         end do
      end do

      ! zero out pa matrix
      do ii=1,NPTSH4
         do jj=1,NPTSW4
            pa(ii,jj,ith) = cmplx(0.0,0.0)
         end do
      end do

      ! pa magnitudes
      paave = 0.0
      do ii=1,NPTSH2
         do jj=1,NPTSW2
            pa(ii+NPTSH,jj+NPTSW,ith) = cmplx(abs(aa(ii,jj,ith)),0.0)
            paave = paave + pa(ii+NPTSH,jj+NPTSW,ith)/prod2
         end do
      end do

      do ii=1,NPTSH2
         do jj=1,NPTSW2
            pa(ii+NPTSH,jj+NPTSW,ith) = pa(ii+NPTSH,jj+NPTSW,ith) - paave
         end do
      end do

      ! 2d fft
      call sfftw_execute(plan_pa(ith))  ! cpa = fft(pa)
      call sfftw_execute(plan_pb(ith))  ! cpb = fft(pb)

      do ii=1,NPTSH4
         do jj=1,NPTSW4
            ctrans(ii,jj,ith) = conjg(cpa(ii,jj,ith))*cpb(ii,jj,ith)
         end do
      end do

      ! inverse 2d fft      
      call sfftw_execute(plan_ctrans(ith))
      c(1:NPTSH4,1:NPTSW4,ith) = c(1:NPTSH4,1:NPTSw4,ith)/prod4

      call fftshift2d(c(1:NPTSH4,1:NPTSW4,ith),NPTSH4,NPTSW4)

      if(normalize) then   !!<>PSA - new code normalized correlation
            !!!Compute normalization factors
            fsum(1:NPTSH4,1:NPTSW4,ith) = cmplx(0.,0.)
            fsum2(1:NPTSH4,1:NPTSW4,ith) = cmplx(0.,0.)
            fsum(1,1,ith) = pb(1,1,ith)
            fsum(1,2,ith) = fsum(1,1,ith) + pb(1,2,ith)
            fsum(2,1,ith) = fsum(1,1,ith) + pb(2,1,ith)

            fsum2(1,1,ith) = pb(1,1,ith)**2.
            fsum2(1,2,ith) = fsum(1,1,ith) + pb(1,2,ith)**2.
            fsum2(2,1,ith) = fsum(1,1,ith) + pb(2,1,ith)**2.

            do ii=2,NPTSH4
                do jj=2,NPTSW4
                    fsum(ii,jj,ith) = fsum(ii-1,jj,ith)+fsum(ii,jj-1,ith)-fsum(ii-1,jj-1,ith)+pb(ii,jj,ith)
                    fsum2(ii,jj,ith) = fsum2(ii-1,jj,ith)+fsum2(ii,jj-1,ith)-fsum2(ii-1,jj-1,ith)+pb(ii,jj,ith)**2.
                enddo
            enddo
            paave = sum(abs(pa(NPTSH+1:NPTSH+NPTSH2,NPTSW+1:NPTSW+NPTSW2,ith)**2.))

            do ii=NPTSH2-NOFFH-NDISPby2+1,NPTSH2+NOFFH+NDISPby2+1
                do jj=NPTSW2-NOFFW-NDISPby2+1,NPTSW2+NOFFW+NDISPby2+1
                    e2sum = fsum2(ii+NPTSH-1,jj+NPTSW-1,ith) - fsum2(ii-NPTSH,jj+NPTSW-1,ith) - fsum2(ii+NPTSH-1,jj-NPTSW,ith) + fsum2(ii-NPTSH, jj-NPTSW,ith)
                    esum = fsum(ii+NPTSH-1,jj+NPTSW-1,ith) - fsum(ii-NPTSH,jj+NPTSW-1,ith) - fsum(ii+NPTSH-1, jj-NPTSW,ith) + fsum(ii-NPTSH, jj-NPTSW,ith)

                    c(ii,jj,ith)  = abs(c(ii,jj,ith)/sqrt(paave*(e2sum - esum*esum/prod2)))
                end do
            end do
     
       else  !!<> PSA - Original code
            ! normalize
            c(1:NPTSH4,1:NPTSW4,ith) = abs(c(1:NPTSH4,1:NPTSW4,ith))**2.
       endif


      ! determine rough offset
      cmax = 0.0
      imax = 0
      jmax = 0

      do ii=NPTSH2-NOFFH+1,NPTSH2+NOFFH+1
         do jj=NPTSW2-NOFFW+1,NPTSW2+NOFFW+1
            if (abs(c(ii,jj,ith))>cmax) then
               cmax = abs(c(ii,jj,ith))
               imax = ii
               jmax = jj
            end if
         end do
      end do

      ! rough offset, in local pixels
      rough_az_off = imax-4
      rough_rg_off = jmax-4


      !!!Preprocess the covariance before interpolation
      cpiece(1:NDISP,1:NDISP,ith) = c((imax-NDISPby2):(imax+NDISPby2-1),(jmax-NDISPby2):(jmax+NDISPby2-1),ith)
!!      paave = cmplx(0.0, 0.0)
!!      do ii=1, NDISP
!!        do jj=1,NDISP
!!            paave = paave + cpiece(ii,jj,ith)
!!        end do
!!      end do
!!      paave = paave / prodpiece  !!Mean of the covariance surface
!!      cpiece(1:NDISP,1:NDISP,ith) = cpiece(1:NDISP,1:NDISP,ith) - paave

      ! corr = upsample(c,16)
      call upsample2d_complex(cpiece(1:NDISP,1:NDISP,ith),corr(1:NLARGE,1:NLARGE,ith), &
           CPIECEtrans(1:NDISP,1:NDISP,ith),CPIECEuptrans(1:NLARGE,1:NLARGE,ith), &
           plan_cpiece(ith),plan_cpiecei(ith),NDISP,NDISP,NOVS)

!!      corr(1:NLARGE, 1:NLARGE,ith) = corr(1:NLARGE,1:NLARGE,ith) + paave

      ! determine offset
      cmax = 0.0
      imax = 0
      jmax = 0

      do ii=1,NLARGE
         do jj=1,NLARGE
            if (abs(corr(ii,jj,ith))>cmax) then
               cmax = abs(corr(ii,jj,ith))
               imax = ii
               jmax = jj
            end if
         end do
      end do

!      print *, imax,rough_az_off,jmax,rough_rg_off,cmax 

      ! estimate offsets in pixels
      az_off(irgloc) = gross_az_off  -NPTSH + ((rough_az_off-1.0)*NOVS+ (imax-1.0))/(2.0*NOVS)

      rg_off(irgloc) = gross_rg_off  -NPTSW + ((rough_rg_off-1.0)*NOVS + (jmax-1.0))/(2.0*NOVS)
      snr_off(irgloc) = cmax

   end do rg_loc_loop   ! loop over range locations
   !$omp end parallel do

   ii = 1
   jj = iazloc
   call setLineBand(offset, az_off, jj, ii)

   ii = 2
   jj = iazloc
   call setLineBand(offset, rg_off, jj, ii)

   ii = 1
   jj = iazloc
   call setLineBand(snr, snr_off,jj,ii)

end do az_loc_loop      ! loop over azimuth locations

! ++++++++++++++++++++++++++ END LOOP ++++++++++++++++


! deallocate memory
deallocate( s1, s2)
deallocate(az_loc, rg_loc)
deallocate(az_off, rg_off)
deallocate(rdata)

! destroy FFT plans
call sfftw_destroy_plan(plan_pa)
call sfftw_destroy_plan(plan_pb)
call sfftw_destroy_plan(plan_ctrans)
call sfftw_destroy_plan(plan_a)
call sfftw_destroy_plan(plan_ai)
call sfftw_destroy_plan(plan_b)
call sfftw_destroy_plan(plan_bi)
call sfftw_destroy_plan(plan_cpiece)
call sfftw_destroy_plan(plan_cpiecei)


deallocate(plan_pa, plan_pb, plan_ctrans, plan_a)
deallocate(plan_ai, plan_b, plan_bi)
deallocate(plan_cpiece, plan_cpiecei)


deallocate( a, b, aa, bb)
deallocate(pa, pb, cpa, cpb)
deallocate(c, ctrans, cpiece, corr)
deallocate(Atrans, Auptrans)
deallocate(Btrans, Buptrans)
deallocate(CPIECEtrans, CPIECEuptrans)

if(normalize)then
    deallocate(fsum)
    deallocate(fsum2)
endif


! print runtime statistics
seconds = omp_get_wtime() - seconds
write(*,*) 'Execution time: ',seconds,' seconds'

end subroutine denseoffsets 

! ============== END PROGRAM ========================

!  *********

subroutine fftshift2d(a,m,n)
  ! performs fftshift (2-d) on mxn matrix a

  implicit none

  complex :: a(:,:)
  integer, intent(in) :: m, n

  ! computation variables
  complex, allocatable :: atemp(:,:)
  integer :: p,q

  ! allocate temp memory
  allocate( atemp(m,n) )

  ! copy a to atemp
  atemp = a

  ! fftshift
  p = nint(m/2.0)
  q = nint(n/2.0)

  a(1:p,1:q) = atemp((p+1):m,(q+1):n)
  a(1:p,(q+1):n) = atemp((p+1):m,1:q)
  a((p+1):m,1:q) = atemp(1:p,(q+1):n)
  a((p+1):m,(q+1):n) = atemp(1:p,1:q)

  ! deallocate memory
  deallocate( atemp )

end subroutine fftshift2d

subroutine derampc(c_img, ny, nx)

    implicit none
    complex :: c_img(:,:)
    integer, intent(in) :: ny, nx
    integer :: i,j
    complex :: c_phac, c_phdn
    real :: r_phac, r_phdn

    c_phac = cmplx(0.,0.)
    c_phdn = cmplx(0.,0.)

    do i=1,ny-1
        do j=1,nx
            c_phac = c_phac + c_img(i,j)*conjg(c_img(i+1,j))
        end do
    end do

    do i=1,ny
        do j=1,nx-1
            c_phdn = c_phdn + c_img(i,j)*conjg(c_img(i,j+1))
        end do
    end do

    if(cabs(c_phdn) .eq. 0) then
        r_phdn = 0.0
    else
        r_phdn = atan2(aimag(c_phdn),real(c_phdn))
    endif

    if(cabs(c_phac) .eq. 0) then
        r_phac = 0.0
    else
        r_phac = atan2(aimag(c_phac),real(c_phac))
    endif

    do i=1,ny
        do j=1,nx
            c_img(i,j) = c_img(i,j)*cmplx(cos(r_phac*i+r_phdn*j), sin(r_phac*i+r_phdn*j))
        end do
    end do
end subroutine derampc
