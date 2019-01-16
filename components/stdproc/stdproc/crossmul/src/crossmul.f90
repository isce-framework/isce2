!c  crossmul - cross multiply two files, one conjugated, form int and amp file
      subroutine crossmul(cst, slcAccessor1, slcAccessor2, ifgAccessor, ampAccessor) BIND(C,name='crossmul_f')

      use, intrinsic :: iso_c_binding
      use crossmulState

      implicit none

      include 'omp_lib.h'
      type(crossmulType):: cst
      integer (C_INT64_T) slcAccessor1
      integer (C_INT64_T) slcAccessor2
      integer (C_INT64_T) ifgAccessor
      integer (C_INT64_T) ampAccessor
      complex*8, allocatable:: in1(:,:),in2(:,:)
      complex*8, allocatable:: igram(:,:,:),amp(:,:,:)
      complex*8, allocatable:: up1(:,:,:),up2(:,:,:)
      complex*8, allocatable:: inline1(:,:),inline2(:,:)
      complex*8, allocatable:: igramacc(:,:),ampacc(:,:)
      complex*8, allocatable:: igramtemp(:,:),amptemp(:,:)
      integer n, i, j, k, nnn, line
      integer nblocks, iblk, nl, ith


      !!!!!!For now, making local copies
      !!!!!!Could access anywhere in code using cst%
      integer :: na, nd, looksac, looksdn, blocksize
      double precision:: scale

      na = cst%na
      nd = cst%nd
      looksac = cst%looksac
      looksdn = cst%looksdn
      blocksize = cst%blocksize
      scale = cst%scale

      !$omp parallel
      n=omp_get_num_threads()
      !$omp end parallel
      print *, 'Max threads used: ', n


!c  get ffts lengths for upsampling
      do i=1,16
         nnn=2**i
         if(nnn.ge.na)go to 11
      end do
11    print *,'FFT length: ',nnn

      call cfft1d_jpl(nnn, igramacc, 0)  !c Initialize FFT plan
      call cfft1d_jpl(2*nnn, igramacc, 0)

      !c Number of blocks needed
      nblocks = CEILING(nd/(1.0*blocksize*looksdn))
      print *, 'Overall:', nd, blocksize*looksdn, nblocks
      allocate(in1(na,looksdn*blocksize), in2(na,looksdn*blocksize))
      allocate(igramtemp(na/looksac,blocksize), amptemp(na/looksac,blocksize))



      !c  allocate the local arrays
      allocate (igram(na*2,looksdn,n),amp(na*2,looksdn,n))
      allocate (igramacc(na,n),ampacc(na,n))
      allocate (up1(nnn*2,looksdn,n),up2(nnn*2,looksdn,n),inline1(nnn,n),inline2(nnn,n))

      do iblk=1, nblocks
          k = (iblk-1)*blocksize*looksdn+1
          in1 = cmplx(0., 0.)
          in2 = cmplx(0., 0.)
          igramtemp = cmplx(0., 0.)
          amptemp = cmplx(0., 0.)

          if (iblk.ne.nblocks) then
              nl = looksdn*blocksize
          else
              nl = (nd - (nblocks-1)*blocksize*looksdn)
          endif

!c          print *, 'Block: ', iblk, k, nl

          do j=1, nl
            call getLineSequential(slcAccessor1,in1(:,j),k)
          end do


          if (slcAccessor1.ne.slcAccessor2) then
              do j=1, nl
                call getLineSequential(slcAccessor2,in2(:,j),k)
              end do
          else
            in2 = in1
          endif
          in1 = in1*scale
          in2 = in2*scale

          

          !$omp parallel do private(j,k,i,line,ith) &
          !$omp shared(in1,in2,igramtemp,amptemp,nl) &
          !$omp shared(looksdn,looksac,scale,na,nnn, nd)&
          !$omp shared(up1,up2,inline1,inline2,igram,amp)&
          !$omp shared(igramacc,ampacc,n)
          do line=1,nl/looksdn

                ! get thread number
                ith = omp_get_thread_num() + 1          

                up1(:,:,ith)=cmplx(0.,0.)  ! upsample file 1
                do i=1,looksdn
                    inline1(1:na,ith)=in1(:,i+(line-1)*looksdn)
                    inline1(na+1:nnn, ith)=cmplx(0.,0.)
                    call cfft1d_jpl(nnn, inline1(1,ith), -1)


                    up1(1:nnn/2,i,ith)=inline1(1:nnn/2,ith)
                    up1(2*nnn-nnn/2+1:2*nnn,i,ith)=inline1(nnn/2+1:nnn,ith)
                    call cfft1d_jpl(2*nnn, up1(1,i,ith), 1)
                end do
                up1(:,:,ith)=up1(:,:,ith)/nnn

                up2(:,:,ith)=cmplx(0.,0.)  ! upsample file 2
                do i=1,looksdn
                    inline2(1:na,ith)=in2(:,i+(line-1)*looksdn)
                    inline2(na+1:nnn,ith)=cmplx(0.,0.)
                    call cfft1d_jpl(nnn, inline2(1,ith), -1)

                    up2(1:nnn/2,i,ith)=inline2(1:nnn/2,ith)
                    up2(2*nnn-nnn/2+1:2*nnn,i,ith)=inline2(nnn/2+1:nnn,ith)
                    call cfft1d_jpl(2*nnn, up2(1,i,ith), 1)
               end do
               up2(:,:,ith)=up2(:,:,ith)/nnn

               igram(1:na*2,:,ith)=up1(1:na*2,:,ith)*conjg(up2(1:na*2,:,ith))
               amp(1:na*2,:,ith)=cmplx(cabs(up1(1:na*2,:,ith))**2,cabs(up2(1:na*2,:,ith))**2)

               !c  reclaim the extra two across looks first
               do j=1,na
                   igram(j,:,ith) = igram(j*2-1,:,ith)+igram(j*2,:,ith)
                   amp(j,:,ith) = amp(j*2-1,:,ith)+amp(j*2,:,ith)
               end do

               !c     looks down
               igramacc(:,ith)=sum(igram(1:na,:,ith),2)
               ampacc(:, ith)=sum(amp(1:na,:,ith),2)

               !c     looks across
               do j=0,na/looksac-1
                  do k=1,looksac
                      igramtemp(j+1,line)=igramtemp(j+1,line)+igramacc(j*looksac+k,ith)
                      amptemp(j+1, line)=amptemp(j+1,line)+ampacc(j*looksac+k,ith)
                  end do
                  amptemp(j+1, line)=cmplx(sqrt(real(amptemp(j+1, line))),sqrt(aimag(amptemp(j+1, line))))
               end do


         end do
         !$omp end parallel do

         do line=1, nl/looksdn
            call setLineSequential(ifgAccessor,igramtemp(1,line))
            call setLineSequential(ampAccessor,amptemp(1,line))
         end do

      enddo
      deallocate (up1,up2,igramacc,ampacc,inline1,inline2,igram,amp)
      deallocate(in1, in2, igramtemp, amptemp)
      call cfft1d_jpl(nnn, igramacc, 2)  !c Uninitialize FFT plan
      call cfft1d_jpl(2*nnn, igramacc, 2)

      end


