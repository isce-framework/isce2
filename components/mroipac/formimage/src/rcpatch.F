      subroutine rcpatch(ptImageRaw,dataLineGet,nnn,nlinesaz,trans,unpacki,unpackq,ref,irec,ifrst,nbytes,ngood,nr_fftf,nr_fftr,iq,iflip,agc,dwp,nagc,ndwp)

      
      implicit none
      integer*1 :: dataLineGet(nbytes)
      integer*4 nnn,irec,ifrst,nbytes,ngood,fdsc,nlinesaz,i,j,k, iq,iflip
      integer*8 ptImageRaw
      integer*4 offsetGet
      integer*4 nr_fftf, nr_fftr
      real*4    unpacki(256),unpackq(256)

      complex*8 trans(nnn,nlinesaz),ref(nr_fftf),tmp(nr_fftf)
      integer*4 nagc, ndwp, dwpval, nrec
      real*4    dwp(2,ndwp),agc(2,nagc), agcval
      integer*4 line,ii, ierr

      agcval = 1.
      dwpval = 0
      if(iq.eq.0) then
         write(6,*) 'Offset Video: Range starting record, pixel: ',irec,ifrst
         offsetGet = 0
         do j = 1, nnn
            line=j-1+irec-1
            if(mod(line,1024).eq.1)write(6,*) 'Line ',line
            if(j-1+irec .lt. 1) goto 777

            offsetGet = j-1+irec
            call getLine(ptImageRaw,dataLineGet,offsetGet)
!c            read(fdsc,rec=j-1+irec,iostat=ierr) (inbuf(k),k=1,nbytes)
!c            if(ierr .ne. 0) goto 777
            if(offsetGet .lt. 0) goto 777

!c Handle gain change and dwp change of raw data
!c
            nrec = j-1+irec
            agcval = 1.
            do i = nagc, 1, -1
               if (nrec .ge. agc(1,i)) then
                 agcval = 10. ** (agc(2,i)/20.)
                 goto 180
               endif
            enddo

180         dwpval = 0
            do i = ndwp, 1, -1
               if (nrec .ge. dwp(1,i)) then
                  dwpval = dwp(2,i)
                  goto 181
               endif
            enddo

181         do i=max(1,dwpval+1),ngood-ifrst+max(0,dwpval)
               tmp(i)=cmplx(unpacki(1+(iand(int(dataLineGet(offsetGet + i+ifrst-dwpval)),255))),0.)
!c               tmp(i)=cmplx(unpacki(1+(iand(int(inbuf(i+ifrst-dwpval)),255))),0.)
!c              int() above is used to promote inbuf elements 
!c              to be a default integer
!c              so they are the same type/kind as the constant 255.
!c              Some compilers enforce that both parameters to iand()
!c              be the same type/kind.
            end do
            do i=ngood-ifrst+1+max(0,dwpval),nr_fftf
               tmp(i)=cmplx(0.,0.)
            end do
            do i=1,max(0,dwpval)
               tmp(i)=cmplx(0.,0.)
            enddo
            goto 778

!c     if there is a read error, assume user is requesting a line 
!c     from either before or after the file extent.  Fill with zeros.

 777        do ii=1,nr_fftf
               tmp(ii)=cmplx(0.,0.)
            enddo

 778        call cfft1d_jpl(nr_fftf,tmp,-1)
         
!c  baseband the resulting spectrum since input is offset video

            do i=1,nr_fftf/4
               tmp(i)=tmp(i+3*nr_fftf/4)*ref(i+3*nr_fftf/4)*agcval
            end do
            do i=1,nr_fftf/4
               tmp(i+nr_fftf/4)=tmp(i+nr_fftf/2)*ref(i+nr_fftf/2)*agcval
            end do

            call cfft1d_jpl(nr_fftr,tmp,1)

            do i=1,nlinesaz
               trans(j,i)=tmp(i)
            end do
         end do
      else

         write(6,*) 'I/Q: Range starting record, pixel: ',irec,ifrst
         offsetGet = 0
         do j = 1, nnn
            line=j-1+irec-1
            if(mod(line,1024).eq.1)write(6,*) 'Line ',line
            if(j-1+irec .lt. 1) goto 779

            offsetGet = j-1+irec
            call getLine(ptImageRaw,dataLineGet,offsetGet)

!c            read(fdsc,rec=j-1+irec,iostat=ierr) (inbuf(k),k=1,nbytes)
!c            if(ierr .ne. 0) goto 779
            if(offsetGet .lt. 0) goto 779
!c
!c Handle gain change and dwp change of raw data
!c
            nrec = j-1+irec
            agcval = 1.
            do i = nagc, 1, -1
               if (nrec .ge. agc(1,i)) then
                  agcval = 10. ** (agc(2,i)/20.)
                  goto 280
               endif
            enddo

280         dwpval = 0
            do i = ndwp, 1, -1
               if (nrec .ge. dwp(1,i)) then
                  dwpval = dwp(2,i)
                  goto 281
               endif
            enddo

281         do i=max(1,dwpval+1),ngood/2-ifrst+max(0,dwpval)
               tmp(i)=
     1              cmplx(unpacki(1+(iand(255,int(dataLineGet((i+ifrst-dwpval)*2-1))))),
     $                    unpackq(1+(iand(255,int(dataLineGet((i+ifrst-dwpval)*2))))))
!c     1              cmplx(unpacki(1+(iand(255,int(inbuf((i+ifrst-dwpval)*2-1))))),
!c     $                    unpackq(1+(iand(255,int(inbuf((i+ifrst-dwpval)*2))))))


            end do

            do i=ngood/2-ifrst+1+max(0,dwpval),nr_fftf
               tmp(i)=cmplx(0.,0.)
            end do
            do i=1,max(0,dwpval)
               tmp(i)=cmplx(0.,0.)
            enddo
            goto 780

!c     if there is a read error, assume user is requesting a line 
!c     from either before or after the file extent.  Fill with zeros.

 779        do ii=1,nr_fftf
               tmp(ii)=cmplx(0.,0.)
            enddo

 780        if(iflip .eq. 1) then
               do i = 1 , nr_fftf
                  tmp(i) = cmplx(aimag(tmp(i)),real(tmp(i)))
               end do
            end if
            call cfft1d_jpl(nr_fftf,tmp,-1)
            
            do i=1,nr_fftf
               tmp(i)=tmp(i)*ref(i)*agcval
            end do
            
            call cfft1d_jpl(nr_fftf,tmp,1)
            
            do i=1,nlinesaz
               trans(j,i)=tmp(i)
            end do
         end do
      end if
         
      return
      end subroutine rcpatch
