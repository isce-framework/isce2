      subroutine rect(infile,outfile,ndac,nddn,nrac,nrdn,a,b,c,d,e,f,filetype,intstyle)
c     
c*****************************************************************************
c**   
c**   FILE NAME: rect.f
c**   
c**   DATE WRITTEN: 27-Nov-98
c**   
c**   PROGRAMMER: P.A.Rosen
c**   
c**   FUNCTIONAL DESCRIPTION:  This program adjusts an image
c**   by affine transformation and interpolation
c**   
c**   UPDATE LOG: Cunren Liang, 24-APR-2015
c**               updated to support file types: real, double
c**               and support the case that input and output file sizes are different
C**
c**               Cunren LIANG, 03-JUN-2015
c**               1. there is 1 pixel offset in the output, which is corrected.
c**              
c**   
c*****************************************************************************
      
      
      implicit none

c      integer CMAX, RMAX
c      parameter (CMAX = 7000, RMAX = 7200)
c      real*4  rvs(0:2*CMAX-1,0:RMAX-1)
c      complex*8  carr(0:20000)
c      real*4  arr(0:40000)

c     input of resampling     
      REAL*4, DIMENSION(:,:), ALLOCATABLE :: rvs

c     variables for reading data
c     For byte format     *****GP 01-05******
      INTEGER*1, DIMENSION(:), ALLOCATABLE :: barr
      real*4, DIMENSION(:), ALLOCATABLE :: rarr
      real*8, DIMENSION(:), ALLOCATABLE :: darr
      COMPLEX*8, DIMENSION(:), ALLOCATABLE :: carr

c     output of resampling     
      REAL*4, DIMENSION(:), ALLOCATABLE :: arr

      real*4  pt1(3),pt2(3),pt3(3),pt4(3)
      real*8  colval, rowval, ocolval, orowval
      real*8  ifrac, jfrac
      real*8  a,b,c,d,e,f
      real*4  interp

      integer oi, oj, i, j, k, ift, iis
      integer iargc, ndac, nddn, nrac, nrdn, ierr
      integer nac
      integer psize

      character*180 fname, infile, outfile, intstyle, filetype

      integer rdflen
      character*255 rdfval
      character*255 rdfcullsp,rdfdata
      character*255 a_rdtmp

      save rvs

c     if(iargc() .eq. 0) then
c        write(*,*) 'usage: rect rect.rdf'
c        stop
c     end if

c     call getarg(1,fname)

c     call rdf_init('ERRFILE=SCREEN')
c     write(6,'(a)') 'Reading command file data...'
c     call rdf_read(fname)

c     a_rdtmp = rdfval('Input Image File Name','-')
c     read(unit=a_rdtmp,fmt='(a)') infile
c     a_rdtmp = rdfval('Output Image File Name','-')
c     read(unit=a_rdtmp,fmt='(a)') outfile
c     a_rdtmp = rdfval('Input Dimensions','-')
c     read(unit=a_rdtmp,fmt=*) ndac, nddn
c     a_rdtmp = rdfval('Output Dimensions','-')
c     read(unit=a_rdtmp,fmt=*) nrac, nrdn
c     a_rdtmp = rdfval('Affine Matrix Row 1','-')
c     read(unit=a_rdtmp,fmt=*) a, b
c     a_rdtmp = rdfval('Affine Matrix Row 2','-')
c     read(unit=a_rdtmp,fmt=*) c, d
c     a_rdtmp = rdfval('Affine Offset Vector','-')
c     read(unit=a_rdtmp,fmt=*) e, f
c     a_rdtmp = rdfval('File Type','-')
c     read(unit=a_rdtmp,fmt='(a)') filetype
c     a_rdtmp = rdfval('Interpolation Method','-')
c     read(unit=a_rdtmp,fmt='(a)') intstyle

c      if(ndac .gt. CMAX) stop 'Increase column array dimension in rect' 
c      if(nddn .gt. RMAX) stop 'Increase row array dimension in rect'

      ift = 0
      psize = 8
      if(index(filetype,'RMG') .ne. 0)then
         ift = 1
         psize = 8
         write (*,*)  'Assuming RMG file type '
c      For byte format  *****GP 01-05******
      elseif(index(filetype,'BYTE') .ne. 0)then
         ift = 2
         psize = 1
         write (*,*)  'Assuming byte file type '
      elseif(index(filetype,'REAL') .ne. 0)then
         ift = 3
         psize = 4
         write (*,*)  'Assuming real*4 file type '
      elseif(index(filetype,'DOUBLE') .ne. 0)then
         ift = 4
         psize = 8
         write (*,*)  'Assuming double (real*8) file type '
      else
         write (*,*)  'Assuming complex file type '
      endif

      iis = 0
      if(index(intstyle,'Bilinear') .ne. 0)then
         iis = 1
         write (*,*)  'Assuming Bilinear Interpolation '
      elseif(index(intstyle,'Sinc') .ne. 0)then
         iis = 2
         write (*,*)  'Assuming Sinc Interpolation '
      else
         write (*,*)  'Assuming Nearest Neighbor '
      end if

c     input of resampling
      if(ift .le. 1) then
         ALLOCATE( rvs(0:2*ndac-1,0:nddn-1) )
      else
         ALLOCATE( rvs(0:ndac-1,0:nddn-1) )
      end if
      write(*,*) 'Allocated a map of dimension ',ndac,nddn

c     variable for reading data
      
      if(ndac .gt. nrac) then
         nac = ndac
      else
         nac = nrac
      end if
      ALLOCATE( carr(0:2*nac-1) )
      write(*,*) 'Allocated an array of dimension ',nac
c     there is no need to allocate an array for rmg type      
c     For byte format     *****GP 01-05******
      ALLOCATE( barr(0:nac-1) )
      write(*,*) 'Allocated an array of dimension ',nac
      ALLOCATE( rarr(0:nac-1) )
      write(*,*) 'Allocated an array of dimension ',nac
      ALLOCATE( darr(0:nac-1) )
      write(*,*) 'Allocated an array of dimension ',nac


c     output of resampling
      if(ift .le. 1) then
         ALLOCATE( arr(0:2*nrac-1) )
      else
         ALLOCATE( arr(0:nrac-1) )
      end if
      write(*,*) 'Allocated array of dimension ',nrac

      write (*,*)  'opening files ...'

c     open files
         open(11,file=infile,form='unformatted',
     .     access='direct',recl=psize*ndac,status='old') 
         open(12,file=outfile,form='unformatted',
     .     access='direct',recl=psize*nrac,status='unknown')

c    forcing NN interpolation for byte format
c      if(ift .eq. 2) then    
c          iis = 0
c      end if


c read in the data

      write (*,*)  'reading data ...'

      if(ift .eq. 0) then
         do j = 0 , nddn-1
            if(mod(j,4096) .eq. 0) write (*,*)  j
            read(11,rec=j+1,iostat=ierr) (carr(k),k=0,ndac-1)
            if(ierr .ne. 0) goto 999
            do k = 0 , ndac -1
               rvs(k,j) = real(carr(k))
               rvs(k+ndac,j) = aimag(carr(k))
            end do
         end do
      elseif(ift .eq. 1) then
         do j = 0 , nddn-1
            if(mod(j,4096) .eq. 0) write (*,*)  j
            read(11,rec=j+1,iostat=ierr) (rvs(k,j),k=0,2*ndac-1)
            if(ierr .ne. 0) goto 999
         end do
      elseif(ift .eq. 2) then
         do j = 0 , nddn-1
            if(mod(j,4096) .eq. 0) write (*,*)  j
            read(11,rec=j+1,iostat=ierr) (barr(k),k=0,ndac-1)
            if(ierr .ne. 0) goto 999
            do k = 0 , ndac -1
               rvs(k,j) = barr(k)
            end do
         end do
      elseif(ift .eq. 3) then
         do j = 0 , nddn-1
            if(mod(j,4096) .eq. 0) write (*,*)  j
            read(11,rec=j+1,iostat=ierr) (rarr(k),k=0,ndac-1)
            if(ierr .ne. 0) goto 999
            do k = 0 , ndac -1
               rvs(k,j) = rarr(k)
            end do
         end do
      else
         do j = 0 , nddn-1
            if(mod(j,4096) .eq. 0) write (*,*)  j
            read(11,rec=j+1,iostat=ierr) (darr(k),k=0,ndac-1)
            if(ierr .ne. 0) goto 999
            do k = 0 , ndac -1
               rvs(k,j) = darr(k)
            end do
         end do
      end if

 999  write (*,*)  'finished read ',j,' now interpolating ...'

c do the interpolation

      do j = 0 , nrdn-1
         if(mod(j,4096) .eq. 0) write (*,*)  j
         rowval = dble(j)

         if(iis .eq. 0) then    !  nearest neighbor

            do i = 0 , nrac-1
               colval = dble(i)
               ocolval =  a * colval + b * rowval + e
               orowval =  c * colval + d * rowval + f
               oi = nint(ocolval)
               oj = nint(orowval)
               if(.not.(oi .lt. 0 .or. oi .ge. ndac .or. oj .lt. 0 .or
     $              . oj .ge. nddn)) then
                  arr(i) = rvs(oi,oj)
                  if(ift .le. 1) then
                      arr(i+nrac) = rvs(oi+ndac,oj)
                  end if
               else
                  arr(i) = 0.
                  if(ift .le. 1) then
                     arr(i+nrac) = 0.
                  end if
               end if
            end do
            
         elseif(iis. eq. 1) then !          bilinear interpolation

            do i = 0 , nrac-1
               colval = dble(i)
               ocolval =  a * colval + b * rowval + e
               orowval =  c * colval + d * rowval + f
               oi = nint(ocolval)
               oj = nint(orowval)
               ifrac = (ocolval - oi)
               jfrac = (orowval - oj)
               if(ifrac .lt. 0.d0) then
                  oi = oi - 1
                  ifrac = (ocolval - oi)
               end if
               if(jfrac .lt. 0.d0) then
                  oj = oj - 1
                  jfrac = (orowval - oj)
               end if
               if(.not.(oi .lt. 0 .or. oi .ge. ndac-1 .or. oj .lt. 0 .or
     $              . oj .ge. nddn-1)) then
                  pt1(1) = 0.
                  pt1(2) = 0.
                  pt1(3) = rvs(oi,oj)
                  pt2(1) = 1.
                  pt2(2) = 0.
                  pt2(3) = rvs(oi+1,oj)
                  pt3(1) = 0.
                  pt3(2) = 1.
                  pt3(3) = rvs(oi,oj+1)
                  pt4(1) = 1.
                  pt4(2) = 1.
                  pt4(3) = rvs(oi+1,oj+1)
                  call bilinear(pt1,pt2,pt3,pt4,sngl(ifrac),sngl(jfrac),arr(i))
                  if(ift .le. 1) then
                     pt1(1) = 0.
                     pt1(2) = 0.
                     pt1(3) = rvs(oi+ndac,oj)
                     pt2(1) = 1.
                     pt2(2) = 0.
                     pt2(3) = rvs(oi+1+ndac,oj)
                     pt3(1) = 0.
                     pt3(2) = 1.
                     pt3(3) = rvs(oi+ndac,oj+1)
                     pt4(1) = 1.
                     pt4(2) = 1.
                     pt4(3) = rvs(oi+1+ndac,oj+1)
                     call bilinear(pt1,pt2,pt3,pt4,sngl(ifrac),sngl(jfrac),arr(i+nrac))
                  end if
               else
                  arr(i) = 0.
                  if(ift .le. 1) then
                     arr(i+nrac) = 0.
                  end if
               end if
            end do
            

         elseif(iis. eq. 2) then !          sinc interpolation

            do i = 0 , nrac-1
               colval = dble(i)
               ocolval =  a * colval + b * rowval + e
               orowval =  c * colval + d * rowval + f
               oi = nint(ocolval)
               oj = nint(orowval)
               ifrac = (ocolval - oi)
               jfrac = (orowval - oj)
               if(ifrac .lt. 0.d0) then
                  oi = oi - 1
                  ifrac = (ocolval - oi)
               end if
               if(jfrac .lt. 0.d0) then
                  oj = oj - 1
                  jfrac = (orowval - oj)
               end if
               
!               if(.not.(oi .lt. 4 .or. oi .ge. ndac-3 .or. oj .lt. 4 .or
!     $              . oj .ge. nddn-3)) then
! I changed the upper sentence, as I have debug the array problem in interp.f, Cunren Liang, 03-JUN-2015
               if(.not.(oi .lt. 4 .or. oi .ge. ndac-4 .or. oj .lt. 4 .or
     $              . oj .ge. nddn-4)) then
                  arr(i)      = interp(oi, oj, ifrac, jfrac, rvs, ndac, 0)
                  if(ift .le. 1) then
                     arr(i+nrac) = interp(oi, oj, ifrac, jfrac, rvs, ndac, ndac)
                  end if
               else
                  arr(i) = 0.
                  if(ift .le. 1) then
                     arr(i+nrac) = 0.
                  end if
               end if
            end do

         end if

         if(ift .eq. 0) then
            do k = 0 , nrac -1
               carr(k) = cmplx(arr(k),arr(k+nrac))
            end do
            write(12,rec=j+1) (carr(k),k=0,nrac-1)
         elseif(ift .eq. 1) then
            write(12,rec=j+1) (arr(k),k=0,2*nrac-1)
         elseif(ift .eq. 2) then
            do k = 0 , nrac -1
               barr(k) = arr(k)
            end do
            write(12,rec=j+1) (barr(k),k=0,nrac-1)
         elseif(ift .eq. 3) then
            do k = 0 , nrac -1
               rarr(k) = arr(k)
            end do
            write(12,rec=j+1) (rarr(k),k=0,nrac-1)
         else
            do k = 0 , nrac -1
               darr(k) = arr(k)
            end do
            write(12,rec=j+1) (darr(k),k=0,nrac-1)
         end if
         
      end do

      DEALLOCATE(rvs)
      DEALLOCATE(carr)
      DEALLOCATE(barr)
      DEALLOCATE(rarr)
      DEALLOCATE(darr)
      DEALLOCATE(arr)

      close(unit=11)
      close(unit=12)
      end
