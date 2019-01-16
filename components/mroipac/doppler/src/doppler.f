      subroutine doppler(rawAccessor)

      use dopplerState
      implicit none
      integer i_rs,i_as
      parameter(i_rs=50000,i_as=512)

      integer*8 rawAccessor
C      character*80 string,a_file
C      integer iargc,i,j,i_arg,i_samples,i_strtline,i_nlines,ir,i_pcnt,i_lpw
      integer ir,i_pcnt,i_lpw,i,i_mod,i_mod1,j
      complex c_image(i_rs,0:1),dbuf(i_rs),c_dbuf(i_rs),line(i_rs)
C      real r_fd(i_rs),
      real r_fest,r_fdraw(i_rs),r_festraw,wgth
      real*8, parameter :: pi = 4.d0*atan(1.d0)

C      write(6,*) ' '
C      write(6,*) '   <<    Doppler Estimate from Complex Data    >>'
C      write(6,*) ' '

C      i_arg = iargc()
C      if(i_arg .lt. 4)then
C         write(6,*) 'Usage: doppler file samples start_line numlines [lpfw]'
C         stop
C      endif

      i_lpw = 1

C      call getarg(1,a_file)
C      call getarg(2,string)
C      read(string,*) i_samples
C      call getarg(3,string)
C      read(string,*) i_strtline
C      call getarg(4,string)
C      read(string,*) i_nlines
C      if(iargc() .gt. 4)then
C         call getarg(5,string)
C         read(string,*) i_lpw
C      endif

C      open(12,file=a_file,access='direct',form='unformatted',recl=8*i_samples)

      write(6,*) 'Reading Data...'
      write(6,*) ' '
      write(6,*) ' Doppler estimation as a function of range :'

      wgth = 1.0

      do i = 1,i_samples
         dbuf(i) = (0.0,0.0)
      enddo

      do i=i_strtline,i_strtline+i_nlines-1
         i_mod = mod(i,2)
C         read(12,rec=i) (c_image(j,i_mod),j=1,i_samples)
         call getLine(rawAccessor,line,i)
         do j=1,i_samples
            c_image(j,i_mod) = line(j)
         enddo
         if(i .gt. i_strtline)then
            i_mod1 = mod(i-1,2)
            do ir = 1,i_samples
               dbuf(ir) = dbuf(ir) + wgth*c_image(ir,i_mod)*conjg(c_image(ir,i_mod1))
            enddo               ! ir-loop
         endif
      enddo

c     take range looks 
      
      do ir=1,i_samples

         c_dbuf(ir) = cmplx(0.,0.)
         do i=1,i_lpw
            c_dbuf(ir) = c_dbuf(ir) + dbuf(min(ir+i-1,i_samples))
         enddo
         
      enddo
      
c Doppler ambiguity resolution 

      i_pcnt = 0
      do ir=1,i_samples
         if(cabs(c_dbuf(ir)) .ne. 0)then
            r_festraw = atan2(aimag(c_dbuf(ir)),real(c_dbuf(ir)))/(2.0*pi)
            r_fest  = atan2(aimag(c_dbuf(ir)),real(c_dbuf(ir)))/(2.0*pi) + 1.0*i_pcnt
         else
            r_fest = 0.0
         endif
         if(ir .ne. 1)then
            if(abs(r_fest-r_fd(ir-1)) .gt. .501)then
               i_pcnt = i_pcnt + nint(sign(1.0D0,r_fd(ir-1)-r_fest))
               r_fest = r_fest + sign(1.0D0,r_fd(ir-1)-r_fest)
            endif
         endif
         r_fd(ir)= r_fest
         r_fdraw(ir) = r_festraw
      end do

C      write(6,*) ' '
C      write(6,*) 'Writing file dop.out'

C      open(13,file='dop.out',status='unknown')

C      do i=1,i_samples
C         write(13,'(x,i10,x,f15.10,x,f15.10)') i,r_fd(i),r_fdraw(i)
C         write(13,'(x,i10,x,f15.10,x,f15.10)') i,r_fd(i)
C      enddo
      
      end
      








