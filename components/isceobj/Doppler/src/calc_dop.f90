  subroutine  calc_dop(imgAccessor)
  use calc_dopState
  use fortranUtils
  implicit none
  ! return Doppler in fraction of PRF
  character*60 file,buf
  integer k,i,eof,pixels
  real*8 prf
  integer*8 imgAccessor
  complex, dimension(:),allocatable :: bi,ai,ab
  character, dimension(:),allocatable :: bytes_line
  real*8 pi

  pi = getPi()

  pixels = (width-header)/2
  allocate(bytes_line(width))
  allocate(ai(pixels),bi(pixels),ab(pixels))

  ! read the first line
  call initSequentialAccessor(imgAccessor,first_line)
  call getLineSequential(imgAccessor,bytes_line,eof)

  ai = (/(cmplx(ichar(bytes_line(header+2*i+1))-Ioffset,&
       ichar(bytes_line(header+2*(i+1)))-Qoffset),i=0,pixels-1)/)
  ab = cmplx(0.,0.)
  
  do i = first_line+1,last_line    
    call getLineSequential(imgAccessor,bytes_line,eof)
     bi = (/(cmplx(ichar(bytes_line(header+2*k+1))-Ioffset,&
          ichar(bytes_line(header+2*(k+1)))-Qoffset),k=0,pixels-1)/)
     ab = ab + conjg(ai)*bi
     ai = bi 
              
  enddo

  fd = sum(atan2(imag(ab),real(ab))/(2.d0*pi))/dble(pixels)
 
  ! write pixel dependent doppler to file
  do i = 1,pixels
     rngDoppler(i) = atan2(imag(ab(i)),real(ab(i)))/(2d0*pi)
  enddo
  
  ! close files
  deallocate(bytes_line,ai,bi,ab)

end 
