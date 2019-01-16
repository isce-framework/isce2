      subroutine derampc(c_img,i_dim)

      implicit none
      integer i_dim,i,j
      complex c_img(i_dim,i_dim),c_phdn,c_phac
      real r_phac,r_phdn

      c_phdn = cmplx(0.,0.)
      c_phac = cmplx(0.,0.)

      do i=1,i_dim-1
         do j=1,i_dim
            c_phac = c_phac + c_img(i,j)*conjg(c_img(i+1,j))
         enddo
      enddo

      do i=1,i_dim
         do j=1,i_dim-1
            c_phdn = c_phdn + c_img(i,j)*conjg(c_img(i,j+1))
         enddo
      enddo

      if(cabs(c_phdn) .eq. 0)then
         r_phdn = 0.0
      else
         r_phdn = atan2(aimag(c_phdn),real(c_phdn))
      endif

      if(cabs(c_phac) .eq. 0)then
         r_phac = 0.0
      else
         r_phac = atan2(aimag(c_phac),real(c_phac))
      endif
c      write(6,*) 'Phase across, down = ',r_phac,r_phdn
      
      do i=1,i_dim
         do j=1,i_dim
            c_img(i,j) = c_img(i,j)*cmplx(cos(r_phac*i+r_phdn*j),
     &           sin(r_phac*i+r_phdn*j))
         enddo
      enddo

      end
            
