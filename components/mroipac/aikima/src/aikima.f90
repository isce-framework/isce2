!c****************************************************************
!c      

      subroutine aikima(inAcc, outAcc, inband, outband)
      
!c****************************************************************
!c**   
!c**   FILE NAME: file_resample
!c**   
!c**   DATE WRITTEN:9/94 
!c**   
!c**   PROGRAMMER: PAR
!c**   
!c**   FUNCTIONAL DESCRIPTION: This will take a file in mag then hgt format,
!c**   and interpolate it to a uniform grid, using the Aikima 
!c**   interpolation algorithm. assumes amp=0 points are invalid
!c**   
!c**   ROUTINES CALLED:idsfft (SH)
!c**   
!c**   NOTES: none
!c**   
!c**   UPDATE LOG:
!c**   
!c*****************************************************************
      
      use aikimaState

      implicit none

      !c INPUT VARIABLES:
      integer*8   inAcc, outAcc
      integer     inband, outband
      
!c     OUTPUT VARIABLES:

      real*4, DIMENSION(:), ALLOCATABLE, save :: r_surfh   !value of surface points heights 
      real*4, DIMENSION(:), ALLOCATABLE, save :: r_xout    !values of x coordinates (c2 pixels)
      real*4, DIMENSION(:), ALLOCATABLE, save :: r_yout    !values of y coordinates (c1 pixels)

      integer i_nxout           !number or x values
      integer i_nyout           !number of y values
      
!c     LOCAL VARIABLES:
      integer*4, DIMENSION(:), ALLOCATABLE, save ::  i_iwk
      real*4   , DIMENSION(:), ALLOCATABLE, save ::  r_wk

      real*4,    DIMENSION(:,:), ALLOCATABLE, save :: r_inarr, r_outarr
      integer*4, DIMENSION(:), ALLOCATABLE, save ::  i_blockx,i_blocky
      real*4   , DIMENSION(:), ALLOCATABLE, save ::  r_bvx,r_bvy
      real*4   , DIMENSION(:), ALLOCATABLE, save ::  r_bvz,xt, yt
      real    r_blockarea,r_blockareac
      real    ach, pch, dbth, db
      integer i,j,i_md,i_xe,i_ye
      integer i_bcnt, i_nxoutb,i_nyoutb,l,m,k
      integer ii,ll, kk, nt, i_yfrom, i_yto, i_xfrom, i_xto
      integer i_iwksize,i_rwksize
      integer linenum

      data dbth /1.e-15/
      
      write (*,*)  ' '
      write (*,*)  'xmax,xmin = ',i_xmax,i_xmin
      write (*,*)  'ymax,ymin = ',i_ymax,i_ymin
      write (*,*)  ' '
      
!c     determine the number of interpolation points
      
      i_nxout = i_xmax-i_xmin + 1 
      i_nyout = i_ymax-i_ymin + 1 
      
      write (*,*)  'i_nxout,i_nyout = ',i_nxout,i_nyout
      
!c     determine the block boundaries and the number of blocks
      
      i_xe = int((i_nxout)/i_skip) + 1
      i_ye = int((i_nyout)/i_skip) + 1
      if(mod(i_nxout,i_skip) .eq. 0)  i_xe = i_xe - 1
      if(mod(i_nyout,i_skip) .eq. 0)  i_ye = i_ye - 1
      
      i_iwksize = MAX(31,27+i_ncp)*(i_skip+2*i_padn)**2+(i_skip+2*i_padn)**2
      i_rwksize = 5*(i_skip+2*i_padn)**2

      ALLOCATE( r_xout (i_skip) )
      ALLOCATE( r_yout (i_skip) )
      ALLOCATE( xt(4*i_skip+1))
      ALLOCATE( yt(4*i_skip+1))

      ALLOCATE(r_inarr(nac,-i_padn-1:i_skip+i_padn+1))
      ALLOCATE(r_outarr(nac,-i_padn-1:i_skip+i_padn+1))

      ALLOCATE( i_iwk(i_iwksize))
      ALLOCATE( r_wk(i_rwksize))
      ALLOCATE (i_blockx(i_xe+1))
      ALLOCATE (i_blocky(i_ye+1))
      ALLOCATE( r_surfh((i_skip+2*i_padn)*(i_skip+2*i_padn)))
      ALLOCATE( r_bvx((i_skip+2*i_padn)*(i_skip+2*i_padn)))
      ALLOCATE( r_bvy((i_skip+2*i_padn)*(i_skip+2*i_padn)))
      ALLOCATE( r_bvz((i_skip+2*i_padn)*(i_skip+2*i_padn)))

!c     determine the block boundaries and the number of blocks
      
      i_xe = int((i_nxout)/i_skip) + 1
      i_ye = int((i_nyout)/i_skip) + 1
      if(mod(i_nxout,i_skip) .eq. 0)  i_xe = i_xe - 1
      if(mod(i_nyout,i_skip) .eq. 0)  i_ye = i_ye - 1
      
      do i=1,i_xe+1
         i_blockx(i) = i_skip*(i-1) + i_xmin
      enddo
      
      do i=1,i_ye+1
         i_blocky(i) = i_skip*(i-1) + i_ymin
      enddo
      
      i_blockx(i_xe+1) = min(i_blockx(i_xe+1),i_xmax+1)
      i_blocky(i_ye+1) = min(i_blocky(i_ye+1),i_ymax+1)

      write (*,*)  'i_xe = ',i_xe
      write (*,*)  'i_ye = ',i_ye
      write (*,*)  'Number of blocks = ',(i_xe)*(i_ye)
!      write (*,*)  ' '
!      write (*,*)  'Block x = ',(i_blockx(i),i=1,i_xe+1)
!      write (*,*)  ' '
!      write (*,*)  'Block y = ',(i_blocky(i),i=1,i_ye+1)
      

!c start reading and interpolating loop
!c
      do kk = 1 , i_ye

!c read in desired data from disk for this block of blocks
            
         i_yfrom =  max(i_blocky(kk)-i_padn,1)
         i_yto   =  min(i_blocky(kk+1)+i_padn-1,ndn)
         do i = i_yfrom, i_yto
            linenum = i-i_blocky(kk)+1
            j = i

            call getLineBand(inAcc,r_inarr(1,linenum),inband,j) 

            do j = 1 , nac
               r_outarr(j,linenum) =r_inarr(j,linenum)
            end do
         end do
         i_nyoutb = (i_blocky(kk+1) - i_blocky(kk))
         do i = 1, i_nyoutb
            r_yout(i) = i
         end do
         do ll = 1 , i_xe
            
            i_nxoutb = (i_blockx(ll+1) - i_blockx(ll))
            r_xout(1) = i_blockx(ll)
            do i = 2, i_nxoutb
               r_xout(i) = r_xout(i-1) + 1
            end do
            
            i_bcnt = 0
            i_xfrom = max(i_blockx(ll)-i_padn,1)
            i_xto   = min(i_blockx(ll+1)+i_padn-1,nac)
            r_blockarea = (i_yto-i_yfrom)*(i_xto-i_xfrom)
            r_blockareac = (i_yto-i_yfrom+1)*(i_xto-i_xfrom+1)
            
            do i = i_yfrom - i_blocky(kk) + 1, i_yto - i_blocky(kk) + 1
               do j = i_xfrom, i_xto
                  if(.not.isnan(r_inarr(j,i))) then
                     i_bcnt = i_bcnt + 1
                     r_bvx(i_bcnt) = j
                     r_bvy(i_bcnt) = i
                     r_bvz(i_bcnt) = r_inarr(j,i)
                  end if
               end do
            end do
            
            
            if(i_pflag .eq. 1)then
               write (*,*)  ' '
               write (*,*)  'Number of input points for xblock ',ll, ' = ',i_bcnt
               write (*,*)  'x from, to    ',i_xfrom, i_xto
               write (*,*)  'y from, to    ',i_yfrom, i_yto
               write (*,*)  'Block edges x ',i_blockx(ll),i_blockx(ll+1)-1
               write (*,*)  'Block edges y ',i_blocky(kk),i_blocky(kk+1)-1
            endif
            
            ach = 0.
            if(i_bcnt .gt. 1)  then
               call CONVEX_HULL(r_bvx,r_bvy,i_bcnt,xt,yt,nt,ACH,PCH)
               
               if(i_pflag .eq. 1)then
                  write (*,*)  ' '
                  write (*,*)  'r_bvx(1),r_bvy(1)           ', r_bvx(1),r_bvy(1) 
                  write (*,*)  'r_bvx(i_bcnt),r_bvy(i_bcnt) ', r_bvx(i_bcnt),r_bvy(i_bcnt)
                  write (*,*)  'convex stuff ',nt,pch,ach,i_bcnt,r_blockarea
               endif
               
               do l = 1 , i_nxoutb*i_nyoutb
                  r_surfh(l) = 0.
               end do

               if((ach .ge. r_blockarea*0.9) .and.  (i_bcnt .ne. int(r_blockareac+0.5))) then
                 i_md = 1
                  call idsfft(i_md,i_ncp,i_bcnt,r_bvx,r_bvy,r_bvz,i_nxoutb,i_nyoutb,r_xout,r_yout,r_surfh,r_thres,i_iwk,r_wk,2*nac)
                  do m =1,i_nyoutb
                     do l = 1,i_nxoutb
                        ii = (m-1)*i_nxoutb + l
                        r_outarr(l+i_blockx(ll)-1,m) = r_surfh(ii)
                     enddo
                  enddo 
               else
                  write (*,*)  'SKIPPING INTERPOLATION '
               end if
            end if
         end do

         do m = 1,i_nyoutb
            linenum=i_blocky(kk)+m-1
!!            write(13,rec=i_blocky(kk)+m-1) (r_outarr(l,m),l=1,nac*2)
            call setLineBand(outAcc,r_outarr(1,m),linenum,outband) 
         enddo
         
      end do

      DEALLOCATE(r_xout, r_yout)
      DEALLOCATE(xt, yt)
      DEALLOCATE(r_inarr, r_outarr)
      DEALLOCATE(i_iwk, r_wk)
      DEALLOCATE(i_blockx, i_blocky)
      DEALLOCATE(r_surfh, r_bvz)
      DEALLOCATE(r_bvx, r_bvy)
      
      end

      function db(x,th)

      real x, th

      db = log10(max(x,th))

      return
      end 
