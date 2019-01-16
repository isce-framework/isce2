
      subroutine inter_motion(r_time,r_xyz,i_npts,r_t,r_xyzint)

      real*8 r_time(*),r_xyz(3,*),r_t,r_xyzint(3)
      real*8 r_temp(3),r_tt(3),r_dy
      integer i_npts,i,j,i_ind,i_use(3),i_order

      i_order = 2+1
      call hunt(r_time,i_npts,r_t,i_ind)

      if(abs(r_t - r_time(i_ind)) .le. abs(r_t - r_time(i_ind+1)) .and. 
     +     i_ind+1 .le. i_npts .and. i_ind-1 .ge. 1)then
         r_tt(1) = r_time(i_ind-1)
         r_tt(2) = r_time(i_ind)
         r_tt(3) = r_time(i_ind+1)
         i_use(1) = i_ind-1
         i_use(2) = i_ind
         i_use(3) = i_ind+1
      elseif(abs(r_t - r_time(i_ind)) .le. abs(r_t - r_time(i_ind+1)) .and. 
     +        i_ind+2 .le. i_npts .and. i_ind-1 .le. 0)then
         r_tt(1) = r_time(i_ind)
         r_tt(2) = r_time(i_ind+1)
         r_tt(3) = r_time(i_ind+2)
         i_use(1) = i_ind
         i_use(2) = i_ind+1
         i_use(3) = i_ind+2
      elseif(abs(r_t - r_time(i_ind)) .gt. abs(r_t - r_time(i_ind+1)) .and.
     +        i_ind+2 .le. i_npts)then
         r_tt(1) = r_time(i_ind)
         r_tt(2) = r_time(i_ind+1)
         r_tt(3) = r_time(i_ind+2)
         i_use(1) = i_ind
         i_use(2) = i_ind+1
         i_use(3) = i_ind+2
      elseif(abs(r_t - r_time(i_ind)) .gt. abs(r_t - r_time(i_ind+1)) .and.
     +        i_ind+1 .le. i_npts .and. i_ind-1 .ge. 1)then
         r_tt(1) = r_time(i_ind-1)
         r_tt(2) = r_time(i_ind)
         r_tt(3) = r_time(i_ind+1)
         i_use(1) = i_ind-1
         i_use(2) = i_ind
         i_use(3) = i_ind+1
      else
         write(6,*) 'Problem with motion data...'
         stop
      endif

      do i=1,3
         do j=1,3
            r_temp(j) = r_xyz(i,i_use(j))
         enddo
         call polint(r_tt,r_temp,i_order,r_t,r_xyzint(i),r_dy)
      enddo

      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

