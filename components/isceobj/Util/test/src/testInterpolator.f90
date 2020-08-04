subroutine testInterpolator(accessor2d,accessor1d)
implicit none
integer*8 accessor2d
integer*8 accessor1d
double precision, allocatable :: line1(:),line2(:)
integer i,j,azOrder,rgOrder,flag,getNumberOfLines,getWidth,width1d,width2d
double precision ret,getPx2d,getPx1d
azOrder = 2
rgOrder = 3

!test new getters
width2d = getWidth(accessor2d)
width1d = getWidth(accessor1d)
i = getNumberOfLines(accessor2d)
j = getNumberOfLines(accessor1d)
write(6,*) "sizes",width1d,j,width2d,i

#allocate buffer the get the dopplers
allocate(line1(width1d))
allocate(line2(width2d))

!test getting the single px for 1 or 2 d
do i = 0,azOrder
  do j = 0,rgOrder
    ret = getPx2d(accessor2d,i,j)
    write(6,*) 'pixel 2d',i,j,ret
    ret = getPx1d(accessor1d,j)
    write(6,*) 'pixel 1d',j,ret
  end do
end do
!get the azimuth doppler for each column 
! (which is width1d since width and length 
! have been flipped
flag = 0
j = 0
do while (flag .ge. 0)
    call getLineSequential(accessor1d,line1,flag)
    if(flag .ge. 0) then
        do i = 1,width1d
            write(6,*) "val ",j,i,line1(i)
        end do
    endif
    write(6,*)"flag",flag
    j = j + 1
end do

flag = 0
j = 0
do while (flag .ge. 0)
   call getLineSequential(accessor2d,line2,flag)
    if(flag .ge. 0) then
        do i = 1,width2d
            write(6,*) "val ",j,i,line2(i)
        end do
    endif
    write(6,*)"flag",flag
    j = j + 1
end do

deallocate(line1)
deallocate(line2)

end subroutine testInterpolator
