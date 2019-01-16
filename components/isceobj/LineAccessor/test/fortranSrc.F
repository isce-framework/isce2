       
       subroutine testImageSetGet(ptImageAccessorGet,ptImageAccessorSet,choice)
       implicit none

       integer*8 ptImageAccessorSet !pointers to image accessor objects
       integer*8 ptImageAccessorGet !pointers to image accessor objects
       integer*4 choice
       integer i,j,k
       integer*4 eofGet, lineWidth,length, col, row,numEl, addVal,newLength
       integer*4, allocatable, dimension(:):: dataLineRowIndx,dataLineColIndx
       real*4, allocatable, dimension(:):: dataLineGet,dataLineSet
	!this code shows few instance of how to read and write data to a file using the image API.
	! data are read from one file and set into another. if choice = 2 the test is done assumning that the file endianness
	! is opposite of the machine one, so in the output each group of fout bytes is swapped.
	! part of the data are accessed first half line at the time in sequential order.
	! second they are accessed randomly (see below for the exact order).
	! third the remaining are accessed one line at the time.
	if ((choice .eq. 1) .or. (choice .eq. 2)) then
	       ! image is already initialized, so get some of the information
	       call getFileWidth(ptImageAccessorGet, lineWidth)	
	       call getFileLength(ptImageAccessorGet, length)
	       ! plan to access the output file randomly, so the file needs to exist already in its full size.
	       call createFile(ptImageAccessorSet, length)	
	       
	       allocate(dataLineGet(lineWidth))! where read data are stored
	       allocate(dataLineSet(lineWidth))! where write data are stored
	       allocate(dataLineRowIndx(lineWidth))! array with the row index positions when reading/writing randomly to file
	       allocate(dataLineColIndx(lineWidth))! array with the column index positions when reading/writing randomly to file

	       ! print the images info
	       call printObjectInfo(ptImageAccessorGet)
	       call printObjectInfo(ptImageAccessorSet)
	       numEl = lineWidth/2
	       j = 1
	       !access the first 1/4 of the file  sequentially, half line at the time. Use get,setSequentialElements() 
	       do i = 1, length/4   
			    do k = 1, 2
				col = 1 + numEl*(k-1)! half line at the time
				!get numEl elements from the image associated with ptImageAccessorGet starting from row j  and column col
				! and put them in dataLineGet
				call getSequentialElements(ptImageAccessorGet,dataLineGet,j,col,numEl)
			
				dataLineSet(1:numEl) = dataLineGet(1:numEl)
				!set the numEl elements to the image associated with ptImageAccessorSet starting from row j  and column col
				call setSequentialElements(ptImageAccessorSet,dataLineSet,j,col,numEl)
			    enddo
			    j = j + 1

	       enddo
	       !access the second 1/4  (plus reminder if length/4 not integer) reading or setting lineWith elements at the time but for each element move one column and one row up taking 
	       ! the modulo of the lineWidth and length (i.e. access elements on the diagonals). in a 3 by 4 matrix this is the order in which 
	       !elements are accessed
	       ! 1  5  9
	       ! 10 2  6
	       ! 7  11 3 
	       ! 4  8  12   
	       
	       numEl = lineWidth
		row = 0 
		col = 0
		j = 0
		newLength = length/4 + mod(length/2,4)
	       do i = 1, newLength*lineWidth
			j = j + 1
			! set the row and column indeces
		        dataLineRowIndx(j) = mod(row, newLength) + length/4 + 1
		        dataLineColIndx(j) = mod(col, lineWidth) + 1
			col = col + 1
			row = row + 1
			if (mod(j,lineWidth) .eq. 0) then
			     !get the data in the positions specified by  dataLineColIndx nad dataLineRowIndx
	                     call getElements(ptImageAccessorGet,dataLineGet,dataLineRowIndx,dataLineColIndx,numEl)
			     ! set the data in the same position
	                     call setElements(ptImageAccessorSet,dataLineGet,dataLineRowIndx,dataLineColIndx,numEl)
		            
			    j = 0
			endif
	       enddo
	       
	       ! access the rest 1/4  of the files one line at the time using the get,setLine
	       
	       do i = length/2 + 1, length/2 + length/4    
			    eofGet = i
			    call getLine(ptImageAccessorGet,dataLineGet,eofGet)
			    dataLineSet(:) = dataLineGet(:)
			    ! data from dataLineSet are put into the tile
			    call setLine(ptImageAccessorSet,dataLineSet,eofGet)


	    
	       enddo
	       ! access the rest 1/4  of the files one line at the time using the get,setLineSequential
	       call initSequentialAccessor(ptImageAccessorGet,length/2 + length/4 + 1)! need to set the first line
	       call initSequentialAccessor(ptImageAccessorSet,length/2 + length/4 + 1)! need to set the first line

	       do    
			    call getLineSequential(ptImageAccessorGet,dataLineGet,eofGet)
			    !when eofGet < 0 then the end of file has been reached.
			    if(eofGet .lt. 0) then
				    exit
			    endif
			    dataLineSet(:) = dataLineGet(:)
			    ! data from dataLineSet are put into the tile
			    call setLineSequential(ptImageAccessorSet,dataLineSet)
			    j = j - 1


	    
	       enddo

	       deallocate(dataLineGet)
	       deallocate(dataLineSet)
	       deallocate(dataLineRowIndx)
	       deallocate(dataLineColIndx)


	       endif

       end
