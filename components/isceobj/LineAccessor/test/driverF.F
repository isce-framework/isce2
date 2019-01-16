	   program driverF 
	   implicit none
	
	character*1 endian, machineEnd
	character*256 infile, filemode,outfile, filemodeOut, vartype, inputArg
       integer*8 ptImageAccessorSet !pointers to image accessor objects
       integer*8 ptImageAccessorGet !pointers to image accessor objects
       integer i,j,k
       integer*4 eofGet, lineWidth,length, row, col, numEl, choice, numBand, bandIn, bandOut,vartypeSize
	   
	! this driver runs 4 different examples where different methods of the image API are used.
	! for choice = 1,2 the program calls testImageSetGet() where data are read in different ways and put back in the output file.
	! if choice = 2 the file and machine indianness are set to opposite values. see the fortranSrc.F file for more details.
	! if choice = 3 the endianness of the input file is changed and written into the output file.
	! if choice = 4 the  inteleaving scheme of the input file is changed and written into the output file.
	
	! to compare the input and output files issue the following command
	! "od -N numBytes -t u1 -v filename"
	! this commad shows the content of each byte in the file "filename". The input file adopted contains for each byte an increasing number
	! from 0 to 159 for a total of 160 bytes.
	   choice = 1
	   if  (iargc() .ge. 1) then
		call getarg(1,inputArg)
		read(inputArg,*) choice
	   endif
	   if((choice .eq. 1) .or. (choice .eq. 2)) then
	       infile = "testFile"
	       filemode = "read"
	       filemodeOut = "writeread"
	       vartype = "FLOAT"
	       endian = 'l'
	       call getLineAccessorObject(ptImageAccessorGet)
	       call getLineAccessorObject(ptImageAccessorSet)
	       call getMachineEndianness(ptImageAccessorGet,machineEnd)
	       call getTypeSize(ptImageAccessorGet,vartype,vartypeSize)
	       if(choice .eq. 1) then
			outfile = "testOutF1"
			if((machineEnd .eq. 'b')) then
			    endian = 'b'
			endif
	       elseif (choice .eq. 2) then
			outfile = "testOutF2"
			if((machineEnd .eq. 'l')) then
			    endian = 'b';
			endif
	       endif
	       row = 3;
	       col = 4
	       call initLineAccessor(ptImageAccessorGet,infile, filemode, endian, vartype, row,col)
	       call initLineAccessor(ptImageAccessorSet,outfile, filemodeOut, endian, vartype, row,col)

	       call testImageSetGet(ptImageAccessorGet,ptImageAccessorSet, choice)

	       call finalizeLineAccessor(ptImageAccessorGet)
	       call finalizeLineAccessor(ptImageAccessorSet)
	   elseif (choice .eq. 3) then

	       infile = "testSwap"
	       outfile = "testSwapOutF"
	       vartype = "FLOAT"
	       
	       call getLineAccessorObject(ptImageAccessorGet)
	       call convertFileEndianness(ptImageAccessorGet,infile,outfile,vartype)
	   elseif (choice .eq. 4) then
	   
	       infile = "testFileBand"
	       outfile = "testFileBandOutF"
	       filemode = "read"
	       vartype = "FLOAT"
	       col = 4
	       numBand = 2
	       bandIn = 1 ! BSQ
	       bandOut = 3 ! BIL
	       
	       call getLineAccessorObject(ptImageAccessorGet)
	       call changeBandScheme(ptImageAccessorGet,infile,outfile,vartype,col,numBand,bandIn,bandOut)
	   endif
	   end
