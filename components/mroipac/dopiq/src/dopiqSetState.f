	  subroutine setLineLength(varInt)
		use dopiqState
		implicit none
		integer varInt
		len = varInt
	  end

	  subroutine setLineHeaderLength(varInt)
	    use dopiqState
	    implicit none
	    integer varInt
        hdr = varInt
	  end

	  subroutine setLastSample(varInt)
	    use dopiqState
	    implicit none
	    integer varInt
	    last = varInt
	  end

	  subroutine setStartLine(varInt)
	    use dopiqState
	    implicit none
	    integer varInt
        i0 = varInt
	  end

	  subroutine setNumberOfLines(varInt)
	    use dopiqState
	    implicit none
	    integer varInt
	    n = varInt
	  end

	  subroutine setMean(varDbl)
	    use dopiqState
	    implicit none
	    real*8 varDbl
	    xmn = varDbl
	  end

	  subroutine setPRF(varDbl)
	    use dopiqState
	    implicit none
	    real*8 varDbl
	    prf = varDbl
	  end
