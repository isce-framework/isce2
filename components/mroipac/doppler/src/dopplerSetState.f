      subroutine setSamples(varInt)
       use dopplerState
       implicit none
       integer varInt
       i_samples = varInt
      end

      subroutine setStartLine(varInt)
       use dopplerState
       implicit none
       integer varInt
       i_strtline = varInt
      end

      subroutine setLines(varInt)
       use dopplerState
       implicit none
       integer varInt
       i_nlines = varInt
      end
