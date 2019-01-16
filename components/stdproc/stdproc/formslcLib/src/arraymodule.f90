      module arraymodule
	! Common data block used by different parts of formSLC to share data

        complex*8, allocatable :: trans1(:,:),ref1(:) !trans1(nnn,mmm)
        real*8, allocatable :: s_mocomp(:),t_mocomp(:)
        integer, allocatable :: i_mocomp(:)
        real*8 ,allocatable :: phasegrad(:)
        double precision, allocatable, dimension(:,:) :: schMoc, vschMoc
        double precision, allocatable ::  timeMoc(:)
        integer*4 mocompSize
      end module

      !trans1      -> 2D array for transformed data
      !ref1        -> 1D array for the reference chirp
      !i_mocomp    -> Flag to check for mocomp processor
      !phasegrad   -> Array of phasegradients needed for mocomp
      !schMoc      -> SCH positions for mocomp processing
      !vschMoc     -> SCH velocities for mocomp processing
      !mocompSize  -> Number of lines
