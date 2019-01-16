       module aikimaState
         integer nac                 !Number of Pixels Across
         integer ndn                 !Number of Pixels Down
         integer i_xmin              !Start pixel to process across
         integer i_xmax              !End pixel to process across
         integer i_ymin              !Start pixel to process down
         integer i_ymax              !End pixel to process down
         integer i_skip              !Block Size
         integer i_padn              !Pad Size
         integer i_ncp               !Number of points for partials
         integer i_pflag             !Print flag
         real*4 r_thres                !Threshold
       end module aikimaState
