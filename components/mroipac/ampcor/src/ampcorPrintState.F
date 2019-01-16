       subroutine ampcorPrintState()
       
         use ampcorState
         implicit none
         integer i

         write(6,*) "a_datatype    = ",(a_datatype(i),i=1,2)
         write(6,*) "i_samples     = ",(i_samples(i),i=1,2)
         write(6,*) "i_strtline    = ",i_strtline
         write(6,*) "i_endline     = ",i_endline
         write(6,*) "i_skipline    = ",i_skipline
         write(6,*) "i_strtsamp    = ",i_strtsamp
         write(6,*) "i_endsamp     = ",i_endsamp
         write(6,*) "i_skipsamp    = ",i_skipsamp
         write(6,*) "i_wsxi        = ",i_wsxi
         write(6,*) "i_wsyi        = ",i_wsyi
         write(6,*) "i_srchx       = ",i_srchx
         write(6,*) "i_srchy       = ",i_srchy
         write(6,*) "i_avgx        = ",i_avgx
         write(6,*) "i_avgy        = ",i_avgy
         write(6,*) "i_covs        = ",i_covs
         write(6,*) "i_cw          = ",i_cw
         write(6,*) "i_grossx      = ",i_grossx
         write(6,*) "i_grossy      = ",i_grossy
         write(6,*) "r_snrth       = ",r_snrth
         write(6,*) "r_covth       = ",r_covth
         write(6,*) "l_debug       = ",l_debug
         write(6,*) "l_display     = ",l_display
   
         return
       end
         
