#!/bin/tcsh -f

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# copyright: 2016 to the present, california institute of technology.
# all rights reserved. united states government sponsorship acknowledged.
#
# THESE SCRIPTS ARE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.
#
# Author: David Bekaert
# Organization: Jet Propulsion Laboratory, California Institute of Technology
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

set overwrite = 0
set data_path = /u/k-data/dbekaert/HMA_nepal/Sentinel1/track_019/processing_1/merged
set proc_dir = /u/k-data/dbekaert/HMA_nepal/Sentinel1/track_019/processing_1/crop_testing

# check if the procesing dir already exists
if (! -e $proc_dir) then
   mkdir $proc_dir
endif
    
# getting the crop extend
cd $data_path/geom_master
crop_rdr.py -b '27.86 28.2 85.1 85.4' > $proc_dir/crop_log.txt


#### NO changes required below ######

# getting the files to crop
cd $proc_dir
ls -1 $data_path/geom_master/*.full  > $proc_dir/geomFiles2crop.txt
ls -1 $data_path/SLC/2*/2*.slc.full > $proc_dir/slcFiles2crop.txt
ls -1 $data_path/baselines/2*/2*.full.vrt >  $proc_dir/baselineFiles2crop.txt

# getting the cropping command
set command_baseline = `grep warp $proc_dir/crop_log.txt`
set command = `grep gdal_translate $proc_dir/crop_log.txt`
echo $command
echo $command_baseline

# generating the new geometry files
# create geom directory
cd $proc_dir
if (! -d geom_master ) then
    mkdir geom_master
endif
cd geom_master
foreach file(`cat $proc_dir/geomFiles2crop.txt`)
   set filename = `basename $file`
   
   # crop the files
   if ( -f $filename & $overwrite == 0) then
        echo File exist
   else
        echo $command $file $filename
        `echo $command $file $filename`

       # generate the xml files for it
        echo gdal2isce_xml.py -i $filename
        `echo gdal2isce_xml.py -i $filename`
   endif
end

# generating the new geometry files
# create SLC directory
cd $proc_dir
if (! -d SLC ) then
    mkdir SLC
endif
cd SLC
foreach file(`cat $proc_dir/slcFiles2crop.txt`)
   set filename = `basename $file`
   set date = `basename $file | cut -c1-8`
   echo $date

   # make the SLC date dir
   if (! -d $date ) then
      mkdir $date
   endif
   cd $date
                  

   # crop the files
   if ( -f $filename & $overwrite == 0) then
        echo File exist
   else
        echo $command $file $filename
        `echo $command $file $filename`

       # generate the xml files for it
        echo gdal2isce_xml.py -i $filename
        `echo gdal2isce_xml.py -i $filename`
   endif
  
   cd $proc_dir/SLC
end
                      
# generating the new baseline files
# create the baseline directory
cd $proc_dir
if (! -d baselines ) then
    mkdir baselines
endif
cd baselines
foreach file(`cat $proc_dir/baselineFiles2crop.txt`)
    set filename = `basename $file`  
    set date = `basename $file | cut -c1-8`
    echo $date

    # make the SLC date dir
    if (! -d $date ) then
       mkdir $date
    endif
    cd $date
    
    # crop the files
    if ( -f $filename & $overwrite == 0) then
        echo File exist
   else
        echo $command_baseline $file $date
        `echo $command_baseline $file $date`

        # generate the xml files for it
        echo gdal2isce_xml.py -i $date
        `echo gdal2isce_xml.py -i $date`
    endif      

    cd $proc_dir/baselines
end

