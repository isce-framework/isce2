## StripMap stack processor

The detailed algorithms and workflow for stack processing of stripmap SAR data can be found here:

+ Fattahi, H., M. Simons, and P. Agram (2017), InSAR Time-Series Estimation of the Ionospheric Phase Delay: An Extension of the Split Range-Spectrum Technique, IEEE Transactions on Geoscience and Remote Sensing, 55(10), 5984-5996, doi:[10.1109/TGRS.2017.2718566](https://ieeexplore.ieee.org/abstract/document/7987747/).

-----------------------------------

To use the stripmap stack processor, make sure to add the path of your `contrib/stack/stripmapStack` folder to your `$PATH` environment varibale. 

Currently supported workflows include a coregistered stack of SLC, interferograms, ionospheric delays. 

Here are some notes to get started with processing stacks of stripmap data with ISCE. 

#### 1. Create your project folder somewhere   

```
mkdir MauleAlosDT111
cd MauleAlosDT111
```

#### 2. Prepare DEM    
a) create a folder for DEM;    
b) create a DEM using dem.py with SNWE of your study area in integer;   
d) Keep only ".dem.wgs84", ".dem.wgs84.vrt" and ".dem.wgs84.xml" and remove unnecessary files;    
d) fix the path of the file in the xml file of the DEM by using fixImageXml.py.   

```
mkdir DEM; cd DEM
dem.py -a stitch -b -37 -31 -72 -69 -r -s 1 -c
rm demLat*.dem demLat*.dem.xml demLat*.dem.vrt
cd ..
```

#### 3. Download data    

##### 3.1 create a folder to download SAR data (i.e. ALOS-1 data from ASF)   

```
mkdir download 
cd download
```

##### 3.2 Download the data that that you want to process to the "downlowd" directory   

#### 4. Prepare SAR data    
    
Once all data have been downloaded, we need to unzip them and move them to different folders and getting ready for unpacking and then SLC generation. This can be done by running the following command in a directory above "download":
      
```
prepRawALOS.py -i download/ -o SLC
```

This command generates an empty SLC folder and a run file called: "run_unPackALOS".    

You could also run prepRawSensor.py which aims to recognize the sensor data automatically followed by running the sensor specific preparation script. For now we include support for ALOS and CSK raw data, but it is trivial to expand and include other sensors as unpacking routines are already included in the distribution.

```
prepRawSensor.py -i download/ -o SLC
```

#### 5. Execute the commands in "run_unPackALOS" file   

If you have a cluster that you can submit jobs, you can submit each line of command to a processor. The commands are independent and can be run in parallel.    

After successfully running the previous step, you should see acquisition dates in the SLC folder and the ".raw" files for each acquisition.    

Note: For ALOS-1, If there is an acquisition that does not include .raw file, this is most likely due to PRF change between frames and can not be currently handled by ISCE. You have to ignore those.    

#### 6. Run "stackStripmap.py"   

This will generate many config and run files that need to be executed. Here is an example:    

```
stackStripMap.py -s SLC/ -d DEM/demLat*.dem.wgs84 -t 250 -b 1000 -a 14 -r 4 -u snaphu
```

This will produce:    
a) baseline folder, which contains baseline information   
b) pairs.png which is a baseline-time plot of the network of interferograms      
c) configs: which contains the configuration parameter to run different InSAR processing steps   
d) run_files: a folder that includes several run and job files that needs to be run in order    

#### 7. Execute the commands in run files (run_1*, run_2*, etc) in the "run_files" folder   
