## Sentinel-1 TOPS stack processor

The detailed algorithm for stack processing of TOPS data can be find here:

+ Fattahi, H., P. Agram, and M. Simons (2016), A Network-Based Enhanced Spectral Diversity Approach for TOPS Time-Series Analysis, IEEE Transactions on Geoscience and Remote Sensing, 55(2), 777-786, doi:[10.1109/TGRS.2016.2614925](https://ieeexplore.ieee.org/abstract/document/7637021).

-----------------------------------

To use the sentinel stack processor, make sure to add the path of your `contrib/stack/topsStack` folder to your `$PATH` environment varibale. 

The scripts provides support for Sentinel-1 TOPS stack processing. Currently supported workflows include a coregistered stack of SLC, interferograms, offsets, and coherence. 

`stackSentinel.py` generates all configuration and run files required to be executed on a stack of Sentinel-1 TOPS data. When stackSentinel.py is executed for a given workflow (-W option) a **configs** and **run_files** folder is generated. No processing is performed at this stage. Within the run_files folder different run\_#\_description files are contained which are to be executed as shell scripts in the run number order. Each of these run scripts call specific configure files contained in the “configs” folder which call ISCE in a modular fashion. The configure and run files will change depending on the selected workflow. To make run_# files executable, change the file permission accordingly (e.g., `chmod +x run_1_unpack_slc`).

```bash
stackSentinel.py -H     #To see workflow examples,
stackSentinel.py -h     #To get an overview of all the configurable parameters
```

Required parameters of stackSentinel.py include:

```cfg
-s SLC_DIRNAME          #A folder with downloaded Sentinel-1 SLC’s. 
-o ORBIT_DIRNAME        #A folder containing the Sentinel-1 orbits. Missing orbit files will be downloaded automatically
-a AUX_DIRNAME          #A folder containing the Sentinel-1 Auxiliary files
-d DEM_FILENAME         #A DEM (Digital Elevation Model) referenced to wgs84
```

In the following, different workflow examples are provided. Note that stackSentinel.py only generates the run and configure files. To perform the actual processing, the user will need to execute each run file in their numbered order.

In all workflows, coregistration (-C option) can be done using only geometry (set option = geometry) or with geometry plus refined azimuth offsets through NESD (set option = NESD) approach, the latter being the default. For the NESD coregistrstion the user can control the ESD coherence threshold (-e option) and the number of overlap interferograms (-O) to be used in NESD estimation.

#### AUX_CAL file download ####

The following calibration auxliary (AUX_CAL) file is used for **antenna pattern correction** to compensate the range phase offset of SAFE products with **IPF verison 002.36** (mainly for images acquired before March 2015). If all your SAFE products are from another IPF version, then no AUX files are needed. Check [ESA document](https://earth.esa.int/documents/247904/1653440/Sentinel-1-IPF_EAP_Phase_correction) for details. 

Run the command below to download the AUX_CAL file once and store it somewhere (_i.e._ ~/aux/aux_cal) so that you can use it all the time, for `stackSentinel.py -a` or `auxiliary data directory` in `topsApp.py`.

```
wget https://qc.sentinel1.eo.esa.int/product/S1A/AUX_CAL/20140908T000000/S1A_AUX_CAL_V20140908T000000_G20190626T100201.SAFE.TGZ
tar zxvf S1A_AUX_CAL_V20140908T000000_G20190626T100201.SAFE.TGZ
rm S1A_AUX_CAL_V20140908T000000_G20190626T100201.SAFE.TGZ
```

#### 1. Create your project folder somewhere ####

```
mkdir MexicoSenAT72
cd MexicoSenAT72
```

#### 2. Prepare DEM ####

Download of DEM (need to use wgs84 version) using the ISCE DEM download script.

```
mkdir DEM; cd DEM
dem.py -a stitch -b 18 20 -100 -97 -r -s 1 –c
rm demLat*.dem demLat*.dem.xml demLat*.dem.vrt
cd ..
```

#### 3. Download Sentinel-1 data to SLC ####



#### 4.1 Example workflow: Coregistered stack of SLC ####

Generate the run and configure files needed to generate a coregistered stack of SLCs. In this example, a pre-defined bounding box is specified. Note, if the bounding box is not provided it is set by default to the common SLC area among all SLCs. We recommend that user always set the processing bounding box. Since ESA does not have a fixed frame definition, we suggest to download data for a larger bounding box compared to the actual bounding box used in stackSentinel.py. This way user can ensure to have required data to cover the region of interest. Here is an example command to create configuration files for a stack of SLCs:

```
stackSentinel.py -s ../SLC/ -d ../DEM/demLat_N18_N20_Lon_W100_W097.dem.wgs84 -a ../../AuxDir/ -o ../../Orbits -b '19 20 -99.5 -98.5' -W slc
```

by running the command above, the configs and run_files folders are created. User needs to execute each run file in order. The order is specified by the index number of the run file name. For the example above, the run_files folder includes the following files:

-	run_1_unpack_slc_topo_master
-	run_2_average_baseline
-	run_3_extract_burst_overlaps
-	run_4_overlap_geo2rdr_resample
-	run_5_pairs_misreg
-	run_6_timeseries_misreg
-	run_7_geo2rdr_resample
-	run_8_extract_stack_valid_region
-	run_9_merge
-	run_10_grid_baseline

The generated run files are self descriptive. Below is a short explanation on what each run_file does:

**run_1_unpack_slc_topo_master:**

Includes commands to unpack Sentinel-1 TOPS SLCs using ISCE readers. For older SLCs which need antenna elevation pattern correction, the file is extracted and written to disk. For newer version of SLCs which don’t need the elevation antenna pattern correction, only a gdal virtual “vrt” file (and isce xml file) is generated. The “.vrt” file points to the Sentinel SLC file and reads them whenever required during the processing. If a user wants to write the “.vrt” SLC file to disk, it can be done easily using gdal_translate (e.g. gdal_translate –of ENVI File.vrt File.slc). 
The “run_1_unpack_slc_topo_master” also includes a command that refers to the config file of the stack master, which includes configuration for running topo for the stack master. Note that in the pair-wise processing strategy one should run topo (mapping from range-Doppler to geo coordinate) for all pairs. However, with stackSentinel, topo needs to be run only one time for the master in the stack. 

**run_2_average_baseline:**

Computes average baseline for the stack. These baselines are not used for processing anywhere. They are only an approximation and can be used for plotting purposes. A more precise baseline grid is estimated later in run_10.

**run_3_extract_burst_overlaps:**

Burst overlaps are extracted for estimating azimuth misregistration using NESD technique. If coregistration method is chosen to be “geometry”, then this run file won’t exist and the overlaps are not extracted.

**run_4_overlap_geo2rdr_resample:***

Running geo2rdr to estimate geometrical offsets between slave burst overlaps and the stack master burst overlaps. The slave burst overlaps are then resampled to the stack master burst overlaps. 

**run_5_pairs_misreg:**

Using the coregistered stack burst overlaps generated from the previous step, differential overlap interferograms are generated and are used for estimating azimuth misregistration using Enhanced Spectral Diversity (ESD) technique. 

**run_6_timeseries_misreg:**

A time-series of azimuth and range misregistration is estimated with respect to the stack master. The time-series is a least squares esatimation from the pair misregistration from the previous step.

**run_7_geo2rdr_resample:**

Using orbit and DEM, geometrical offsets among all slave SLCs and the stack master is computed. The goometrical offsets, together with the misregistration time-series (from previous step) are used for precise coregistration of each burst SLC. 

**run_8_extract_stack_valid_region:**

The valid region between burst SLCs at the overlap area of the bursts slightly changes for different acquisitions. Therefore we need to keep track of these overlaps which will be used during merging bursts. Without these knowledge, lines of invalid data may appear in the merged products at the burst overlaps.

**run_9_merge:**

Merges all bursts for the master and coregistered SLCs. The geometry files are also merged including longitude, latitude, shadow and layer mask, line-of-sight files, etc. .

**run_10_grid_baseline:**

A coarse grid of baselines between each slave SLC and the stack master is generated. This is not used in any computation.

#### 4.2 Example workflow: Coregistered stack of SLC with modified parameters ####

In the following example, the same stack generation is requested but where the threshold of the default coregistration approach (NESD) is relaxed from its default 0.85 value 0.7.

```
stackSentinel.py -s ../SLC/ -d ../DEM/demLat_N18_N20_Lon_W100_W097.dem.wgs84 -a ../../AuxDir/ -o ../../Orbits -b '19 20 -99.5 -98.5' -W slc -e 0.7
```

When running all the run files, the final products are located in the merge folder which has subdirectories **geom_master**, **baselines** and **SLC**. The **geom_master** folder contains geometry products such as longitude, latitude, height, local incidence angle, look angle, heading, and shadowing/layover mask files. The **baselines** folder contains sparse grids of the perpendicular baseline for each acquisition, while the **SLC** folder contains the coregistered SLCs

#### 4.3 Example workflow: Stack of interferograms ####

Generate the run and configure files needed to generate a stack of interferograms.
In this example, a stack of interferograms is requested for which up to 2 nearest neighbor connections are included.

```
stackSentinel.py -s ../SLC/ -d ../../MexicoCity/demLat_N18_N20_Lon_W100_W097.dem.wgs84 -b '19 20 -99.5 -98.5' -a ../../AuxDir/ -o ../../Orbits -c 2
```

In the following example, all possible interferograms are being generated and in which the coregistration approach is set to use geometry and not the default NESD. 

```
stackSentinel.py -s ../SLC/ -d ../../MexicoCity/demLat_N18_N20_Lon_W100_W097.dem.wgs84 -b '19 20 -99.5 -98.5' -a ../../AuxDir/ -o ../../Orbits -C geometry -c all
```

When executing all the run files, a coregistered stack of slcs are produced, the burst interferograms are generated and then merged. Merged interferograms are multilooked, filtered and unwrapped. Geocoding is not applied. If users need to geocode any product, they can use the geocodeGdal.py script.

#### 4.4 Example workflow: Stack of correlation ####

Generate the run and configure files needed to generate a stack of coherence.
In this example, a correlation stack is requested considering all possible coherence pairs and where the coregistration approach is done using geometry only.

```
stackSentinel.py -s ../SLC/ -d ../../MexicoCity/demLat_N18_N20_Lon_W100_W097.dem.wgs84 -b '19 20 -99.5 -98.5' -a ../../AuxDir/ -o ../../Orbits -C geometry -c all -W correlation
```

This workflow is basically similar to the previous one. The difference is that the interferograms are not unwrapped.

#### 5. Execute the commands in run files (run_1*, run_2*, etc) in the "run_files" folder ####
