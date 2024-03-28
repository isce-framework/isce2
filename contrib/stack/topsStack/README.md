## Sentinel-1 TOPS stack processor

The detailed algorithm for stack processing of TOPS data can be find here:

+ Fattahi, H., P. Agram, and M. Simons (2016), A Network-Based Enhanced Spectral Diversity Approach for TOPS Time-Series Analysis, IEEE Transactions on Geoscience and Remote Sensing, 55(2), 777-786, doi:[10.1109/TGRS.2016.2614925](https://ieeexplore.ieee.org/abstract/document/7637021).

-----------------------------------

To use the sentinel stack processor, make sure to add the path of your `contrib/stack/topsStack` folder to your `$PATH` environment varibale.

The scripts provides support for Sentinel-1 TOPS stack processing. Currently supported workflows include a coregistered stack of SLC, interferograms, offsets, and coherence.

`stackSentinel.py` generates all configuration and run files required to be executed on a stack of Sentinel-1 TOPS data. When stackSentinel.py is executed for a given workflow (-W option) a **configs** and **run_files** folder is generated. No processing is performed at this stage. Within the run_files folder different run\_#\_description files are contained which are to be executed as shell scripts in the run number order. Each of these run scripts call specific configure files contained in the “configs” folder which call ISCE in a modular fashion. The configure and run files will change depending on the selected workflow. To make run_# files executable, change the file permission accordingly (e.g., `chmod +x run_01_unpack_slc`).

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

The AUX_CAL file is available on [Sentinel-1 Mission Performance Center](https://sar-mpc.eu/ipf-adf/aux_cal/?sentinel1__mission=S1A&validity_start=2014&validity_start=2014-09&adf__active=True). We recommend download it using the web brower or the `wget` command below, and store it somewhere (_i.e._ ~/aux/aux_cal) so that you can use it all the time, for `stackSentinel.py -a` or `auxiliary data directory` in `topsApp.py`.

```
wget https://sar-mpc.eu/download/ca97845e-1314-4817-91d8-f39afbeff74d/ -O S1A_AUX_CAL_V20140908T000000_G20190626T100201.SAFE.zip
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
dem.py -a stitch -b 18 20 -100 -97 -r -s 1 -c
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

-	run_01_unpack_topo_reference
-	run_02_unpack_secondary_slc
-	run_03_average_baseline
-	run_04_extract_burst_overlaps
-	run_05_overlap_geo2rdr
-	run_06_overlap_resample
-	run_07_pairs_misreg
-	run_08_timeseries_misreg
-	run_09_fullBurst_geo2rdr
-	run_10_fullBurst_resample
-   run_11_extract_stack_valid_region
-   run_12_merge_reference_secondary_slc
-   run_13_grid_baseline

The generated run files are self descriptive. Below is a short explanation on what each run_file does:

**run_01_unpack_slc_topo_reference:**

Includes a command that refers to the config file of the stack reference, which includes configuration for running `topo` for the stack reference. Note that in the pair-wise processing strategy, one should run `topo` (mapping from range-Doppler to geo coordinate) for all pairs. However, with `stackSentinel.py`, `topo` needs to be run only one time for the reference in the stack. This stage will also unpack Sentinel-1 TOPS reference SLC. Reference geometry files are saved under `geom_reference/`. Reference burst SLCs are saved under `reference/`.

**run_02_unpack_secondary_slc:**

Unpack secondary Sentinel-1 TOPS SLCs using ISCE readers. For older SLCs which need antenna elevation pattern correction, the file is extracted and written to disk. For newer version of SLCs which don’t need the elevation antenna pattern correction, only a gdal virtual “vrt” file (and isce xml file) is generated. The “.vrt” file points to the Sentinel SLC file and reads them whenever required during the processing. If a user wants to write the “.vrt” SLC file to disk, it can be done easily using `gdal_translate` (e.g. `gdal_translate –of ENVI File.vrt File.slc`). Secondary burst SLCs are saved under `secondarys/`.

**run_03_average_baseline:**

Computes average baseline for the stack, saved under `baselines/`. These baselines are not used for processing anywhere. They are only an approximation and can be used for plotting purposes. A more precise baseline grid is estimated later in `run_13_grid_baseline` only for `-W slc` workflow.

**run_04_extract_burst_overlaps:**

Burst overlaps are extracted for estimating azimuth misregistration using the [NESD technique](https://ieeexplore.ieee.org/document/7637021). If coregistration method is chosen to be “geometry”, then this run file won’t exist and the overlaps are not extracted. Saved under `reference/overlap/` and `geom_reference/overlap`.

**run_05_overlap_geo2rdr:**

Running geo2rdr to estimate geometrical offsets between secondary burst overlaps (`secondary/`) and the stack reference (`reference`) burst overlaps. Saved under `coreg_secondarys/YYYYMMDD/overlap`.

**run_06_overlap_resample:**

The secondary burst overlaps are then resampled to the stack reference burst overlaps. Saved under `coreg_secondarys/YYYYMMDD/overlap`.

**run_07_pairs_misreg:**

Using the coregistered stack burst overlaps generated from the previous step, differential overlap interferograms are generated and are used for estimating azimuth misregistration using Enhanced Spectral Diversity (ESD) technique. Saved under `misreg/azimuth/pairs/` and `misreg/range/pairs/`.

**run_08_timeseries_misreg:**

A time-series of azimuth and range misregistration is estimated with respect to the stack reference. The time-series is a least-squares estimation from the pair misregistration from the previous step. Saved under `misreg/azimuth/dates/` and `misreg/range/dates/`.

**run_09_fullBurst_geo2rdr:**

Using orbit and DEM, geometrical offsets among all secondary SLCs and the stack reference is computed. Saved under `coreg_secondarys/`.

**run_10_fullBurst_resample:**

The geometrical offsets, together with the misregistration time-series (from the previous step) are used for precise coregistration of each burst SLC by resampling to the stack reference burst SLC. Saved under `coreg_secondarys/`.

**run_11_extract_stack_valid_region:**

The valid region between burst SLCs at the overlap area of the bursts slightly changes for different acquisitions. Therefore, we need to keep track of these overlaps which will be used during merging bursts. Without these knowledges, lines of invalid data may appear in the merged products at the burst overlaps.

**run_12_merge_reference_secondary_slc:**

Merges all bursts for the reference and coregistered SLCs and apply multilooking to form full-scene SLCs (saved under  `merged/SLC/` if --virtual_merge is True). The geometry files are also merged including longitude, latitude, shadow and layer mask, line-of-sight files, etc. under `merged/geom_reference/`.

**run_13_grid_baseline:**

A coarse grid of baselines between each secondary SLC and the stack reference is generated. This is not used in any computation. Saved under `merged/baselines/`.

#### 4.2 Example workflow: Coregistered stack of SLC with modified parameters ####

In the following example, the same stack generation is requested but where the threshold of the default coregistration approach (NESD) is relaxed from its default 0.85 value 0.7.

```
stackSentinel.py -s ../SLC/ -d ../DEM/demLat_N18_N20_Lon_W100_W097.dem.wgs84 -a ../../AuxDir/ -o ../../Orbits -b '19 20 -99.5 -98.5' -W slc -e 0.7
```

When running all the run files, the final products are located in the merge folder which has subdirectories **geom_reference**, **baselines** and **SLC**. The **geom_reference** folder contains geometry products such as longitude, latitude, height, local incidence angle, look angle, heading, and shadowing/layover mask files. The **baselines** folder contains sparse grids of the perpendicular baseline for each acquisition, while the **SLC** folder contains the coregistered SLCs

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

Compared to the "Coregistered stack of SLC workflow" (`-W SLC`),

~~**run_13_grid_baseline:**~~ This step does not exist in `-W interferogram` workflow.

But additional run_files are created in the `-W interferogram` workflow,

-   run_13_generate_burst_igram
-   run_14_merge_burst_igram
-   run_15_filter_coherence
-   run_16_unwrap

Below is a short explanation on what each run_file does:

**run_13_generate_burst_igram:**

Take the stack of coregistered burst SLCs (`reference` and  `coreg_secondary`) to generate burst interferograms. These burst-level interferograms are saved under `interferograms/`.

**run_14_merge_burst_igram:**

Merge the burst interferograms and apply multilooking to form a full-scene interferogram for each acquisition. Saved under `merged/interferograms/fine.int`

**run_15_filter_coherence:**

Use the full-scene SLCs in `merged/SLC/` to generate the complex coherence. Apply filtering to the full-scene interferograms and the coherence files. These files are saved as `fine.cor`, `filt_fine.int`, `filt_fine.cor` under `merged/interferograms/`.

**run_16_unwrap:**

Apply unwrapping to the multilooked and filtered interferograms `merged/interferograms/filt_fine.int`, generate the unwrapped files, `merged/interferograms/filt_fine.unw`.


#### 4.4 Example workflow: Stack of correlation ####

Generate the run and configure files needed to generate a stack of coherence.
In this example, a correlation stack is requested considering all possible coherence pairs and where the coregistration approach is done using geometry only.

```
stackSentinel.py -s ../SLC/ -d ../../MexicoCity/demLat_N18_N20_Lon_W100_W097.dem.wgs84 -b '19 20 -99.5 -98.5' -a ../../AuxDir/ -o ../../Orbits -C geometry -c all -W correlation
```

This workflow is basically similar to the previous one. The difference is that the interferograms are not unwrapped.

#### 5. Execute the commands in run files (run_01*, run_02*, etc) in the "run_files" folder ####



-----------------------------------
### Ionospheric Phase Estimation

Ionospheric phase estimation can be performed for each of the workflow introduced above. Generally, we should do ionospheric phase estimation for data acquired at low latitudes on ascending tracks. However, ionospheric phase estimation only works well in areas with high coherence since it requires phase unwrapping. Details about Sentinel-1 ionospheric correction can be found in

+ C. Liang, P. Agram, M. Simons, and E. J. Fielding, “Ionospheric correction of InSAR time series analysis of C-band Sentinel-1 TOPS data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 9, pp. 6755-6773, Sep. 2019.

Ionospheric phase estimation has more requirements than regular InSAR processing. The most important two requirements include

-	All the acquistions need to be connected in the entire network.

-	The swath starting ranges need to be the same among all acquistions; otherwise a phase offset between adjacent swaths need to be estimated in order to maintain consistency among the swaths, which might not be accurate enough.

#### 1. select the usable acquistions ####

In stack ionospheric phase estimation, acquistions with same swath starting ranges are put in a group. A network is formed within a group. Extra pairs are also processed to connect the different groups so that all acquistions are connected. But we need to estimate a phase offset for these extra pairs, which might not be accurate. Therefore, for a particualr swath starting ranges, if there are only a few acquistions, it's better to just discard them so that we don't have to estimate the phase offsets.

```
s1_select_ion.py -dir data/slc -sn 33.550217/37.119545 -nr 10
```

Acquistions to be used need to fully cover the south/north bounds. After running this command, acquistion not to be used will be put in a folder named 'not_used'. It's OK to run this command multiple times.

#### 2. generate configure and run files ####

In stackSentinel.py, two options are for ionospheric phase estimation

-	--param_ion
-	--num_connections_ion

An example --param_ion file 'ion_param.txt' is provided in the code directory. For example, we want to do ionospheric phase estimation when processing a stack of interferograms

```
stackSentinel.py -s ../data/slc -d ../data/dem/dem_1_arcsec/demLat_N32_N41_Lon_W113_W107.dem.wgs84 -b '33.550217 37.119545 -111.233932 -107.790451' -a ../data/s1_aux_cal -o ../data/orbit -C geometry -c 2 --param_ion ../code/ion_param.txt --num_connections_ion 3
```

If ionospheric phase estimation is enabled in stackSentinel.py, it will generate the following run files. Here ***ns*** means number of steps in the original stack processing, which depends on the type of stack (slc, correlation, interferogram, and offset).

-	run_ns+1_subband_and_resamp
-	run_ns+2_generateIgram_ion
-	run_ns+3_mergeBurstsIon
-	run_ns+4_unwrap_ion
-	run_ns+5_look_ion
-	run_ns+6_computeIon
-	run_ns+7_filtIon
-	run_ns+8_invertIon
-   run_ns+9_filtIonShift
-   run_ns+10_invertIonShift
-   run_ns+11_burstRampIon
-   run_ns+12_mergeBurstRampIon

Note about **'areas masked out in ionospheric phase estimation'** in ion_param.txt. Seperated islands or areas usually lead to phase unwrapping errors and therefore significantly affect ionospheric phase estimation. It's better to mask them out. Check ion/date1_date2/ion_cal/raw_no_projection.ion for areas to be masked out. However, we don't have this file before processing the data. To quickly get this file, we can process a stack of two acquistions to get this file. NOTE that the reference of this two-acquisition stack should be the same as that of the full stack we want to process.

**run_ns+1_subband_and_resamp**

Two subband burst SLCs are generated for each burst. For secondary acquistions, the subband burst SLCs are also resampled to match reference burst SLCs. If the subband burst SLCs already exists, the program simply skips the burst.

**run_ns+2_generateIgram_ion**

Generate subband burst interferograms.

**run_ns+3_mergeBurstsIon**

Merge subband burst interferograms, and create a unique coherence file used in ionospheric phase estimation. This will be done swath by swath if the two acquistions of the pair has different swath starting ranges.

**run_ns+4_unwrap_ion**

Unwrap merged subband interferograms. This will be done swath by swath if the two acquistions of the pair has different swath starting ranges.

**run_ns+5_look_ion**

Take further looks on the unwrapped interferograms, and create the unique coherence file based on the further number of looks. This will be done swath by swath if the two acquistions of the pair has different swath starting ranges.

**run_ns+6_computeIon**

Compute ionospheric phase. This will be done swath by swath if the two acquistions of the pair has different swath starting ranges, and then the swath ionospheric phase will be merged.

**run_ns+7_filtIon**

Filter ionospheric phase.

**run_ns+8_invertIon**

Estimate ionospheric phase for each date. We highly recommend inspecting all pair ionospheric phases ion/date1_date2/ion_cal/filt.ion, and exclude those with anomalies in this command.

Typical anomalies include dense fringes caused by phase unwrapping errors, and a range ramp as a result of errors in estimating phase offsets for pairs with different swath starting ranges (check pairs_diff_starting_ranges.txt).

#### 3. run command files generated ####

Run the commands sequentially.

#### 4. check results ####

Results from ionospheric phase estimation.

-	`reference` and `coreg_secondarys`: now contains also subband burst SLCs
-	`ion`: original ionospheric phase estimation results
    -	`date1_date2/ion_cal/azshift.ion`: azimuth ionospheric shift
    -	`date1_date2/ion_cal/filt.ion`: filtered ionospheric phase
    -	`date1_date2/ion_cal/raw_no_projection.ion`: original ionospheric phase
    -	`date1_date2/lower/merged/fine_look.unw`: unwrapped lower band interferogram
    -	`date1_date2/upper/merged/fine_look.unw`: unwrapped upper band interferogram
-   `ion_azshift_dates`: azimuth ionospheric shift for each acquistion
-   `ion_burst_ramp_dates`: azimuth burst ramps caused by ionosphere for each acquistion
-   `ion_burst_ramp_merged_dates`: merged azimuth burst ramps caused by ionosphere for each acquistion
-	`ion_dates`: ionospheric phase for each acquistion

If ionospheric phase estimation processing is swath by swath because of different swath starting ranges, there will be swath processing directories including

-	`ion/date1_date2/ion_cal_IW*`
-	`ion/date1_date2/lower/merged_IW*`
-	`ion/date1_date2/upper/merged_IW*`

After processing, we can plot ionospheric phase estimation results using plotIonPairs.py and plotIonDates.py. For example

```
plotIonPairs.py -idir ion -odir ion_plot
plotIonDates.py -idir ion_dates -odir ion_dates_plot
```

Relationships of the ionospheric phases:

```
ion_dates/date1.ion - ion_dates/date2.ion = ion/date1_date2/ion_cal/filt.ion
ion_dates/date1.ion - ion_dates/date2.ion = ionospheric phase in merged/interferograms/date1_date2/filt_fine.unw
```
