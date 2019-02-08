************
Running ISCE
************

Once everything is installed, you will need to set up a few environment variables to run the scripts included in ISCE (see :ref:`Setting_Up_Environment_Variables`\): ::

        export ISCE_HOME=<isce_directory>
        export PYTHONPATH=$ISCE_HOME/applications:$ISCE_HOME/components

where <isce_directory> is the directory specified in the *SConfigISCE* file as PRJ_SCONS_INSTALL, usually $HOME/isce

If you have installed ISCE using the installation script, you can simply source the *.isceenv* file located in the $HOME/.isce folder.

============================
Interferometry with insarApp
============================

The standard interferometric processing script is *insarApp.py*, which is invoked with the command: ::

   $ISCE_HOME/applications/insarApp.py insar.xml

where *insar.xml* (or whatever you would like to call it) contains input parameters (known as "properties") 
and names of supporting input xml files (known as "catalogs") needed to run the script.

.. warning:: Before issuing the above command, navigate first to the output folder where all the generated files will be written to.


Input Xml File
==============

The input xml file that is passed to *insarApp.py* describes the data needed to generate an interferogram, which basically are:

* a pair of images taken from the same scene, one is called *master* and the other *slave*,
* a digital elevation model (DEM) of the same area.

The input data are restricted to image products as provided by the vendor (mostly Level 0 products), accompanied by their metadata i.e., header files. ISCE supports the following sensors: ALOS, COSMO_SKYMED, ERS, ENVISAT, JERS, RADARSAT1, RADARSAT2, TERRASARX, GENERIC. The DEM is not mandatory since the application can download a suitable one from the SRTM database.

In the ISCE distribution, there is a subdirectory called *"examples/"* that contains sample xml input files specific to *insarApp.py* for several of the supported satellites. 


Describing the Input Data
-------------------------

Even though the overall structure of the xml file is fixed, the information needed to describe the input data depends on the sensor that is used.

For example, for the ALOS satellite, *insar.xml* would look as follows: ::

    <?xml version="1.0" encoding="UTF-8"?>
    <insarApp>
    <component name="insar">
        <property name="Sensor Name">
	    <value>ALOS</value>
	</property>
	<component name="Master">
	    <property name="IMAGEFILE">
	        <value>../ALOS2/IMG-HH-ALPSRP028910640-H1.0__A</value>
	    </property>
	    <property name="LEADERFILE">
                <value>../ALOS2/LED-ALPSRP028910640-H1.0__A</value>
	    </property>
	    <property name="OUTPUT">
	        <value>master.raw</value>
	    </property>
        </component>
    	<component name="Slave">
	    <property name="IMAGEFILE">
	        <value>../ALOS2/IMG-HH-ALPSRP042330640-H1.0__A</value>
	    </property>
	    <property name="LEADERFILE">
	        <value>../ALOS2/LED-ALPSRP042330640-H1.0__A</value>
	    </property>
	    <property name="OUTPUT">
	        <value>slave.raw</value>
	    </property>
        </component>
    	<component name="Dem">
            <catalog>dem.xml</catalog>
        </component>
    </component>
    </insarApp>


*insarApp* accepts the following properties in the *insar.xml* file:

+-------------------------+------+-----------------------------------------------------------------------------------+
| Property                | Note | Description                                                                       |
+=========================+======+===================================================================================+
| Sensor Name             | M    | Name of the sensor (ALOS, ENVISAT, ERS...)                                        |
+-------------------------+------+-----------------------------------------------------------------------------------+
| Doppler Method          | D    | Doppler calculation method. Can be: useDOPIQ (*default*), useCalcDop, useDoppler  |
+-------------------------+------+-----------------------------------------------------------------------------------+
| Azimuth Patch Size      | C    | Size of overlap/save patch size for formslc                                       |
+-------------------------+------+-----------------------------------------------------------------------------------+
| Number of Patches       | C    | Number of patches to process of all available patches                             |
+-------------------------+------+-----------------------------------------------------------------------------------+
| Patch Valid Pulses      | C    | Number of good lines per patch                                                    |
+-------------------------+------+-----------------------------------------------------------------------------------+
| Posting                 | D    | Pixel size of the resampled image (*default*: 15)                                 |
+-------------------------+------+-----------------------------------------------------------------------------------+
| Unwrap                  | D    | If True (*default*), performs the unwrapping                                      |
+-------------------------+------+-----------------------------------------------------------------------------------+
| useHighResolutionDemOnly| D    | If True, only download dem if SRTM high res dem is available. (*default*: False)  |
+-------------------------+------+-----------------------------------------------------------------------------------+

| **Notes:**
| **M:** This property is mandatory.
| **C:** This property is optional and its value is calculated by *insarApp*, if not specified.
| **D:** This property is optional and uses the default value, if not specified.


The following components are also accepted by the application:

+-------------------------+------+-----------------------------------------------------------------------------------+
| Property                | Note | Description                                                                       |
+=========================+======+===================================================================================+
| Master                  | S    | Description of the first image                                                    |
+-------------------------+------+-----------------------------------------------------------------------------------+
| Slave                   | S    | Description of the second image                                                   |
+-------------------------+------+-----------------------------------------------------------------------------------+
| Dem                     | O    | Description of the DEM                                                            |
+-------------------------+------+-----------------------------------------------------------------------------------+

| **Notes:**
| **S:** The Master and Slave components are mandatory. Their properties (e.g. IMAGEFILE, LEADERFILE, etc.) depend on the sensor type.
| **O:** The DEM component is optional. If not specified, the application will try to download one from the SRTM database.


Describing the DEM
------------------

The file *dem.xml* is a catalog that specifies the parameters describing a DEM which is to be used to remove the topographic phase in the interferogram. Presently ISCE supports only one format for DEM (short integer equiangular projection). The xml file should contain the following information: ::

        <component>
            <name>Dem</name>
            <property>
                <name>DATA_TYPE</name>
                <value>SHORT</value>
            </property>
            <property>
                <name>TILE_HEIGHT</name>
                <value>1</value>
            </property>
            <property>
                <name>WIDTH</name>
                <value>3601</value>
            </property>
            <property>
                <name>FILE_NAME</name>
                <value>SaltonSea.dem</value>
            </property>
            <property>
                <name>ACCESS_MODE</name>
                <value>read</value>
            </property>
            <property>
                <name>DELTA_LONGITUDE</name>
                <value>0.000833333</value>
            </property>
            <property>
                <name>DELTA_LATITUDE</name>
                <value>-0.000833333</value>
            </property>
            <property>
                <name>FIRST_LONGITUDE</name>
                <value>-117.0</value>
            </property>
            <property>
                <name>FIRST_LATITUDE</name>
                <value>34.0</value>
            </property>
        </component>

If a DEM component is given and the DEM is referenced to the EGM96 datum (which is the case for SRTM DEMs), the DEM component will be converted into WGS84 datum. A new DEM file with suffix *wgs84* is created. If the given DEM is already referenced to the WGS84 datum no conversion occurs. 

If a DEM compoenent is not given in the input file, *insarApp.py* attempts to download a suitable DEM from the publicly available SRTM database. After downloading and datum-converting the DEM, there will be two files, a EGM96 SRTM DEM with no suffix and a WGS84 SRTM DEM with the *wgs84* suffix. If no DEM component is specified and no SRTM data exists, *insarApp.py* cannot produce any geocoded or topo-corrected products.

There are a number of optional input parameters that are specifiably in the input file. They control how the processing is done. *insarApp.py* picks reasonable defaults for these, and for the most part they do not need to be set by the user. See the examples directory for specification and usage.

In order to run the interferometric application, the user is assumed to have gathered all needed data (master and slave images, with their metadata, and optionnally a DEM) and generated the xml files (*insar.xml* and, if a DEM is given, *dem.xml*). In the near future (as of July 2012), the distribution will include a script to guide the user in the generation of xml input files.


Input Arguments
===============

Alternatively, the user can choose to pass arguments and options directly in the command line when calling *insarApp.py*: ::

   python $ISCE_HOME/applications/insarApp.py LIST_OF_ARGS

where LIST_OF_ARGS is a list of arguments that will be parsed by the application.

The arguments have to be passed as a pair of key and value in this form: ``key=value`` where *key* represents the name of the attribute whose value is to be specified, e.g. ``insarApp.sensorName=ALOS``. The above xml file parameters would look like this in the command line: ::

   python $ISCE_HOME/applications/insarApp.py insarApp.sensorName=ALOS 
      insarApp.Master.imagefile=../ALOS2/IMG-HH-ALPSRP028910640-H1.0__A 
      insarApp.Master.leaderfile=../ALOS2/LED-ALPSRP028910640-H1.0__A 
      insarApp.Master.output=master.raw 
      insarApp.Slave.imagefile=../ALOS2/IMG-HH-ALPSRP042330640-H1.0__A 
      insarApp.Slave.leaderfile=../ALOS2/LED-ALPSRP042330640-H1.0__A 
      insarApp.Slave.output=slave.raw 
      insarApp.Dem.dataType=SHORT 
      insarApp.Dem.tileHeight=1 
      insarApp.Dem.width=3601 
      insarApp.Dem.filename=SaltonSea.dem 
      insarApp.Dem.accessMode=read 
      insarApp.Dem.deltaLongitude=0.000833333 
      insarApp.Dem.deltaLatitude=-0.000833333 
      insarApp.Dem.firstLongitude=-117.0 
      insarApp.Dem.firstLatitude=34.0


As it can be seen, passing all the arguments might be painstaking and requires the user to know the private name of each attribute. That is why it is recommended to use an xml file instead.

.. **Not implemented yet:**

.. It is not possible to start the application at a given step, although an option exists that directs the script to execute only some steps of the application but it is not implemented yet: insarApp.--steps=xxx
 
==============================================
Comparison Between ROI_PAC and ISCE Parameters
==============================================

The following table, valid as of July 1, 2012, shows the parameters used within ROI_PAC and their equivalents in ISCE.

.. csv-table::
   :file: property_isce_roipac.csv
   :quote: '
   :header: ROI_PAC Name, ISCE Name, Type, Description, Defaults


| **Legend:**
| \** Not yet implemented in ISCE.
| \* Implemented in ISCE, but hardcoded at a lower level; not yet exposed to user.
| N/A not applicable in ISCE

| **Types:**
| "property" is the ISCE name for an input parameter
| "component" is the ISCE name for a collection of input parameters and other components that configure a function to be performed
| "catalog" is the ISCE name for a parameter file

================
Process Workflow
================

Once the input data are ready (see previous sections), the user can run the *insarApp* application, which will generate an interferogram according to parameters given in the xml file.

The process invoked by *insarApp.py* can be broken down into several simple steps:

* `Preparing the application to run`_

* `Processing the input parameters`_

* `Preparing the data to be processed`_

* `Running the interferometric application`_


The following diagram gives an overview of the steps taken by the *insarApp* script to generate an interferogram, including the initial part under the user's control (in green).


   .. figure:: insarApp_workflow1.png
      :align: center
      :width: 1050px

   .. figure:: insarApp_workflow2.png
      :align: center
      :width: 1050px

      insarApp workflow diagram


**Convention**:
In the next sections where we describe the process more in detail, we use the following emphasis convention:

* *path/to/folder/*: path to a folder or a file (relative to $ISCE_HOME)
* *file.ext*: file name (the file path should be easily deduced from context)
* *variableName*: name of a variable used in the Python code
* ``function()``: name of a function or a method
* **Class**: class name (if not given, the name of the file that implements it should be *class.py*)


Preparing the Application to Run
================================

Once the required data have been gathered, the user can call *insarApp.py* with *insar.xml* as argument, where the xml file is an ASCII-file describing the input parameters. The python code starts by preparing the application to run while implementing all the methods needed to generate an interferogram.


Under the hood
**************

When the user issues the command: ::

   python $ISCE_HOME/applications/insarApp.py insar.xml

python starts by executing the ``__main__`` block inside *insarApp.py*. The first line of that block creates an **Insar** object called *insar*: ::

   insar = Insar()

.. note:: The above command creates an instance of the **Insar** class (also known as an **Insar** object) and calls its ``__init__()`` method.

The **Insar** class, defined in *insarApp.py*, is a child class of **Application** that inherits from **Component**, which in turn derives from **ComponentInit**. Hence, when instantiated through its method ``__init__()``, *insar* has all the properties and methods of its ancestors.

An object *_insar* of type **InsarProc** is then added to *insar*: ::

   self._insar = InsarProc.InsarProc()

That object holds the properties, along with the methods (setters and getters) to modify and return their values, which will be useful for the interferometric process.

Using the **RunWrapper** class and the functions defined in *Factories.py*, the application will then wrap all the methods needed to run *insar*, e.g.: ::

   self.runPreprocessor = InsarProc.createPreprocessor(self)

.. note:: The above command calls the function ``createPreprocessor()``, found in *Factories.py* (imported by *__init__.py* inside *components/isceobj/InsarProc/*). It takes the function ``runPreprocessor()`` defined in *components/isceobj/InsarProc/runPreprocessor.py* and attaches it to the object *insar* by means of a **RunWrapper** object. Now, *insar* has an attribute called *runPreprocessor* which is linked to a function also called ``runPreprocessor()``.

The methods thus defined become methods of *insar* and will be called later, directly from *insar*, to process the data.

Once the initialization is done, the code calls the method ``run()`` defined in **Application**, *insar*'s parent class: ::

   insar.run()


Processing the Input Parameters
===============================

After the initialization of the application, the command line is processed to extract the argument(s) passed to *insarApp.py*. The application needs parameters to be given in order to run. Those input parameters can be passed directly in the command line or via an xml file (called e.g. *insar.xml*) and are used to initialize the properties and the facilities of the application.

.. note:: Only xml files are supported in the current distribution.


Under the hood
**************

The command line is processed by **Application**'s method ``_processCommandLine()``, which gets the command line and parses it through the method ``commandLineParser()`` of a **Parser** object *PA*. Since the passed argument refers to an xml file, *PA* calls the method ``parse()`` of an **XmlParser** instance.

The parsing is facilitated by the ElementTree XML API (module xml.etree.ElementTree) which reads the xml file and stores its content in an **ElementTree** object called *root*. *root* is then parsed recursively by a **Parser** object to extract the components and the properties inside the file (``parseComponent()``, ``parseProperty()``). When done, we get a dictionary called *catalog*, containing a cascading set of dictionaries with all the properties included in the xml file(s). In our example, *catalog*'s content would look like this: ::

   { 'sensor name': 'ALOS', 
     'Master': { 'imagefile': '../ALOS2/IMG-HH-ALPSRP028910640-H1.0__A',
                 'leaderfile': '../ALOS2/LED-ALPSRP028910640-H1.0__A',
      	       	 'output': 'master.raw' },
     'Slave': { 'imagefile': '../ALOS2/IMG-HH-ALPSRP042330640-H1.0__A',
                'leaderfile': '../ALOS2/LED-ALPSRP042330640-H1.0__A',
	        'output': 'slave.raw' },
     'Dem': { 'data_type': 'SHORT', 
              'tile_height': 1, 
      	      'width': 3601, 
      	      'file_name': 'SaltonSea.dem',
      	      'access_mode': 'read'
      	      'delta_longitude': 0.000833333,
	      'delta_latitude': -0.000833333,
	      'first_longitude': -117.0,
	      'first_latitude': 34.0 } }

Then, the application parameters are defined through *insar*'s method ``_parameters()``: those are the parameters that can be configured in the input xml file. For each parameter, the following information is needed: private name (known to the application only), public name (disclosed to the user), type (int, string, etc.), units, default value (if parameter is omitted), mandatoriness (the parameter must be present in the xml file or not), description. Each and everyone of those parameters are represented by an attribute of the *insar* object, whose name is the parameter's private name and whose value is given by the parameter's default value. Also, we end up with several dictionaries (*descriptionOfVariables*, *typeOfVariables* and *dictionaryOfVariables*) and lists (*mandatoryVariables* and *optionalVariables*) that help organize the parameters according to their characteristics.

With the configurable parameters thus defined, the code calls ``initProperties()`` which checks *catalog*'s content and assigns the user's values to the given parameters.

Then, the application facilities are defined through *insar*'s method ``_facilities()``. Facilities are objects whose class can only be determined when the code reads the user's parameters. Their nature cannot be hardcoded in advance, so that they will be created by the code at runtime using modules called factories. For *insarApp.py*, those facilities are *master* (master sensor), *slave* (slave sensor), *masterdop* (master doppler), *slavedop* (slave doppler) and *dem*. For each facility, the following information is needed: private name (known to the application only), public name (disclosed to the user), module (package where the factory is present), factory (name of the method capable of creating the facility), args and kwargs (additional arguments that the factory might need in order to create the facility), mandatoriness, description. Each and everyone of those facilities are represented by an attribute of the *insar* object, whose name is the facility's private name and whose value is an object of class **EmpytFacility**. The *dictionaryOfFacilities* is updated to reflect the list of facilities that can be configured in the input xml file.

Finally, the facilities are given their actual type and properties according to the user's parameters, with the method ``_processFacilities()``.



Preparing the Data to Be Processed
==================================

The application needs to read and ingest the pair of image products with their header files and the given doppler method to produce raw data which will be processed later. If a dem has not been given, the application proceeds to download one from the SRTM database (make sure that you have an internet connection).


Under the hood
**************

At this step, ``run()`` executes *insar*'s ``main()`` method which calls ``help()`` to output an initial message about the application and creates a **Catalog** object for logging purposes: ::

   self.insarProcDoc = isceobj.createCatalog('insarProc')

The current time is also recorded in order to assess the duration of the following steps, the first of which is ``runPreprocessor()``.

``runPreprocessor()`` takes the four input facilities (*master*, *slave*, *masterdop* and *slavedop*) and generates one raw image for each pair of **Sensor** and **Doppler** objects: *master.raw* and *slave.raw* (the output names can be configured in the xml file). First, ``runPreprocessor()`` passes the pair *master*/*masterdop* to a ``make_raw()`` method - to avoid confusion, let's call it ``insar.make_raw()``, which returns a **make_raw** object. To do that, ``insar.make_raw()`` creates a **make_raw** object (whose class is defined in *applications/make_raw.py*), wires the pair of facilities as input ports to that object and executes its ``make_raw()`` method - called ``make_raw.make_raw()`` to avoid confusion.

``make_raw.make_raw()`` starts by activating the **make_raw** object's ports, i.e., adding *master* as its sensor attribute and *masterdop* as its doppler attribute. Then, it extracts the raw data from *master*. Here it is assumed that each supported sensor has implemented a method called ``extractImage()``. For example, the **ALOS** class, defined in *components/isceobj/Sensor/ALOS.py*, expects four parameters, of which three are mandatory, to be given in the input xml file: IMAGEFILE, LEADERFILE, OUTPUT and RESAMPLE_FLAG (optional). ``extractImage()`` parses the *leaderfile* and the *imagefile*, extracts raw data to *output* (with resampling or not), creates the appropriate metadata objects with ``populateMetadata()`` (**Platform**, **Instrument**, **Frame**, **Orbit**, **Attitude** and **Distortion**) and generates a *.aux* file (*master.raw.aux*) with ``readOrbitPulse()``. Once the raw data has been extracted, ``make_raw.make_raw()`` calculates the doppler values and fits a polynomial to those values by calling *masterdop*'s method ``calculateDoppler()`` and ``fitDoppler()``. The Doppler polynomial coefficients and the pulse repetition frequency are then transferred to a **Doppler** object called *dopplerValues*. The spacecraft height and height_dt (``calculateHeighDt()``), velocity (``calculateVelocity()``) and squint angle (``calculateSquint()``) are also computed whereas the sensing start is adjusted according to values in the pulse timing *.aux* file (``adjustSensingStart()``).

Most of the attributes in the **make_raw** object are copied to a **RawImage** object, called *masterRaw*: filename, Xmin, Xmax, number of good bytes (Xmax - Xmin), width (Xmax). The same steps are done with the pair *slave*/*slavedop* as well. Finally, the following values are assigned to *_insar*'s attributes: *_masterRawImage*, *_slaveRawImage*, *_masterFrame*, *_slaveFrame*, *_masterDoppler*, *_slaveDoppler*, *_masterSquint*, *_slaveSquint*.

Once ``runPreprocessor()`` has been executed, *insar*'s ``main()`` method checks if a dem has been given. If not, it assesses the common geographic area between the master and slave frames, taking into account the master and slave squint angles, with the method ``extractInfo()``. Then, ``createDem()`` downloads a DEM from the STRM database, generates an xml file and creates a **DemImage** object assigned to *_insar* as *_demImage*.



Running the Interferometric Application
=======================================

Now that all the data and metadata are ready to get processed, we can proceed to the core of the interferometric application with the following steps:

A. data focussing
B. interferogram building
C. interferogram refining
D. coherence computing
E. filter application
F. phase unwrapping
G. geocoding


Under the hood
**************

A. Data Focussing

   1) *runPulseTiming*

      This wrapper is linked to the method ``runPulseTiming()`` which generates an interpolated orbit for each image (master and slave).

      From the master frame, the method ``pulseTiming()`` generates an **Orbit** object containing a list of **StateVector** objects - one for each range line in the frame. The state vectors are interpolated from the original orbit, using the Hermite interpolation scheme (a C code). The satellite's position and velocity are evaluated at the time of each pulse.

      Idem for the slave frame.

      The pair of pulse **Orbit** objects generated are assigned to *_insar* as *_masterOrbit* and *_slaveOrbit*.

   2) *runEstimateHeights*

      This wrapper is linked to the method ``runEstimateHeights()`` which calculates the height and the velocity of the platform for each image (master and slave).

      For the master image (and then for the slave image), the code starts by instantiating a **CalcSchHeightVel** object using the function ``createCalculateFdHeights()`` defined in *components/stdproc/orbit/__init__.py*. The **CalcSchHeightVel** class is defined in *components/stdproc/orbit/orbitLib/CalcSchHeightVel.py*. Three input ports are wired to the **CalcSchHeightVel** object: *_masterFrame* (*_slaveFrame* for the slave image), *_masterOrbit* (*_slaveOrbit*) and *planet*. *planet* is extracted from *_masterFrame*. The **CalcSchHeightVel** object's method ``calculate()`` is then called, computing the height and the velocity of the platform.

      The computed master and slave heights are assigned to *_insar* as *_fdH1* (with ``setFirstFdHeight()``) and *_fdH2* (with ``setSecondFdHeight()``) respectively.

   3) *runSetmocomppath*

      This wrapper is linked to the method ``runSetmocomppath()`` which selects a common motion compensation path for both images.

      The method begins with the instantiation of a **Setmocomppath** object using the function ``createSetmocomppath()`` found in *components/stdproc/orbit/__init__.py*. The **Setmocomppath** class is defined in *Setmocomppath.py*, located in the same folder. Three input ports are wired to the **Setmocomppath** object: *planet*, *_masterOrbit* and *_slaveOrbit*. Then, the method ``setmocomppath()`` of that object is executed: using a Fortran code, it takes the pair of orbits and picks a motion compensation trajectory. It returns a **Peg** object (representing a peg point with the following information: longitude, latitude, heading and radius of curvature), which is the average of the two peg points computed from the master orbit and the slave orbit. It gives also the average height and velocity of each platform.

      The computed peg, average heights and velocities are assigned to *_insar* as *_peg*, *_pegH1* (with ``setFirstAverageHeight()``), *_pegH2* (with ``setSecondAverageHeight()``), *_pegV1* (with ``setFirstProcVelocity()``) and *_pegV2* (with ``setSecondProcVelocity()``).

   4) *runOrbit2sch*

      This wrapper is linked to the method ``runOrbit2sch()`` which converts the orbital state vectors of the master and slave orbits from xyz to sch coordinates.

      For the master orbit (and then for the slave orbit), the method starts by instantiating an **Orbit2sch** object using the function ``createOrbit2sch()`` found in *components/stdproc/orbit/__init__.py*. The **Orbit2sch** class is defined in *Orbit2sch.py*, located in the same folder. The mean value of *_pegH1* and *_pegH2* (first and second average heights) is assigned to the **Orbit2sch** object while three input ports are wired to it: *planet*, *_masterOrbit* (*_slaveOrbit* for the slave image) and *_peg*. Then, the ``orbit2sch()`` method converts the coordinates of the orbit into the sch coordinate system, using a Fortran code. It returns an **Orbit** object with a list of **StateVector** objects whose coordinates are now in sch.

      The two newly-computed orbits replace *_masterOrbit* and *_slaveOrbit* in *_insar*.

   5) *updatePreprocInfo*

      This wrapper is linked to the method ``runUpdatePreprocInfo()`` that calls ``runFdMocomp()`` to calculate the motion compensation correction for Doppler centroid: here, it returns *fd*, the average correction for *masterOrbit* and *slaveOrbit*. *fd* is used as the fractional centroid of *averageDoppler*, which is the average of *_masterDoppler* and *_slaveDoppler* (the doppler centroids previously calculated in ``runPreprocessor()``).

      *averageDoppler* is then assigned to *_insar* as *_dopplerCentroid*.

   6) *runFormSLC*

      This wrapper is linked to the method ``runFormSLC()`` which focuses the two raw images using a range-doppler algorithm with motion compensation.

      For the master raw image (and then for the slave raw image), the method starts by instantiating a **Formslc** object using the function ``createFormSLC()`` found in *components/stdproc/stdproc/formslc/__init__.py*.  The **Formslc** class is defined in *Formslc.py*, located in the same folder. Seven input ports are wired to the **Formslc** object: *_masterRawImage* (*_slaveRawImage*), *masterSlcImage* (*slaveSlcImage*), *_masterOrbit* (*_slaveOrbit*), *_masterFrame* (*_slaveFrame*), *planet*, *_masterDoppler* (*_slaveDoppler*) and *_peg*. The spacecraft height is set to the mean value of *_fdH1* (first Fd Height) and *_fdH2* (second Fd Height), and its velocity to the mean value of *_pegV1* (first Proc Velocity) and *_pegV2* (second Proc Velocity). The method ``formslc()`` of the **Formslc** object is then called, which generates a *.slc* file (*master.slc* and *slave.slc*).

      The two generated **SlcImage** objects are assigned to *_insar* as *_masterSlcImage* and *_slaveSlcImage*, along with *_patchSize*, *_numberValidPulses* and *_numberPatches*. The two **Formslc** objects used to generate the slcs are also assigned to *_insar* as *_formSLC1* and *_formSLC2*.


B. Interferogram Building

   7) *runOffsetprf*

      This wrapper is linked to the method ``runOffsetprf()`` which calculates the offset between the two slc images.

      It starts by instantiating an **Offsetprf** object using the function ``createOffsetprf()`` found in *components/isceobj/Util/__init__.py*. The **Offsetprf** class is defined in *Offsetprf.py*, located in the same folder. The method ``offsetprf()`` of the **Offsetprf** object is then called, with *_masterSlcImage* and *_slaveSlcImage* passed as arguments. It returns, via a Fortran code, an **OffsetField** object which compiles a list of **Offset** objects, each describing the coordinates of an offset, its value in both directions (across and down) and the signal-to-noise ratio (SNR).

      The computed **OffsetField** object is assigned to *_insar* twice: as *_offsetField* and *_refinedOffsetField*.

   8) *runOffoutliers*

      This wrapper is linked to the method ``runOffoutliers()`` which culls outliers from the previously computed offset field. The offset field is approximated by a best fitting plane, and offsets are deemed to be outliers if they are greater than a user selected distance.

      It is executed three times with a *distance* value set to 10, 5 then 3 meters. For each iteration, it makes use of an **Offoutliers** object, created by the function ``createOffoutliers()`` found in *components/isceobj/Util/__init__.py*. The **Offoutliers** class is defined in *Offoutliers.py*, located in the same folder. One input port is wired to the **Offouliers** object: *_refinedOffsetField*. The SNR is fixed to 2.0 while the distance is the value set at each iteration. The method ``offoutliers()`` of the **Offoutliers** object is then called and returns a new **OffsetField** object, replacing *_refinedOffsetField* in *_insar*.

   9) *prepareResamps*

      This wrapper is linked to the method ``runPrepareResamps()`` which calculates some parametric values for resampling (slant range pixel spacing, number of azimuth looks, number of range looks, number of resamp lines) and fixes the number of fit coefficients to 6.

   10) *runResamp*
   
      This wrapper is linked to the method ``runResamp()`` which resamples the interferogram based on the provided offset field.

      It begins with the instantiation of a **Resamp** object, using the function ``createResamp()`` found in *components/stdproc/stdproc/resamp/__init__.py*. The **Resamp** class is defined in *Resamp.py*, located in the same folder. Two input ports are wired to the **Resamp** object: *_refinedOffsetField* and *instrument*, along with some more parameters. The method ``resamp()`` is then called with four input arguments: the two slcs as well as **AmpImage** and **IntImage** objects. Through a Fortran code, the two slcs are coregistered and processed to form an interferogram that is then multilooked according to the values calculated in the previous step. Two files are generated: *resampImage.amp* and *resampImage.int*.

      The **AmpImage** and **IntImage** objects are assigned to *_insar* as *_resampAmpImage* and *_resampIntImage*.

   11) *runResamp_image*

      This wrapper is linked to the method ``runResamp_image()`` which plots the offsets as an image.

      It begins with the instantiation of a **Resamp_image** object using the function ``createResamp_image()`` found in *components/stdproc/stdproc/resamp_image/__init__.py*. The **Resamp_image** class is defined in *Resamp_image.py*, located in the same folder. Two input ports are wired to the **Resamp_image** object: *_refinedOffsetField* and *instrument*, along with some more parameters. Then, the method ``resamp_image()`` of that object is called with two **OffsetImage** objects as arguments (one accross and one down): using a Fortran code, that method takes the offsets and plots them as an image, generating two files: *azimuthOffset.mht* and *rangeOffset.mht*.

      The accross **OffsetImage** and down **OffsetImage** objects are assigned to *_insar* as *_offsetRangeImage* and *_offsetAzimuthImage* respectively.

   12) *runMocompbaseline*

      This wrapper is linked to the method ``runMocompbaseline()`` which calculates the mocomp baseline. It iterates over the S-component of the master image and interpolates linearly the SCH coordinates at the corresponding S-component in the slave image. The difference between the master SCH coordinates and the slave SCH coordinates provides a 3-D baseline.

      It begins with the instantiation of a **Mocompbaseline** object using the function ``createMocompbaseline()`` found in *components/stdproc/orbit/__init__.py*. The **Mocompbaseline** class is defined in *Mocompbaseline.py*, located in the same folder. Four input ports are wired to the **Mocompbaseline** object: *_masterOrbit*, *_slaveOrbit*, *ellipsoid* and *_peg*, along with some more parameters. Then, the method ``mocompbaseline()`` of that object is called: using a Fortran code, that method gets the insar baseline from mocomp and position files, updating the properties of the **Mocompbaseline** object.

      The **Mocompbaseline** object is assigned to *_insar* as *_mocompBaseline*.

   13) *runTopo*

       This wrapper is linked to the method ``runTopo()`` which approximates the topography for each pixel of the interferogram.

       At this step, *_resampIntImage* is duplicated as *_topoIntImage* inside *_insar*. Then the code starts by instantiating a **Topo** object using the function ``createTopo()`` found in *components/stdproc/stdproc/topo/__init__.py*. The **Topo** class is defined in *Topo.py*, located in the same folder. Five input ports are wired to the **Topo** object: *_peg*, *_masterFrame*, *planet*, *_demImage* and *_topoIntImage*, along with some more parameters. Then, the method ``topo()`` of that object is called: using a Fortran code, it approximates the topography and generates temporary files giving, for each pixel, the following values: latitude (*lat*), longitude (*lon*), height in SCH coordinates (*zsch*), real height in XYZ coordinates (*z*) and the height in XYZ coordinates rounded to the nearest integer (*iz*).

       The **Topo** object is assigned to *_insar* as *_topo*.

   14) *runCorrect*

       This wrapper is linked to the method ``runCorrect()`` which carries out a flat earth correction of the interferogram.

       It starts by instantiating a **Correct** object using the function ``createCorrect()`` found in *components/stdproc/stdproc/correct/__init__.py*. The **Correct** class is defined in *Correct.py*, located in the same folder. Four input ports are wired to the **Correct** object: *_peg*, *_masterFrame*, *planet*, and *_topoIntImage*, along with some more parameters. Then, the method ``correct()`` of that object is called: using a Fortran code, it reads the interferogram and the SCH height file, and removes the topography phase from the interferogram. It generates two files: *topophase.flat* (the flattened interferogram) and *topophase.mph* (the topography phase).
       
   15) *runShadecpx2rg*

       This wrapper is linked to the method ``runShadecpx2rg()`` which combines a shaded relief from the DEM in radar coordinates and the SAR complex magnitude image into a single two-band image.

      It begins with the instantiation of a **Shadecpx2rg** object using the function ``createShadecpx2rg()`` found in *components/isceobj/Util/__init__.py*. The **Shadecpx2rg** class is defined in *Shadecpx2rg.py*, located in the same folder. After initializing some parameters, the method ``shadecpx2rg()`` of that object is called with four arguments: a **DemImage** object referencing the height file (*iz*), an **IntImage** object referencing the resampled amplitude image (*resampImage.amp*), an **IntImage** object referencing the Rg Dem image to be written (*rgdem*) and a shade factor equal to 3. Using a Fortran code, that method computes, for each pixel, a shade value and multiplies that factor to the magnitude value. It generates a file called *rgdem*.

      The **RgImage** object referencing the file *rgdem* and the **DemImage** referencing the file *iz* are assigned to *_insar* as *_rgDemImage* and *_heightTopoImage* respectively.


C. Interferogram Refining

   16) *runRgoffset*

       This wrapper is linked to the method ``runRgoffset()`` which estimates the subpixel offset between two images stored as one rg file.

       It starts by instantiating an **Rgoffset** object using the function ``createRgoffset()`` found in *components/isceobj/Util/__init__.py*. The **Rgoffset** class is defined in *Rgoffset.py*, located in the same folder. After initializing some parameters, the method ``rgoffset()`` of that object is called with an **RgImage** object as argument, referencing the file *rgdem*.

       It generates an **OffsetField** object that is assigned to *_insar*, replacing *_offsetField* and *_refinedOffsetField*.

   17) *runOffoutliers*

       See step 8. This method culls outliers from the offset field. It is executed three times with a *distance* value set to 10, 5 then 3 meters.

   18) *runResamp_only*

       This wrapper is linked to the method ``runResamp_only()`` which resamples the interferogram.

       It begins with the instantiation of a **Resamp_only** object using the function ``createResamp_only()`` found in *components/stdproc/stdproc/resamp_only/__init__.py*. The **Resamp_only** class is defined in *Resamp_only.py*, located in the same folder. Two input ports are wired to the **Resamp_only** object: *_refinedOffsetField* and *instrument*, along with some more parameters. Then, the method ``resamp_only()`` of that object is called with two **IntImage** objects as arguments (one referencing the resampled interferogram *resampImage.int* to be read, and the other referencing a file called *resampOnlyImage.int* to be written): using a Fortran code, that method takes the interferogram and resamples it to coordinates set by offsets (*_refinedOffsetField*), generating a file called *resampOnlyImage.int*.

       The **IntImage** object referencing the file *resampOnlyImage.int* is assigned to *_insar* as *_resampOnlyImage*.

   19) *runTopo*

       At this step, *_resampOnlyImage* is duplicated as *_topoIntImage* inside *_insar*. Then the code approximates the topography as in step 13.

   20) *runCorrect*

       See step 14.


D. Coherence Computation

   21) *runCoherence*

       This wrapper is linked to the method ``runCoherence()`` which calculates the interferometric correlation.

       It starts by instantiating a **Correlation** object, whose class is defined in *components/mroipac/correlation/correlation.py*. Two input ports are wired to that object: an **IntImage** referencing the file *topophase.flat* and an **AmpImage** object referencing the amplitude image *resampImage.amp*. One output port is also wired to that object: an **OffsetImage** object referencing a file called *topophase.cor* to be written. Then, one of the **Correlation** object's methods is executed: ``calculateEffectiveCorrelation()`` if method is 'phase_gradient', or ``calculateCorrelation()`` if method is 'cchz_wave'. Both rely on C codes to calculate the interferometric correlation.

       Here the default method is 'phase_gradient': the script executes ``calculateEffectiveCorrelation()``. That method uses the phase gradient to calculate the effective correlation:

       * First, ``phase_slope()`` is called to calculate the phase gradient. It takes nine arguments: the interferogram filename (*topophase.flat*), the phase gradient filename (a temporary file to be written), the number of samples per row (interferogram width), the size of the window for the gradient calculation (default: 5), the gradient threshold for phase gradient masking (default: 0), the starting range pixel offset (0), the last range pixel offset (-1), the starting azimuth pixel offset (0) and the last azimuth pixel offset (-1).

       * Then, ``phase_mask()`` is called to create the phase gradient mask. It takes eleven arguments: the interferogram filename, the phase gradient filename (the temporary file previously created), the phase standard deviation filename (a temporary file to be written), the standard deviation threshold for phase gradient masking (default: 1), the number of samples per row, the range and azimuth smoothing window for the phase gradient (default: 5x5), the starting/last range/azimuth pixel offsets.

       * Finally, ``magnitude_threshold()`` is called to threshold the phase file using the magnitude values in the coregistered interferogram. It takes five arguments: the interferogram filename, the phase standard deviation filename (the temporary file previously created), the output filename (*topophase.cor*), the magnitude threshold for phase gradient masking (default: 5e-5) and the number of samples per row.

       The other method ``calculateCorrelation()`` uses the maximum likelihood estimator to calculate the correlation. It calls ``cchz_wave()`` which takes nine arguments: the interferogram filename, the amplitude filename (*resampImage.amp*), the output correlation filename (*topophase.cor*), the width of the interferogram file, the width of the triangular smoothing function (default: 5 pixels), the starting/last range/azimuth pixel offsets. 


E. Filter Application

   22) *runFilter*

       This wrapper is linked to the method ``runFilter()`` which applies the Goldstein-Werner power-spectral filter to the flattened interferogram.

       It starts by instantiating a **Filter** object, whose class is defined in *components/mroipac/filter/Filter.py*. One input port and one output port are wired to that object: an **IntImage** referencing the flattened interferogram (*topophase.flat*), and another **IntImage** object referencing the filtered interferogram to be created (*filt_topophase.flat*), respectively. Then, the method ``goldsteinWerner()`` is called with an argument *alpha*, representing the strength of the Goldstein-Werner filter (default: 0.5). That method applies a power-spectral smoother to the phase of the interferogram:
       
       * First, separate the magnitude and phase of the interferogram and save both bands.

       * Second, apply the power-spectral smoother to the original interferogram.

       * Third, take the phase regions that were zero in the original image and apply them to the smoothed phase.

       * Fourth, combine the smoothed phase with the original magnitude, since the power-spectral filter distorts the magnitude.  

       The first steps are done with the method ``psfilt()`` while the last one is done with the method ``rescale_magnitude()``. Both methods are based on C code.

       Now *_topophaseFlatFilename* in *_insar* is set to *filt_topophase.flat*.


F. Phase Unwrapping

   23) *runGrass*

       This wrapper is linked to the method ``runGrass()`` which unwraps the filtered interferogram using the grass algorithm.

       This step is executed only if required by the user in the xml file. It starts by instantiating a **Grass** object, whose class is defined in *components/mroipac/grass/grass.py*. Two input ports are wired to that object: an **IntImage** referencing the filtered interferogram (*filt_topophase.flat*) and an **OffsetImage** object referencing the coherence image to be created (*filt_topophase.cor*). One output port is also wired to the **Grass** object: an **IntImage** object referencing the unwrapped interferogram to be created (*filt_topophase.unw*). Then, the method ``unwrap()`` is called:

       * First, it creates a flag file for masking out the areas of low correlation (default threshold: 0.1) calling the following C functions: ``residues()``, ``trees()`` and ``corr_flag()``.

       * Then, it unwraps the interferogram using the grass algorithm with the C function ``grass()``.


G. Geocoding

   24) *runGeocode*

       This wrapper is linked to the method ``runGeocode()`` which generates a geocoded interferogram.

       It begins with the instantiation of a **Geocode** object using the function ``createGeocode()`` found in *components/stdproc/rectify/__init__.py*. The **Geocode** class is defined in *Geocode.py*, located in the subfolder *components/stdproc/rectify/geocode/*. Five input ports are wired to the **Geocode** object: *_peg*, *_masterFrame*, *planet*, *_demImage* and an *IntImage* object referencing the filtered interferogram (*filt_topophase.flat*), along with some more parameters. Then, the method ``geocode()`` of that object is called: using a Fortran code, that method takes the interferogram and orthorectifies it (i.e., correcting its geometry so that it can fit a map with no distortions).

      Two files are generated at this step: a geocoded interferogram (*topophase.geo*) and a cropped dem (*dem.crop*).


After the interferometric process is done, the application stops the timer and returns the total time required to finish all the operations. Finally, it dumps all the metadata about the process into an *insarProc.xml* file: ::

   self.insarProcDoc.renderXml()
