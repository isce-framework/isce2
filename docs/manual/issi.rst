============================
Ionospheric Faraday Rotation
============================

Background
**********

.. comment (Old title) Ionospheric Specifications for SAR Interferometry (ISSI)}

Motivation
----------

Inhomogeneities in ionospheric structure such as plasma irregularities
lead to distortions in low frequency (L-band and lower) Synthetic
Aperture Radar (SAR) images [FreSa04]_. These inhomogeneities
hamper the interpretation of ground deformation when these SAR images
are combined to form interferograms. To mitigate the effects of these
distortions, [Pi12]_ outlined a methodology for the estimation
and removal of ionospheric artifacts from individual, fully-polarimetric
SAR images. The estimation methodology also provides a way to create
temporal snapshots of ionospheric behavior with large (40km) spatial
extent. To demonstrate these capabilities for fully polarimetric spaceborne
SAR to provide images of the ionosphere, we have developed computer
software to implement the methodology outlined in [Pi12]_.


Methodology behind ISSI
-----------------------

The measured scattering matrix from an L-band polarimetric SAR can
be written as:

.. math::
        :label: measuredScatteringMatrix

        M=A(r,\theta)e^{i\phi}R^{T}R_{F}SR_{F}T+N,

with

.. math::
        :nowrap:

        \begin{eqnarray*}
        M & = & \left(\begin{array}{cc}
        M_{hh} & M_{hv}\\
        M_{vh} & M_{vv}
        \end{array}\right)\\
        R^{T} & = & \left(\begin{array}{cc}
        1 & \delta_{1}\\
        \delta_{2} & f_{1}
        \end{array}\right)\\
        R_{F} & = & \left(\begin{array}{cc}
        \cos\Omega & \sin\Omega\\
        -\sin\Omega & \cos\Omega
        \end{array}\right)\\
        S & = & \left(\begin{array}{cc}
        S_{hh} & S_{hv}\\
        S_{vh} & S_{vv}
        \end{array}\right)\\
        T & = & \left(\begin{array}{cc}
        1 & \delta_{3}\\
        \delta_{4} & f_{2}
        \end{array}\right)
        \end{eqnarray*}

where :math:`S_{ij}` is the scattering matrix, :math:`A(r,\theta)` is the gain
of the radar as a function of range and elevation angle, :math:`\delta_{i}`
are the cross-talk parameters, :math:`f_{i}` are the channel imbalances,
and :math:`\Omega` is the Faraday rotation [Fre04]_. By rearranging
equation :eq:`measuredScatteringMatrix`, we can apply the cross-talk
and channel imbalance corrections while preserving the effects of
Faraday rotation, yielding.

.. math::

        M'=R_{F}SR_{F}=\frac{1}{f_{1}-\delta_{1}\delta_{2}}\frac{1}{f_{1}-\delta_{3}\delta_{4}}\left(\begin{array}{cc}
        f_{1} & -\delta_{2}\\
        -\delta_{1} & 1
        \end{array}\right)M\left(\begin{array}{cc}
        f_{2} & -\delta_{3}\\
        -\delta_{4} & 1
        \end{array}\right).

We can now use :math:`M'`, the partially-polarimetrically calibrated data
matrix, to estimate the Faraday rotation using the method outlined
in [Bick65]_. We begin by transforming :math:`M'` into a circular
basis, yielding,

.. math::
        :label: circularBasis

        \left(\begin{array}{cc}
        Z_{11} & Z_{12}\\
        Z_{21} & Z_{22}
        \end{array}\right)=\left(\begin{array}{cc}
        1 & i\\
        i & 1
        \end{array}\right)M'\left(\begin{array}{cc}
        1 & i\\
        i & 1
        \end{array}\right).

The Faraday rotation can then be calculated as, 

.. math::
        :label: faradayRotation

        \Omega=\arg(Z_{12}Z{}_{21}^{*})

Given a measurement of Faraday rotation and an estimate of the strength
of the Earth's magnetic B-field, one can then estimate the Total Electron
Count (TEC) using the relationship,

.. math::
        :label: TECIntegral

        \int n_{e}B\cos\theta ds=\frac{\Omega f^{2}}{k},

where :math:`k=\frac{\left|e\right|^{3}}{8\pi^{2}c\epsilon_{0}m{}_{e}^{2}}`
with :math:`e` being the elementary charge, :math:`c` is the speed of light
:math:`\epsilon_{0}` is the permittivity of free space, :math:`m{}_{e}` is the
electron mass, :math:`n_{e}` is the electron density, :math:`\theta` is the
angle between the SAR signal propagation direction and the B-field
and :math:`f` is the carrier frequency of the radar. Since the the angle
$\theta$, does not change much along the path through the ionosphere,
we can move the :math:`B\cos\theta` term out of the integral in equation
:eq:`TECIntegral`. This allows us to rewrite equation :eq:`TECIntegral` as,

.. math::
        :label: TEC

        TEC=\frac{\Omega f^{2}}{kB\cos\theta}

where :math:`TEC=\int n_{e}ds`.

Ideally, we would calculate the strength of the Earth's B-field along
the path from the radar to the ground at each pixel in the SAR image.
Since, at the scale of a typical SAR image, the B-field is smoothly
varying, we will make the assumption that we can approximate the effect
of the magnetic field by using the *average* B-field value
over the area of the SAR image. Additionally, we will make the assumption
that B-field is homogeneous enough to allow us to replace the line-of-sight
path integration with a vertical integration through the ionosphere.
This is an assumption that can easily be changed in the future. We
begin by calculating the geographic coordinates of the corners of
our SAR image. Then, we estimate the total strength of the magnetic
B-field in the direction of the radar line-of-sight in a vertical
column above each geographic location. The average total strength
of the magnetic B-field in radar line-of-sight is then used to calculate
the TEC at each pixel in the SAR image using equation :eq:`TEC`.

Finally, the phase change contribution to the SAR image from the Faraday
rotation can be calculated using the estimate of TEC found from equation
:eq:`TEC` as,

.. math::
        :nowrap:

        \begin{eqnarray*}
        \phi_{I} & = & -\frac{\omega}{2c}\int Xds\\
         & = & -\frac{2\pi}{c}\frac{e^{2}}{8\pi^{2}\epsilon_{0}m_{e}f}\int n_{e}ds\\
         & = & \frac{8.45\times10^{-7}}{f}TEC
        \end{eqnarray*},

where :math:`X=\frac{\omega_{p}}{\omega}`, and :math:`\omega_{p}=\left(\frac{n_{e}e^{2}}{\epsilon_{0}m_{e}}\right)^{\frac{1}{2}}`
is the angular plasma frequency. This value can be calculated at each
pixel in the SAR image.


Running ISSI
************

SAR data can be acquired from ground processing facilities as raster images of
focused or unfocused radar echos. ISSI can accept either data format and
produce images of Faraday rotation, TEC and phase delay.
The most straightforward application of the ISSI methodology begins with
focused and aligned SAR data, which typically comes in the form of single-look
complex (SLC) images. These images are first rotated into a circular basis
using equation :eq:`circularBasis`, and an estimate of Faraday rotation is
formed using equation :eq:`faradayRotation`. TEC and phase delay are then
calculated using subsequent results.

When beginning with unfocused radar echos, we must first prepare SLC images,
taking care to focus the radar echos using the same Doppler parameters for each
transmit and receive polarity combination. Once SLC's have been produced, we
must align each SLC by resampling the SAR data transmitted with vertical
polarization such that it lies on the same pixel locations as the SAR data
transmitted with horizontal polarization. Once these steps have been completed,
we may proceed as before in converting the images to a circular basis and
forming Faraday rotation, TEC and phase delay images.

As input to the ISSI scripts, we require a set of XML files.  Begin by creating
a file called *FR.xml* and put the following information in it::

        <component>
          <property>
                  <name>HH</name>
                  <factoryname>createALOS</factoryname>   
                  <factorymodule>isceobj.Sensor</factorymodule>
                  <value>HH.xml</value>  
          </property>
          <property>
                  <name>HV</name>
                  <factoryname>createALOS</factoryname>
                  <factorymodule>isceobj.Sensor</factorymodule>
                  <value>HV.xml</value>
          </property>
          <property>
                  <name>VH</name>
                  <factoryname>createALOS</factoryname>
                  <factorymodule>isceobj.Sensor</factorymodule>
                  <value>VH.xml</value>
          </property>  
          <property>
                  <name>VV</name>
                  <factoryname>createALOS</factoryname>
                  <factorymodule>isceobj.Sensor</factorymodule>
                  <value>VV.xml</value>
          </property>
        </component>

Next, we will specify our output file names and options.  Create a file called
*output.xml* and put the following information in it::

        <component>
         <property>
                 <name>FILTER</name>
                 <value>None</value>
         </property>
         <property>
                 <name>FILTER_SIZE_X</name>
                 <value>21</value>
         </property>
         <property>
                 <name>FILTER_SIZE_Y</name>
                 <value>11</value>
         </property>
         <property>
                 <name>TEC</name>
                 <value>tec.slc</value>
         </property>
         <property>
                 <name>FARADAY_ROTATION</name>
                 <value>fr.slc</value>
         </property>
         <property>
                 <name>PHASE</name>
                 <value>phase.slc</value>
         </property>
        </component>

Finally, create four XML files, one for each polarity combination, *HH.xml*,
*HV.xml*, *VH.xml* and *VV.xml*, and place the following information in them::

        <component>
         <property>
                 <name>LEADERFILE</name>
                 <value>LED-ALPSRP016410640-P1.0__A</value>
         </property>
         <property>
                 <name>IMAGEFILE</name>
                 <value>IMG-HH-ALPSRP016410640-P1.0__A</value>
         </property>
        </component>

We can now produce estimates of Faraday rotation, TEC and phase delay by running

``$ISCE_HOME/applications/ISSI.py FR.xml output.xml``

The code will create the Faraday rotation output in a file named fr.slc, TEC
output in a file named tec.slc, and phase delay in a file named phase.slc. 


ISSI in Detail
**************

This section details the structure and usage of ISSI.py, an application within ISCE that performs polarimetric processing. It assumes that the user has already installed Python and ISCE successfully. ISSI.py was written over a year ago by a computer scientist, no longer contributing to the project at NASA JPL. As a result it is structured differently from most other scripts written by the current software developers. Unfortunately, this means that understanding the processing flow of ISSI.py is difficult, and other applications within ISCE do not serve as templates to help with the task. Also, the structure of this program is extremely object oriented, where executing a function in ISSI.py may call methods from up to five different Python scripts located elsewhere within ISCE’s file structure. Thus, this task ultimately devolves to tracing out the processing flow as it takes you from script to script. Throughout this journey into ISSI.py we limit the depth of understanding to processing tasks directly relevant to polarimetric processing. Other components such as the wrapping process, the wrapped C and Fortran code itself, and computer resource management are worthy of mention, but we do not dig into the specifics of their operation. We treat these portions of the code as black boxes whose functionality is well understood.

The following diagram gives an overview of the steps taken by the ISSI scripts to calculate Faraday rotation, TEC and phase delay.

   .. figure:: ISSI_workflow.png
      :align: center
      :width: 1050px

      ISSI workflow diagram


Extracting Information from Xml Files
-------------------------------------

ISSI.py begins by running its ``main()`` method. It first creates an object called *fi* which is an instance of the class FactoryInit. FactoryInit is a class in FactoryInit.py whose methods and attributes allow the program to extract information found in the six xml files required for processing (FR.xml, output.xml, HH.xml, HV.xml, VH.xml, VV.xml; see `Running ISSI`_). Whenever the program creates an instance of a Python class, it always runs the ``__init__()`` method found within that class’ script. The FactoryInit class’ ``__init__()`` method simply defines the attributes of the object and then returns to ISSI.py. In the remainder of the document, we gloss over this initialization process for other objects because they all follow identical procedures.

ISSI.py then extracts information from the first input argument, FR.xml. To do this it sets two attributes of *fi*, *fileInit* and *defaultInitModule*, to ‘FR.xml’ and ‘InitFromXmlFile’, respectively, and then runs the method ``initComponentFromFile()``. FactoryInit.py contains the definition of ``initComponentFromFile()`` because it is a method of the FactoryInit class; it returns a dictionary with all information found within FR.xml. The function ``getComponent()``, again inside FactoryInit.py, searches that dictionary and returns an instance of the Sensor (i.e., the satellite) class responsible for creating that particular data file defined as the input argument of ``getComponent()``. ISSI.py supports different Sensor classes, and this guide follows the processing path assuming all four raw images are products of the ALOS/PALSAR mission. We therefore indicate by *hh*, *hv*, *vh* and *vv* the four instances of the class ALOS, found in ALOS.py.

At this point, ISSI.py moves to the second input argument, output.xml. Frustratingly, extracting information from this xml file requires a completely new object of class InitFromXmlFile, found in InitFromXmlFile.py. One of its methods called ``init()`` (**NOT** to be confused with ``__init__()``) extracts information from output.xml and returns it to an attribute local to ISSI.py called *variables*. ISSI.py then distributes the information found in *variables* to other different local attributes, including filter size and file names to be used later, that serve as input variables to an instance of the class Focuser called *focuser*. ``main()`` in ISSI.py concludes by setting the *filter* and *filterSize* attributes of *focuser* and then running ``focuser()``, a method in the Focuser class that begins processing the raw images.

``focuser()`` begins by calling the function ``useCalcDop()``, found in __init__.py, that instantiates and returns an instance, called *doppler*, of the class Calc_dop, found within Calc_dop.py. We see that ``useCalcDop()`` simply redirects the program to the class Calc_dop. The object *doppler* later provides the attributes and methods necessary to calculate Doppler information about the processed radar images.


Extracting Data from Input Files
--------------------------------

Extraction of raw data from the input files begins with the creation of objects, called *hh.raw*, *hv.raw*, *vh.raw* and *vv.raw*, which hold the output raw data from the extraction process. Following this the program runs the method ``make_raw()``, in ISSI.py, with both the raw data and Doppler objects passed as input arguments. The method ``make_raw()`` immediately creates an instance of the class make_raw, called *mr*. The class make_raw resides in another ISCE application named make_raw.py.

During the initialization of this class the program creates two input port objects. Port objects essentially serve as conduits between Python objects, allowing one to access the attributes of another. After generating *mr*, the method ``make_raw()``, back in ISSI.py, finalizes the input ports by running the method ``wireInputPort()``. The relevance of this method lies only with Python object communication rather than polarimetric radar processing, so we will not examine it in detail here. Finally, the object *mr* runs its own method called ``make_raw()``.

.. note:: We pause here to show the importance of constant vigilance when working with ISCE components. Follow closely: we just ran ISSI.py’s method called ``make_raw()``, which then created an instance of the class make_raw, found in make_raw.py, which then ran ``make_raw()``, a method of the class make_raw located in make_raw.py.

Then, in the ``make_raw()`` method of make_raw.py, the method ``extractImage()`` runs. The sensor class of the image file, in our case ALOS for all four polarized images, contains the method ``extractImage()``. Note that the script ALOS.py contains definitions for not one but four different classes. The user must therefore look closely to see which methods in ALOS.py fall under which classes; this will become relevant soon.

``extracImage()`` begins with if statements designed to ensure that the user passed correct xml and image files to ISSI.py. The first if statement ensures that the attributes *_imageFileList* and *_leaderFileList* are lists rather than strings. The second quits the program if the number of leader and image files is not the same. The final if statements protect against the case that an image file contains more than one image; if the user operates ISSI.py as instructed, these if statements should be inconsequential. The program then creates instances of three new classes: an instance of Frame, in Frame.py, called *frame*, and instances of the classes LeaderFile and ImageFile, called *leaderFile* and *imageFile*, respectively, both in ALOS.py. The input argument to initialization of the LeaderFile class is the leader file in memory.

The program then attempts to parse the leader file by running ``parse()`` on *leaderFile*. After opening the leader file in read mode, ``parse()`` then creates an instance of the class CEOSDB, in CEOS.py, called *leaderFDR*. An xml file containing the ALOS leader file record (also known as CEOS data file), provided already by ISCE, and the leader file itself serve as input arguments for the initialization of CEOSDB. During initialization a local variable called *rootChildren* stores an element tree of the information stored in the ALOS leader file record xml file. If the image file being processed comes from a spacecraft with no leader file record, *rootChildren* simply becomes an empty list.

With *leaderFDR* completely initialized, it runs its method ``parse()``. ``parse()`` opens the ALOS leader file record and extracts any information it contains via an element tree; this user manual does not look any more closely at how ISSI.py parses xml files. Find documentation on element trees for more information. If values found within *leaderFDR*’s recently parsed metadata indicate to, the final lines of ``parse()``, the method acting upon *leaderFile*, perform the same element tree parsing process on scene header, platform position, spacecraft attitude, and spacecraft calibration xml files, also all provided within ISCE. After closing the leader file, we return to ``extractImage()`` where *imageFile* submits itself to a similar parsing process, also called ``parse()`` but found under the class ImageFile.

Then, we open the image file and, just like before, create an instance of the class CEOSDB called *imageFDR*, run ``parse()`` on this object, set the number SAR channels as found in *imageFDR*’s metadata, and run ``_calculateRawDimensions()`` if the input argument *calculateRawDimensions* is true. For the ALOS case, *calculateRawDimensions* is false so the program skips over ``_calculateRawDimensions()``. Finally, we close the image file and return to ``extractImage()``.

The next portion of code decides whether the image ought to be resampled; it currently does not resample the image. Instead it moves on to run ``extractImage()``, a method of the class ImageFile, on the image file itself. ``extractImage()`` checks the data type of the image file. If the data is Level 1.5, it raises an exception and exits the program. If the data is Level 1.1, a single-look-complex (SLC) image, it runs the method ``extractSLC()``. Finally, if the data is Level 1.0, the original raw image, the program runs ``extractRaw()``. Level 1.0 data is the most basic form of radar image, so we will explore this branch in order to ensure complete coverage of ISSI.py. If the processing facility for the image file is ERSDAC, the program runs the method ``alose_Py()`` to extract the raw image. If not, it runs ``alos_Py()``.

.. note:: Whether the SLCs are resampled or not, a config.txt file is created giving image metadata in PolSARpro format.

.. note:: Whenever you encounter a method whose name ends with _Py, you have found the beginning of the wrapping process described elsewhere in ISCE.pdf. In the current case, alos_Py ultimately refers to a function found in ALOS_pre_process.c, one of the many pieces of original scientific software that inspired the ISCE project. Other sections of ISCE.pdf describe in detail the Python wrapping process, and understanding the source code is left to radar scientists. Therefore here we go no further into any method ending in _Py.

The methods ``alos_Py()`` and ``alose_Py()`` both perform the actual image extraction; look closely at ALOS_pre_process.c to understand how. After they run, the program sets some local variables and then runs a method ``createRawImage()``. ``createRawImage()`` returns an instance of the class RawImage, in RawImageBase.py, called *rawImage*. The RawImage class serves as ISSI.py’s means of storing and manipulating a raw image. The program creates a new instance of this class every time it needs to process a raw image in any way. After setting some attributes of *rawImage* with information from the raw image’s metadata, it sets the raw image to be the image in the frame of the original ALOS sensor object. Frames can hold more than one image, however the design of ISSI.py ensures that each frame holds only one.


Pre-Focusing Calculations
-------------------------

Minor bookkeeping as well as orbit and Doppler calculations follow the data extraction procedure. ``populateMetadata()``, a method of the ALOS class, first creates and fills metadata objects from the CEOS format metadata generated earlier. It is worth noting here that one of the methods in ``populateMetadata()``, called ``_populateDistortions()``, creates the transmit and receive polarimetric calibration distortion matrices. The polarimetric calibration process later implements these matrices during the formation of the SLC image.

The method ``readOrbitPulse()``, with the leader file, image file, and image width as input parameters, prepares to calculate the ALOS positions and times of the raw image. The method creates instances of three image classes, RawImage, StreamImage, and Image, called *rawImage*, *leaImage* and *auxImage*, respectively. The class StreamImage holds and manipulates the leader file in memory while the Image class creates a generic image object. Each image object has an associated image accessor, which it passes to other objects, allowing them to access the image in memory. Finally, ``readOrbitPulse()`` runs three separate methods called ``setNumberBitesPerLine_Py()``, ``setNumberLines_Py()`` and ``readOrbitPulse_Py()``. These methods wrap Fortran source code that fills *auxImage* with an auxiliary file of file extension .raw.aux, containing the ALOS positions and times of the raw image. After this process, the method finalizes the three image objects and returns to ``extractImage()``. The program appends the frame created earlier to the list of frames in memory and then returns to ``make_raw()`` to begin Doppler calculations.

If the image extracted earlier is Level 1.0 data, ``make_raw()`` wires three input ports to *doppler* so that it may access attributes of the instrument, raw image, and frame objects. *doppler* then calculates the Doppler fit for the raw image using ``calculateDoppler()``, a method of the Calc_dop class. This method creates yet another RawImage object to access the image, and then passes that object’s accessor to ``calc_dop_Py()``, a method that wraps the source code calc_dop.f90. As with all methods that include wrapped source code, ``calculateDoppler()`` contains a significant amount of pre-processing steps, including setting the state of Fortran compatible parameters necessary for the wrapped source code as well as allocating memory for its processes. After calc_dop.f90 calculates the Doppler fit for the image, ``calculateDoppler()`` deallocates memory and runs ``getState()``, a method that grabs the information calc_dop.f90 calculated and loads it into attributes of the Python object *doppler*.

Next, *doppler* runs its method ``fitDoppler()``, whose original purpose is to fit a polynomial to the Doppler values. Inside the ``fitDoppler()`` method itself, however, we find that rather than perform a polynomial fit, it simply sets the first Doppler coefficient to the zero order term found earlier, leaving all others at zero. To conclude Doppler processing, ``make_raw()`` establishes both pulse repetition frequency and the Doppler coefficients as local variables and then loads them directly into an object called *dopplerValues*, an instance of the class Doppler found in Doppler.py. If the original input data is Level 1.1, an SLC image, the program does not calculate Doppler values and instead loads all zeros into *dopplerValues*. Doppler coefficients allow the generation of an SLC image from a raw image; if the data comes in as an SLC image, the Doppler coefficients are unnecessary.

Following Doppler processing, ``make_raw()`` comes to a close by calculating the velocity, squint angle, and change in height of the spacecraft. Each calculation requires a different method, and each method gets certain parameters of the image and uses them to calculate the desired result in Python. The only method worth investigating here is ``calculateHeightDt()`` because it implements the method ``interpolateOrbit()``. Found in Orbit.py, ``interpolateOrbit()`` offers three ways of interpolating the state vector of an orbit; it performs linear interpolation, interpolation with an eighth order Legendre polynomial, or Hermite interpolation. The math of these different interpolation techniques lies beyond the scope of this user guide. After interpolating the orbit at both the start and mid-times of the image capture, ``calculateHeightDt()`` calculates the height of the spacecraft using a method in Orbit.py called ``calculateHeight()``. ``calculateHeight()`` itself runs a method called ``xyz_to_llh()`` that converts the spacecraft ellipsoid from Cartesian coordinates to latitude, longitude, and height, and returns height. The method ``calculateHeightDt()`` concludes using the height and time parameters just calculated to determine the change in height over time.

Finally, ``make_raw()`` concludes with ``renderHdr()``, a method in Image.py that creates an xml file containing important parameters of the raw image.


Focusing the Raw Image
----------------------

The process of creating an SLC image begins with estimating an average Doppler coefficient *fd* for all of the polarized images. It adds all four coefficients together and divides by four. ISSI.py then runs ``focus()``, with the raw image object and average Doppler coefficient as input arguments.

The first step in ``focus()``, after getting parameters necessary for processing, calculates a value called peg point. Also found in ISSI.py, ``calculatePegPoint()`` passes the frame, planet, and orbit and returns peg, height, and velocity values. It also makes heavy use of both the ``interpolateOrbit()`` and ``xyz_to_llh()`` methods to calculate points in both location and time. It also implements ``geo_hdg()``, another method in Ellipsoid.py, that calculates the spacecraft’s heading given its start and middle locations. An instance of the class Peg, in Peg.py, called *peg*, stores the peg point information; ``calculatePegPoint()`` returns the Peg object as well as height and speed.

Interpolating and returning the spacecraft’s orbit comes next, beginning with the method ``createPulsetiming()``. This method returns an instance of the class Pulsetiming, in Pulsetiming.py, called *pt*, which runs the method ``pulsetiming()``. ``pulsetiming()`` interpolates the spacecraft orbit and calculates a state vector for each line of the image. It appends each successive state vector together in order to return the complete orbit of the spacecraft. The program then converts this complete orbit to SCH coordinates with an instance of the class Orbit2sch, found in Orbit2sch.py, called *o2s*. It wires a few input ports, sets the average height of the spacecraft, and then performs the conversion with its method ``orbit2sch()``. After setting parameters and allocating memory, ``orbit2sch()`` runs orbit2sch.F, source code wrapped by the method ``orbit2sch_Py()``.

Back now in ISSI.py, ``focus()`` creates instances of the RawImage and SlcImage classes called *rawImage* and *slcImage*, respectively. While *rawImage* provides access to the raw image in memory, *slcImage* facilitates the creation of the SLC image in memory. The program also creates an instance of the class Formslc, found in Formslc.py, called *focus*, which contains the attributes and methods necessary to process raw data into an SLC image. With these objects prepared, ``focus()`` wires input ports and sets variables necessary for generating the SLC image. Notice that, while ``focus()`` has many lines, the vast majority of its commands simply get and set data calculated elsewhere; most of ``focus()`` simply prepares for the actual SLC generation, executed in method called ``formslc()``.

``formslc()`` finishes wiring the ported objects, allocates memory, sets parameters, and runs ``formslc_Py()``, the method that wraps formslc.f90. This Fortran code completely generates the SLC image, and after it finishes, another wrapping function called ``getMocompPositionSize_Py()`` returns information about motion compensation performed in formslc.f90. ``formslc()`` concludes by setting a few more local variables, running ``getState()``, which returns more motion compensation parameters from the Fortran processing, deallocating memory, and creating an xml header file for the new SLC image.

Once more in ``focus()``, both *rawImage* and *slcImage* run their ``finalizeImage()`` methods. Now only one last step remains for ``focus()``, to convert the SLC image from writeable to readable. It accomplishes this by creating another SlcImage object identical to that created earlier, but setting it as readable rather than writeable. Finalizing this image object and defining local variables of image length and width conclude the conversion process from raw data to SLC image.


Resampling the SLC Image
------------------------

``focuser()`` next runs the method ``resample()``, in ISSI.py, on the VH and VV polarized SLC images. As usual, ``resample()`` begins by getting and setting parameters and objects relevant to the resampling process. It creates two SlcImage objects called *slcImage*, which refers to the SLC image currently in memory, and *resampledSlcImage*, which facilitates the creation of a resampled SLC image file. Following this, it creates an instance of the class OffsetField, in Offset.py, called *offsetField*, that represents a collection of offsets defining an offset field. The program then proceeds to create an instance of the class Offset, also in Offset.py, called *offset*, with a constant 0.5 pixel shift in azimuth. This offset adds to the offset field, ready for use later in the resampling process.

An instance of the class Resamp_only, found in Resamp_only.py, called *resamp*, enables the resampling process. After setting local parameters and establishing ports, resamp runs the method ``resamp_only()`` on the two SlcImage objects. As usual, the method imports objects from ports, establishes parameters, allocates memory, and runs the wrapping method, in this case ``resamp_only_Py()``, which points to resamp_only.f90. Resamp_only.f90 concludes, ``getState()`` runs, and ``resamp_only()`` deallocates memory before returning to ``resample()``. It finalizes both image objects, renames the resampled image files to be the new SLC images, and returns to the ``focuser()`` processing flow.

Once more in ``focuser()``, if the original input data is Level 1.1, the program changes the extracted files’ extensions from .raw to .slc. This step is necessary because the extraction process detailed earlier gives the files .raw extensions by default. And finally, just before beginning polarimetric processing, ``focuser()`` checks the endianness of the image files and swaps it if necessary.


Polarimetric Processing
-----------------------

The final line of ``focuser()`` executes the method ``combine()``, which combines all four polarized images to form Faraday rotation (FR), total electron content (TEC) and phase images. The method ``combine()`` begins with an instance of the class FR, found in FR.py, called ``issiObj``. All of the SLC images as well as size parameters and objects to hold the ouput of polarimetric processing pass as input arguments to the initialization of FR. If the input data to ISSI.py is Level 1.0, as we assume, issiObj runs the method ``polarimetricCorrection()``, with the distortion matrices as its input arguments.

Before this point in ISSI.py, nearly all the wrapped source code is Fortran. For polarimetric processing, however, nearly all the source code is compiled C code. Fortunately for us, Python interacts well with C and requires a much simpler wrapping process. This process consists of converting Python parameters, such as strings, characters, floats, etc., into C compatible parameters via built in Python functions such as ``c_char_p()`` or ``c_float()``, and then executing the wrapped code itself. Such a straightforward wrapping procedure greatly simplifies understanding ISSI.py, and therefore this user guide.

``polarimetricCorrection()`` creates the appropriate ctype parameters, including file names and distortion matrices, and runs ``polcal()``, found in polcal.c. Interestingly, polcal.c performs only part of the calibration process, calling upon yet another wrapped file polarimetricCalibration.f to perform the calibration computation. The interconnection of C and Fortran code is beyond the scope of this section. After the source code completes its tasks, ``polarimetricCorrection()`` shifts the results to the output files and returns to ``combine()`` in ISSI.py.

The program next calculates Faraday rotation (FR) via the method ``calculateFaradayRotation()``, also in FR.py. This method begins with ``_combinePolarizations()``, which itself creates necessary ctype parameters and then runs the wrapping method ``cfr()``, which points to cfr.c. Using the Bickel and Bates 1965 method, cfr.c calculates complex FR from the four polarized images. Following this, ``calculateFaradayRotation()`` calls ``_filterFaradayRotation()``, a method that utilizes the filter parameters found in output.xml to filter the FR. After generating an instance of the class Filter, found in Filter.py, the method runs one of three possible filter types, medianFilter, gaussianFilter, and meanFilter. Each of these filter methods establishes important parameters and then runs a Python wrapping method, ``medianFilter_Py()``, ``gaussianFilter_Py()``, or ``meanFilter_Py()``, that actually performs the filtering process. These filtering methods actually call upon more than one piece of source code. See the appendix workflow for more detail.

Calculation of the average real valued FR follows next. The program generates the appropriate ctype parameters and then runs ``cfrToFr()``, a Python method that wraps cfrToFr.c. After ``cfrToFr()`` calculates and returns the average real valued FR at each pixel (in radians), ``calculateFaradayRotation()`` generates a resource for the new FR file and then returns to ``combine()``.

The final portion of polarimetric processing requires calculation of the geodetic corners of the images. To this end the program sets the date and radar frequency as local parameters and then executes ``calculateLookDirections()``, in ISSI.py, which calculates the satellite’s look direction at each corner of the image. To do this it first calculates the satellite heading at mid-orbit with the function ``calculateHeading()``. Calculate heading gets the orbit and ellipse parameters of the images and, as before, interpolates the orbit and converts the state vector outputs to latitude, longitude and height. The function ``geo_hdg()`` uses that information to calculate the satellite’s heading, and ``calculateHeading()`` returns this information in degrees. ``calculateLookDirections()`` takes the heading value, adds to it the yaw value plus 90 degrees, and returns it as the look direction.

Next, the program calculates the corner locations via ``calculateCorners()``. This method sets the image planet as a local parameter, ports an instance of the class Geolocate, found in Geolocate.py, and sets many more local parameters before running ``geolocate()`` on each corner. ``geolocate()`` creates the necessary ctypes and calls ``geolocate_wrapper()``, a Python method that wraps geolocate_wrapper.c. The C code calls ``geolocate()``, which itself derives from source code called geolocate.f; this Fortran calculates the corners and look angle at each corner. Back in ``geolocate()`` in Geolocate.py, the Python script creates an instance of the class Coordinate, which stores the latitude, longitude, and height of the corner just calculated. It returns the coordinate object, as well as the look and incidence angles, to ``calculateCorners()`` in ISSI.py, which itself returns the parameters for all four corners to ``combine()``.

The program next calls ``makeLookIncidenceFiles()`` to create files containing look and incidence angles in order to test antenna pattern calibration. This method also ports an instance of the Geolocate class, sets planet, orbit, range, etc. as local parameters, and opens the two files meant to store the new angle information. It then gets the time of the acquisition and uses ``interpolateOrbit()`` to return a state vector which is itself used as each pixel in the range direction (width of the image) to calculate the coordinate, look angle, and incidence angle via ``geolocate()``, the method used earlier to calculate corners. The program then stores the look and incidence angle values, calculated for each pixel in the range direction, in every pixel of the column located at that width. ``makeLookIncidenceFiles()`` closes the two files and returns to ``combine()``.

The second to last polarimetric processing method is ``frToTEC()``. Given a coordinate, look angle, and look direction, ``frToTEC()`` calculates the average magnetic field value in the radar line-of-sight. It starts by, for each corner, setting a local parameter k to be the look vector, calculated from look angle and look direction, via the method ``_calculateLookVector()``. Then it appends the result of performing the dot product of k, the look vector, and magnetic field, via the method ``_integrateBVector()``, to a list of such dot products at each corner. ``_integrateBVector()`` creates a vector of altitude information and at each height in that vector calculates the magnetic field vector with ``_calculateBVector()``. ``_calculateBVector()`` establishes necessary ctypes and runs ``calculateBVector()``, a Python method that wraps calculateBVector.c, which itself calls upon igrf2005_sub.f. This Fortran code calculates and returns the magnetic field value at each input coordinate, and ``_calculateBVector()`` returns the North, East, and down components of the magnetic field at each point. ``_integrateBVector()`` then performs the dot product between the magnetic field and look vector and calculates and returns the average dot product value for all points in the height vector. Given the mean value of the dot product and the radar frequency, ``_scaleFRToTEC()`` applies a scaling factor to FR in order to arrive at TEC. With the correct ctypes, ``_scaleFRToTEC()`` calls upon frToTEC.c to perform the actual scaling conversion. After arriving at TEC, ``ftToTEC()`` creates a resource file for the TEC file, and returns to ``combine()`` in ISSI.py.

Finally, ``combine()`` executes the final method of ISSI.py and runs ``tecToPhase()``, also found in FR.py, which applies a scalar value to TEC in order to return phase. With the correct ctypes, ``tecToPhase()`` calls ``convertToPhase()``, a method that wraps tecToPhase.c, which applies the scaling factor. The program concludes by creating a resource file for the phase file. Here lies the end of ISSI.py. [Zeb10]_  [LavSim10]_




.. [FreSa04] Freeman, A., and S. S. Saatchi (2004), On the detection of Faraday rotation in linearly polarized L-band SAR backscatter signatures, IEEE T. Geosci. Remote, 42(8), 1607–1616.

.. [Pi12] Pi, X., A. Freeman, B. Chapman, P. Rosen, and Z. Li (2012), Imaging ionospheric inhomogeneities using spaceborne synthetic aperature radar, J. Geophys. Res.

.. [Fre04] Freeman, A. (2004), Calibration of linearly polarized polarimetric SAR data subject to Faraday rotation, IEEE T. Geosci. Remote, 42(8), 1617–1624.

.. [Bick65] Bickel, S. H., and R. H. T. Bates (1965), Effects of magneto-ionic propagation on the polarization scattering matrix, pp. 1089–1091.

.. [Zeb10] H. Zebker, S. Hensley, P. Shanker, and C. Wortham, Geodetically Accurate InSAR Data Processor, IEEE Transactions on Geoscience and Remote Sensing, 2010.

.. [LavSim10] M. Lavalle and M. Simard, Exploitation of dual and full PolInSAR PALSAR data, in 4th Joint ALOS PI Symposium, Tokyo, Japan, Nov. 2010.
