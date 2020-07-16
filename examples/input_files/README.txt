This directory contains an example insarApp.xml input file to be used with the
application insarApp.py.  The insarApp.xml file is extensively documented with
comments.  Please read that for further information.

The insarApp.xml file contains references to two files (or "catalogs" in ISCE
parlance), reference.xml and secondary.xml,  which contain further input data for
insarApp.xml.  The structure of the reference.xml and secondary.xml files are the same
for any given sensor (only the specific image/meta data filenames will be
different).  The structure of these files, however, are different for the
different sensors.  Examples for each supported sensor are contained in this
directory with names such as reference_alos.xml.  You can edit these files and
change the name used in the insarApp.xml file (or else change the name of these
reference_SENSOR.xml and secondary_SENSOR.xml files to reference.xml and secondary.xml to
match the namse of the catalog files referred to in insarApp.xml).

A catalog file is parsed as if its contents (omitting the inital tag and its
closing tag) were directly inserted into the insarApp.xml file.  For example,
the following insarApp.xml file,

insarApp.xml
------------
<insarApp>
    <component name="insar">
        <property  name="Sensor name">ALOS</property>
        <component name="reference">
            <catalog>reference_alos.xml</catalog>
        </component>
        <component name="secondary">
            <catalog>secondary_alos.xml</catalog>
        </component>
    </component>
</insarApp>


with the following reference_alos.xml and secondary_alos.xml file,

reference_alos.xml
----------
<component name="Reference">
    <property name="IMAGEFILE">
        /<path-to-your-file>/IMG-HH-ALPSRP056480670-H1.0__A
    </property>
    <property name="LEADERFILE">
        /<path-to-your-file>/LED-ALPSRP056480670-H1.0__A
    </property>
    <property name="OUTPUT">20070215.raw </property>
</component>

secondary_alos.xml
---------
<component name="Secondary">
    <property name="IMAGEFILE">
        /<path-to-your-file>/IMG-HH-ALPSRP049770670-H1.0__A
    </property>
    <property name="LEADERFILE">
        /<path-to-your-file>/20061231/LED-ALPSRP049770670-H1.0__A
    </property>
    <property name="OUTPUT">20061231.raw </property>
</component>


is equivalent to the following insarApp_AllInOne.xml file

insarApp_AllInOne.xml
---------------------
<insarApp>
    <component name="insar">
        <property  name="Sensor name">ALOS</property>
        <component name="reference">
            <property name="IMAGEFILE">
                /<path-to-your-file>/IMG-HH-ALPSRP056480670-H1.0__A
            </property>
            <property name="LEADERFILE">
                /<path-to-your-file>/LED-ALPSRP056480670-H1.0__A
            </property>
            <property name="OUTPUT">20070215.raw </property>
        </component>
        <component name="secondary">
            <property name="IMAGEFILE">
                /<path-to-your-file>/IMG-HH-ALPSRP049770670-H1.0__A
            </property>
            <property name="LEADERFILE">
                /<path-to-your-file>/20061231/LED-ALPSRP049770670-H1.0__A
            </property>
            <property name="OUTPUT">20061231.raw </property>
        </component>
    </component>
</insarApp>


You are free to use the "all in one" style or the separate files as catalogs
style.  It makes no difference.


===============================================================================
EXTRAINFORMATION NOT REQUIRED TO GET STARTED---for future reference, as
needed, for clarification.  Don't worry if this information doesn't make
sense at this time:

There are further options documented in the top level README.txt file for
configuring most of the components in ISCE.  As you become familiar with
those other options, you will learn that there is a subtle difference in the
structure of a "component configuration file", which is a stand-alone
configuration file and a "catalog file" which is content to be inserted into
a component configuration file.

The difference between a catalog file and a component configuration file
is in the one extra tag ("<insarApp>" in the above insarApp.xml file) that
is an extra structure around the data that the example reference_alos.xml and
secondary_alos.xml files found in this directory are lacking.  The other difference
is in the name of the files.  A component configuration file must be named
properly as explained in the top level README.txt file in order for the ISCE
framework to find it.  When named appropriately the component configuration
files are found automatically.  A catalog file is referred to explicitly in
the input file and may have any name desired.

The reference_alos.xml and secondary_alos.xml files here could be turned into
component configuration files by adding one tag (with any name desired) at
the top of the file and its required closing tag at the bottom of the file
and by changing their names to reference.xml and secondary.xml. The difference would
also be that the component configuration versions would be loaded automatically
without needing to refer to them in the input file. As catalogs, however, they
can be given any name as long as the insarApp.xml file uses that name in a
catalog tag.

If you were to have both catalog files and component configuration files, then
both will be read when configuring the reference and secondary components. If there is
conflicting information in the catalog file and the component configuration
file, then, by the rules of priority discussed in the top level README.txt, the
catalog referred to in the insarApp.xml file will win because it specifies both
the application (insarApp) and the component (reference or secondary), whereas the
component configuration file would only refer to the component (reference or
secondary).

===============================================================================
