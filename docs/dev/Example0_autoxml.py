#!/usr/bin/env python3

import isce
from isceobj.XmlUtil import FastXML as xml

if __name__ == '__main__':
    '''
    Example demonstrating automated generation of insarApp.xml for
    COSMO SkyMed raw data.
    '''
    
    #####Initialize a component named insar
    insar = xml.Component('insar')

    ####Python dictionaries become components
    ####Reference info
    reference = {}
    reference['hdf5'] = 'reference.h5'
    reference['output'] = 'reference.raw'

    ####Secondary info
    secondary = {}
    secondary['hdf5'] = 'secondary.h5'
    secondary['output'] = 'secondary.raw'

    #####Set sub-component
    insar['reference'] = reference
    insar['secondary'] = secondary

    ####Set properties
    insar['doppler method'] = 'useDEFAULT'
    insar['sensor name'] = 'COSMO_SKYMED'
    insar['range looks'] = 4
    insar['azimuth looks'] = 4
    
    #####Catalog example
    insar['dem'] = xml.Catalog('dem.xml')

    ####Components include a writeXML method
    insar.writeXML('insarApp.xml', root='insarApp')


"""
The output should be of the form.


<insarApp>
    <component name="insar">
        <component name="reference">
            <property name="hdf5">
                <value>reference.h5</value>
            </property>
            <property name="output">
                <value>reference.raw</value>
            </property>
        </component>
        <component name="secondary">
            <property name="hdf5">
                <value>secondary.h5</value>
            </property>
            <property name="output">
                <value>secondary.raw</value>
            </property>
        </component>
        <property name="doppler method">
            <value>useDEFAULT</value>
        </property>
        <property name="sensor name">
            <value>COSMO_SKYMED</value>
        </property>
        <property name="range looks">
            <value>4</value>
        </property>
        <property name="azimuth looks">
            <value>4</value>
        </property>
        <component name="dem">
            <catalog>dem.xml</catalog>
        </component>
    </component>
</insarApp>
"""
