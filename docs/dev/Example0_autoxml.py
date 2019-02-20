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
    ####Master info
    master = {}
    master['hdf5'] = 'master.h5'
    master['output'] = 'master.raw'

    ####Slave info
    slave = {}
    slave['hdf5'] = 'slave.h5'
    slave['output'] = 'slave.raw'

    #####Set sub-component
    insar['master'] = master
    insar['slave'] = slave

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
        <component name="master">
            <property name="hdf5">
                <value>master.h5</value>
            </property>
            <property name="output">
                <value>master.raw</value>
            </property>
        </component>
        <component name="slave">
            <property name="hdf5">
                <value>slave.h5</value>
            </property>
            <property name="output">
                <value>slave.raw</value>
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
