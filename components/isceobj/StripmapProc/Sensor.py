#!/usr/bin/env python3

from isceobj.Sensor import createSensor as CS
import logging

def createSensor(commonSensor, specificSensor, name=None):
    if specificSensor not in [None, '']:
        if name is not None:
            logging.info('{0} sensor object provided explicitly'.format(name))

        return CS(specificSensor, name) 

    if commonSensor not in [None, '']:
        if name is not None:
            logging.info('{0} sensor not provided explicitly, using common sensor'.format(name))

        return CS(commonSensor, name)
        

