#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2011 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




def createCatalog(name):
    from .Catalog import Catalog
    return Catalog(name)

def createOrderedDict():
    from OrderedDict import OrderedDict
    return OrderedDict

def recordInputs(mainCatalog, obj, node, logger, title):
    """This is merely a convenience method to create a new catalog, add all the
    inputs from the given object, print the catalog, and then import the
    catalog in the main catalog. It returns the created catalog."""
    catalog = createCatalog(mainCatalog.name)
    catalog.addInputsFrom(obj, node + ".inputs")
    catalog.printToLog(logger, title + " - Inputs")
    mainCatalog.addAllFromCatalog(catalog)
    return catalog

def recordOutputs(mainCatalog, obj, node, logger, title):
    """This is merely a convenience method to create a new catalog, add all the
    outputs from the given object, print the catalog, and then import the
    catalog in the main catalog. It returns the created catalog."""
    catalog = createCatalog(mainCatalog.name)
    catalog.addOutputsFrom(obj, node + ".outputs")
    catalog.printToLog(logger, title + " - Outputs")
    mainCatalog.addAllFromCatalog(catalog)
    return catalog

def recordInputsAndOutputs(mainCatalog, obj, node, logger, title):
    """This is a short-hand for using both recordInputs and recordOutputs"""
    recordInputs(mainCatalog, obj, node, logger, title)
    recordOutputs(mainCatalog, obj, node, logger, title)

def testInputsChanged(startCatalog, node, obj):
    endCatalog = createCatalog(startCatalog.name)
    endCatalog.addInputsFrom(obj, node + ".inputs")
    if not (startCatalog == endCatalog):
        import sys
        print("The inputs changed.")
        sys.exit(1)

