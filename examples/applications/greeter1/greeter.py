#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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




"""
greeter:
An ISCE application to greet the user illustrating the usage of
Application.Parameter to expose configurable parameters to the user
through an input xml file.

The accompanying greeter.xml file illustrates various formats
allowed for the input file as well of examples that will not work.

To run this example type,

> ./greeter.py greeter.xml

or, one of the following:

> ./greeter.py --greeter.name\ to\ use\ in\ greeting=Joe
> ./greeter.py --greeter."name to use in greeting"=Joe
"""

from __future__ import print_function

import isce
from iscesys.Component.Application import Application

NAME = Application.Parameter('gname',
    public_name='name to use in greeting',
    default="World",
    type=str,
    mandatory=True,
    doc="Name you want to be called when greeted by the code."
)

class Greeter(Application):
    """
    """
    parameter_list = (NAME,)
    #facility_list = ()

    family = 'greeter'

    def main(self):
        print("Hello, {0}!".format(self.gname))

        #Print some extra information to see how it all works
        print()
        print("Some additional information")
        from iscesys.DictUtils.DictUtils import DictUtils
        normname = DictUtils.renormalizeKey(NAME.public_name)
        print("Parameter NAME public name = {0}".format(NAME.public_name))
        print("Parameter NAME internal normalized name = {0}".format(normname))
        if self.descriptionOfVariables[normname]['doc']:
            print("Parameter NAME doc = {0}".format(self.descriptionOfVariables[normname]['doc']))
        if normname in self.unitsOfVariables.keys():
            print("Parameter NAME units = {0}".format(self.unitsOfVariables[normname]['units']))
        print("Application attribute: self.gname = {0}".format(self.gname))

        print()
        print()
        print("For more fun, try these command lines:")
        print("./greeter.py greeter.xml")
        print("Try the different styles that are commented out in greeter.xml")
        print("Try entering data on the command line:")
        print("./greeter.py greeter.'name to use in greeting'=Jane")
        print("or try this,")

        cl = "./greeter.py "
        cl += "Greeter.name\ to\ use\ \ \ IN\ greeting=Juan "
        cl += "greeter.'name to use in greeting'.units='m/s' "
        cl += "greeter.'name to use in greeting'.doc='My new doc string'"

        print("{0}".format(cl))

        print("etc.")

        return

    def __init__(self, family='', name=''):
        super().__init__(family=self.family, name=name)
        return

if __name__ == '__main__':
    greeter = Greeter(name='greetme')
    greeter.configure()
    greeter.run()
