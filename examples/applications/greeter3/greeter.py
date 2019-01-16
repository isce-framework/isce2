#!/usr/bin/env python3

from __future__ import print_function
from __future__ import absolute_import

import isce
from iscesys.Component.Application import Application

NAME = Application.Parameter('gname',
    public_name='name to use in greeting',
    default="World",
    type=str,
    mandatory=False,
    doc="Name you want to be called when greeted by the code."
)

GREETING = Application.Facility('greeting',
    public_name='Greeting message',
    module = 'greetings',
    factory = 'english_standard',
    mandatory=False,
    doc='Generate a greeting message'
)

class Greeter(Application):

    parameter_list = (NAME,)
    facility_list = (GREETING,)
    family = 'greeter'

    def main(self):
        #the main greeting message
        self.greeting(self.gname)

        #some information on the inner workings
        print()
        print("Some information")
        from iscesys.DictUtils.DictUtils import DictUtils
        normname = DictUtils.renormalizeKey(NAME.public_name)
        print("NAME.public_name = {0}".format(NAME.public_name))
        print("normname = {0}".format(normname))
        print("self.gname = {0}".format(self.gname))
        if self.descriptionOfVariables[normname]['doc']:
            print("doc = {0}".format(self.descriptionOfVariables[normname]['doc']))
        if normname in self.unitsOfVariables.keys():
            print("units = {0}".format(self.unitsOfVariables[normname]['units']))

        print()
        print("For more fun, try this command line:")
        print("./greeter.py greeter.xml")
        print("./greeter.py greeterS.xml")
        print("./greeter.py greeterEC.xml")
        print("Try the different styles that are commented out in greeter.xml")
        print("Try entering data on the command line:")
        print("./greeter.py greeter.'name to use in greeting'=Jane")
        print("or try this,")

        cl = "./greeter.py "
        cl += "Greeter.name\ to\ use\ \ \ IN\ greeting=Juan  "
        cl += "greeter.'Greeting Message'.factorymodule=greetings "
        cl += "greeter.'Greeting message'.factoryname=english_cowboy  "
        cl += "greeter.name\ to\ use\ in\ greeting.units='m/s' "
        cl += "greeter.'name to use in greeting'.doc='My new doc'"
        print("{0}".format(cl))

        print("etc.")

        return

    def __init__(self, name=''):
        super().__init__(family=self.family, name=name)
        return

if __name__ == '__main__':
    greeter = Greeter(name='greetme')
    greeter.configure()
    greeter.run()
