#!/usr/bin/env python3

from __future__ import print_function
from __future__ import absolute_import

import isce
from iscesys.Component.Application import Application

NAME = Application.Parameter('gname',
    public_name='name to use in greeting',
    default='World',
    type=str,
    mandatory=False,
    doc="Name you want to be called when greeted by the code."
)

LANGUAGE = Application.Parameter('language',
    public_name='language to use in greeting',
    default='English',
    type=str,
    mandatory=False,
    doc="language you want to be used when greeted by the code."
)

GREETING = Application.Facility('greeting',
    public_name='Greeting message',
    module = 'greetings',
    factory = 'language',
    args = (LANGUAGE,),
    mandatory=False,
    doc="Generate a greeting message."
)

class Greeter(Application):

    parameter_list = (NAME, LANGUAGE)
    facility_list = (GREETING,)
    family = "greeter"

    def main(self):
        #The main greeting
        self.greeting(self.gname)

        #some information on the internals
        from iscesys.DictUtils.DictUtils import DictUtils
        normname = DictUtils.renormalizeKey(NAME.public_name)
        print()
        print("In this version of greeter.py, we use an input parameter to ")
        print("select a greeter 'facility' to perform the greeting.  The")
        print("greeting facility is created in greetings/greetings.py using")
        print("its language method, which takes a string argument specifying")
        print("the desired language as an argument.  The factories to create")
        print("the greeter for each pre-selected language is contained in that")
        print("file.  The components that fill the role of the greeter facility")
        print("are the components (such as EnglishStandard) in the greetings")
        print("directory")

        print()
        print("Some further information")
        print("Parameter NAME: public_name = {0}".format(NAME.public_name))
        print("Parameter NAME: internal normalized name = {0}".format(normname))
        if self.descriptionOfVariables[normname]['doc']:
            print("doc = {0}".format(self.descriptionOfVariables[normname]['doc']))
        if normname in self.unitsOfVariables.keys():
            print("units = {0}".format(self.unitsOfVariables[normname]['units']))
        print("Greeter attribute self.name = {0}".format(self.name))

        normlang = DictUtils.renormalizeKey(LANGUAGE.public_name)
        print("Parameter LANGUAGE: public_name = {0}".format(LANGUAGE.public_name))
        print("normlang = {0}".format(normlang))
        if self.descriptionOfVariables[normlang]['doc']:
            print("doc = {0}".format(self.descriptionOfVariables[normlang]['doc']))
        if normlang in self.unitsOfVariables.keys():
            print("units = {0}".format(self.unitsOfVariables[normlang]['units']))
        print("Greeter attribute self.language = {0}".format(self.language))

        print()
        print("For more fun, try this command line:")
        print("./greeter.py greeter.xml")
        print("./greeter.py greeterS.xml")
        print("./greeter.py greeterEC.xml")
        print("Try the different styles that are commented out in greeter.xml")
        print("Try entering data on the command line mixing with xml:")
        print("./greeter.py greeter.xml greeter.'language to use in greeting'=spanish")
        print("or try this,")

        cl = "./greeter.py "
        cl += "Greeter.name\ to\ use\ \ \ IN\ greeting=Juan  "
        cl += "gREETER.LANGUAGE\ TO\ USE\ IN\ GREETING=cowboy "
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
