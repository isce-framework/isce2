#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2009 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Eric Gurrola, Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
import sys
import os
import operator

from iscesys.Component.Component import Component
from iscesys.DictUtils.DictUtils import DictUtils as DU

class CmdLinePropDict(object):
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = dict()
        return cls._instance

class CmdLineFactDict(object):
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = dict()
        return cls._instance

class CmdLineMiscDict(object):
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = dict()
        return cls._instance

class CmdLineDocDict(object):
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = dict()
        return cls._instance

class CmdLineUnitsDict(object):
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = dict()
        return cls._instance

## A decorator that makes a function taking self as the 1st argument
def curried(func):
    def curried_func(self, *args):
        return func(self, *args)
    return curried_func


class StepHelper(object):
    """This Mixin help sub class's _parameter_steps() methods
    call functions.
    """
    @staticmethod
    def compose(f, g, fargs=(), gargs=(), fkwargs={}, gkwargs={}):
        """compose(f, g)() --> f(g())"""
        from functools import partial
        def fog(*args, **kwargs):
            return  (
                partial(f, *fargs, **fkwargs)(
                    partial(g, *gargs, **gkwargs)(
                        *args, **kwargs
                         )
                    )
                )
        return fog

    def attrgetter(self, attr, attribute=None):
        inst = getattr(self, attribute) if attribute else self
        return getattr(inst, attr)

    def attrsetter(self, attr, value, attribute=None):
        inst = getattr(self, attribute) if attribute else self
        return setattr(inst, attr, value)

    def delayed_attrgetter(self, attr, attribute=None):
        return lambda : self.attrgetter(attr, attribute=attribute)

    def delayed_attrsetter(self, attr, attribute=None):
        return lambda value: self.attrsetter(self,
                                             attr,
                                             value,
                                 attribute=attribute)

    ## self.delayed_attrsetter(attr, delayed_attr
    def delayed_attrcopy_from_to(self, attri, attrf, attribute=None):
        return lambda : self.attrsetter(
            attrf,
            self.attrgetter(
                attri,
                attribute=attribute
                ),
            attribute=attribute
            )

    pass


## Application base class
class Application(Component, StepHelper):
    cont_string = ''

    def run(self, *cmdLine):

        ## Check not any occurance of a steps related command keyword
        if any([operator.contains(
                [y[0] for y in [x.split('=') for x in self.cmdline]], item) for
                item in ("--steps", "--dostep", "--start", "--end", "--next")]
              ):
            print("Step processing")
            self._steps()
            exitStatus = self._processSteps()
        else:
            exitStatus = self.main()

        #Run the user's finalize method
        self._finalize()
        return exitStatus








    # Method allows uses to pass cmdline externally as well
    def _processCommandLine(self,cmdline=None):
        from iscesys.Parsers.Parser import Parser

        if cmdline:
            if(isinstance(cmdline,str)):
                #just in case a string is passed, turn it into a list
                cmdline = [cmdline]
            self.cmdline = cmdline
        else:
            self.cmdline = self._getCommandLine()


        #process the command line and return a dictionary of dictionaries with
        # components per each node.
        # propDict contains the property for each component.
        # factDict contains the info for the component factory.
        # miscDict might contain doc and units. opts are the command lines
        # preceeded by --
        PA = Parser()
        propDict, factDict, miscDict, self._argopts = PA.commandLineParser(
            self.cmdline
            )

        CmdLinePropDict().update(propDict)
        CmdLineFactDict().update(factDict)
        CmdLineMiscDict().update(miscDict)

        #extract doc from miscDict
        docDict = DU.extractDict(miscDict, 'doc')
        CmdLineDocDict().update(docDict)

        #extract units from miscDict
        unitsDict = DU.extractDict(miscDict, 'units')
        CmdLineUnitsDict().update(unitsDict)

        # self.catalog stores the properties for all configurable components
        # as a dictionary of dictionaries which wil be used to recursively
        # initialize the components
        if propDict:
            # propDict contains a only the Application dictionary at the top
            # level
            self.catalog = propDict[list(propDict.keys())[0]]

        self._cmdLineDict = (factDict, docDict, unitsDict)
        return None

    def _getCommandLine(self):
#        if len(sys.argv) < 2:
#            print("An input file is required.")
#            self.Usage()
#            sys.exit(0)
        argv = sys.argv[1:]
        return argv

    ## "Virtual" Usage method
    def Usage(self):
        """
        Please provide a helpful Usage method.
        """
        print("Please provide a Usage method for component, ",
            self.__class__.__name__)
        return
    def help_steps(self):
        """
        Method to print a helpful message when using steps
        """

    def step(self, name, attr=None, local=None, func=None, args=(), delayed_args=(), kwargs={}, dostep=True,
             doc="Please provide a helpful message in the step declaration"):

        if not isinstance(name, str):
            raise ValueError(("The step 'name', given as first argument of a 'step' "+
                              "declaration, is not given as a string"))

        if args and delayed_args:
            raise ValueError("Can only evaluate args or delayed args")

        #add valid step names to the help list
        if isinstance(name, str):
            self.step_list_help.append(name)
        #add valid step names for which dostep==True to the list of steps
        if isinstance(name, str) and dostep:
            self.step_list.append(name)
        self.step_num = len(self.step_list)
        self._dictionaryOfSteps[name] = {'step_index' : self.step_num,
                                         'local' : local,
                                         'attr' : attr,
                                         'func' : func,
                                         'args' : args,
                                         'delayed_args' : delayed_args,
                                         'kwargs' : kwargs,
                                         'doc' : doc}
        return None

    ## Dump Application._pickObj and renderProcDoc().
    def dumpPickleObj(self, name):
        import pickle
        import os
        self.renderProcDoc()
        if not os.path.isdir(self.pickleDumpDir):
            os.mkdir(self.pickleDumpDir)
        if self.renderer == 'xml':
            toDump = getattr(self, self._pickleObj)
            toDump.dump(os.path.join(self.pickleDumpDir, name + '.xml'))
            #dump the procDoc separately
            with open(os.path.join(self.pickleDumpDir, name), 'wb') as PCKL:
                print("Dumping the application's pickle object %s to file  %s" %
                      (self._pickleObj, os.path.join(self.pickleLoadDir, name)))
                pickle.dump(getattr(toDump, 'procDoc'), PCKL,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(self.pickleDumpDir, name), 'wb') as PCKL:
                print("Dumping the application's pickle object %s to file  %s" %
                      (self._pickleObj, os.path.join(self.pickleLoadDir, name)))
                pickle.dump(getattr(self, self._pickleObj), PCKL, protocol=pickle.HIGHEST_PROTOCOL)


        return None


    ## Load Application._pickleObj from Appication.pickleLoadDir
    def loadPickleObj(self, name):
        import  pickle
        import os

        try:
            if self.renderer == 'xml':
                toLoad = self._insarProcFact()
                toLoad.load(os.path.join(self.pickleLoadDir, name + '.xml'))
                setattr(self, self._pickleObj,toLoad)
                with open(os.path.join(self.pickleLoadDir, name), 'rb') as PCKL:
                    setattr(getattr(self, self._pickleObj), 'procDoc',
                            pickle.load(PCKL))

            else:
                with open(os.path.join(self.pickleLoadDir, name), 'rb') as PCKL:
                    setattr(self, self._pickleObj, pickle.load(PCKL))
                    print(
                        "Loaded the application's pickle object, %s from file %s" %
                        (self._pickleObj, os.path.join(self.pickleLoadDir, name))
                        )
        except IOError:
            print("Cannot open %s" % (os.path.join(self.pickleLoadDir, name)))
        return None


    def _processSteps(self):
        import getopt
        start = 0
        startName = self.step_list[0]
        end = self.step_num
        endName = self.step_list[self.step_num-1]


        opts, args = getopt.getopt(self._argopts, 's:e:d:',
                                   ['start=', 'end=', 'dostep=', 'steps', 'next'])
        for o, a in opts:
            if o in ('--start', '-s'):
                startName = a
            elif o in ('--end', '-e'):
                endName = a
            elif o in ('--dostep', '-d'):
                startName = a
                endName = a
            elif o == "--steps":
                pass
            elif o == "--next":
                #determine the name of the most recent pickle file that is in the step_list
                import glob
                pickle_files = glob.glob('PICKLE/*')
                next_step_indx = 0
                while len(pickle_files) > 0:
                    # get the name of the most recent file in the PICKLE directory
                    recent_pname = max(pickle_files, key=os.path.getctime).split('/')[1]
                    # check if pickle rendering is 'xml'
                    if self.renderer == 'xml' and '.xml' in recent_pname:
                        #get the name of the step corresponding to most recent pickle file
                        #with extension ".xml"
                        recent_pname == recent_pname.split(".xml")[0]
                    if recent_pname in self.step_list:
                        next_step_indx = self.step_list.index(recent_pname)+1
                        break
                    else:
                        #remove the filename from the list since it is not in the current step_list
                        pickle_files.pop(pickle_files.index(recent_pname))

                #determine the name of the next step
                if next_step_indx < len(self.step_list):
                    #if the next step index is in the range of possible steps
                    #set 'startName' and 'endName' to the next step
                    startName = self.step_list[next_step_indx]
                    endName = startName
                else:
                    print("Steps has finished the final step. No next step to process.")
                    return
            else:
                print("unhandled option, arg ", o, a)

        if startName in self.step_list:
            start = self.step_list.index(startName)
        else:
            print("ERROR: start=%s is not one of the named steps" % startName)
            return 1

        if endName in self.step_list:
            end = self.step_list.index(endName)
        else:
            print("ERROR: end=%s is not one of the named steps" % endName)
            return 1

        if start > end:
            print(
                "ERROR: start=%s, step number %d comes after end=%s, step number %d"
                %
                (startName, start, endName, end)
                )
            return 1

        if start > 0:
            name = self.step_list[start-1]
            self.loadPickleObj(name)

#        print("self._dictionaryOfSteps['filter'] = ",
#              self._dictionaryOfSteps['filter'])

        for s in self.step_list[start:end+1]:
            print("Running step {}".format(s))
            func = self._dictionaryOfSteps[s]['func']
            args = self._dictionaryOfSteps[s]['args']
            delayed_args = self._dictionaryOfSteps[s]['delayed_args']
            kwargs = self._dictionaryOfSteps[s]['kwargs']
            locvar = self._dictionaryOfSteps[s]['local']
            attr = self._dictionaryOfSteps[s]['attr']

            pargs = ()
            if args:
                for arg in args:
                    pargs += (arg,)
                    pass
                pass
            else:
                for arg in delayed_args:
                    print("eval:",arg)
                    pargs += (eval(arg),)
                    pass
                pass

            result = func(*pargs, **kwargs)
            if locvar:
                locals()[locvar] = result
                pass
            if attr:
                setattr(self, attr, result)
                pass

            self.dumpPickleObj(s)

            if self.step_list.index(s) < len(self.step_list)-1:
                print("The remaining steps are (in order): ",
                      self.step_list[self.step_list.index(s)+1:])
            else:
                print("The is the final step")

            pass # steps loops ends here
        return 0


    def __init__(self, family='', name='',cmdline=None):
        self.name = name
        self._dictionaryOfSteps = {}
        self._argopts = []
        self.step_list = []
        self.step_list_help = []
        self._processCommandLine(cmdline)
        super(Application, self).__init__(family=family, name=name)


        return
