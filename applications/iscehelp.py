#!/usr/bin/env python3
#Author:Giangi Sacco
#Copyright 2009-2014, by the California Institute of Technology.
import isce
import os
import sys
import json
import argparse
import collections
import importlib
from iscesys.DictUtils.DictUtils import DictUtils as DU

class Helper(object):

    def getRegistered(self):
        #Register all the factory that want to provide help
        #Each .hlp file has a json structure  like
        '''
        {TypeName
                     {'args':
                           {
                            #positional arguments have as key the position in str format
                            #since json only allows keys to be string
                            '0':{'value':values,'type':type},
                            '1':{'value':values,'type':type}
                            #keyword arguments have the name of the argument as key
                            argname:{'value':values,'type':type,'optional':bool,'default':default}
                            },
                     'factory':factory,
                     'package':package,
                     }
              }
        '''
        registered = {}
        helplist = os.listdir(self._helpDir)
        for name in helplist:
            fullname = os.path.join(self._helpDir,name)
            if not name.endswith('.hlp'):
                continue
            with open(fullname) as fp:
                registered.update(json.load(fp))

        return collections.OrderedDict(sorted(registered.items()))

    def getTypeFromFactory(self,factory):
        instanceType = 'N/A'
        for k,v in self._registered.items():
            if v['factory'] == factory:
                instanceType = k
                break
        return instanceType

    def getInstance(self,typeobj):
        obj2help = self._registered[typeobj]
        args,kwargs = self.getPosAndKwArgs(obj2help)
        factory = getattr(importlib.import_module(obj2help['package']),obj2help['factory'])
        return factory(*args,**kwargs)

    def convert(self,value,type_):

        try:
            module = importlib.import_module('builtins')
            ret = getattr(module,type_)(value)
        except:
            print("Cannot convert",value,"to a type",type_)
            raise Exception
        return ret

    def askHelp(self, instance, steps=False):
        #since it can be called externally, make sure that we remove the
        #arguments that are not understood by the isce Parser
        try:
            sys.argv = [sys.argv[0]]
            instance._parameters()
            instance.initProperties({})
            instance._init()
            instance._facilities()
            instance._dictionaryOfFacilities = DU.renormalizeKeys(instance._dictionaryOfFacilities)
            self.helper(instance, steps)
        except Exception as e:
            print("No help available.")
    def getPosAndKwArgs(self,obj):
        args = []
        kwargs = {}
        if self._inputs.args:#otherwise no args present
            for arg,i in zip(self._inputs.args,range(len(self._inputs.args))):
                try:
                    #positional argument
                    args.append(self.convert(arg,obj['args'][str(i)]['type']))
                except Exception as e:
                    try:
                        kw,val = arg.split("=")
                        kwargs[kw] = self.convert(val,obj['args'][kw]['type'])
                    except Exception as e:
                        print(e)
                        raise

        return (args,kwargs)

    def step_help(self, instance):
            instance.help_steps()
            instance._add_methods()
            instance._steps()
            print()
            print("Command line options for steps processing are formed by")
            print("combining the following three options as required:\n")
            print("'--start=<step>', '--end=<step>', '--dostep=<step>'\n")
            print("The step names are chosen from the following list:")
            print()
            npl = 5
            nfl = int(len(instance.step_list_help)/npl)
            for i in range(nfl):
                print(instance.step_list[i*npl:(i+1)*npl])
            if len(instance.step_list) % npl:
                print(instance.step_list[nfl*npl:])
            print()
            print("If --start is missing, then processing starts at the "+
                "first step.")
            print("If --end is missing, then processing ends at the final "+
                "step.")
            print("If --dostep is used, then only the named step is "+
                "processed.")
            print()
            print("In order to use either --start or --dostep, it is "+
                "necessary that a")
            print("previous run was done using one of the steps options "+
                "to process at least")
            print("through the step immediately preceding the starting "+
                "step of the current run.")
            print()
            sys.exit(0)


    def helper(self,instance,steps=False):
        #if facility is None we print the top level so the recursion ends right away
        #if facility is defined (not None) and is not part of the facilities
        # then keep going down the tree structure

        instance.help()
        print()
        try:
            try:
                #only applications have it
                instance.Usage()
            except Exception:
                pass
            print()
            if steps:
                self.step_help(instance)
                sys.exit(0)
        except Exception as x:
            sys.exit(0)
        finally:
            pass

        #sometime there is no help available. Postpone the printing until
        #there is something to print for sure
        fullMessage = ""
        fullMessage = "\nSee the table of configurable parameters listed \n"
        fullMessage += "below for a list of parameters that may be specified in the\n"
        fullMessage += "input file.  See example input xml files in the isce 'examples'\n"
        fullMessage += "directory.  Read about the input file in the ISCE.pdf document.\n"

#        maxname = max(len(n) for n in self.dictionaryOfVariables.keys())
#        maxtype = max(len(str(x[1])) for x in self.dictionaryOfVariables.values())
#        maxman = max(len(str(x[2])) for x in self.dictionaryOfVariables.values())
#        maxdoc = max(len(x) for x in self.descriptionOfVariables.values())
        maxname = 27
        maxtype = 10
        maxman = 10
        maxdoc = 30
        underman = "="*maxman
        undertype = "="*maxtype
        undername = "="*maxname
        underdoc = "="*maxdoc
        spc = " "
        n = 1
        spc0 = spc*n

        fullMessage += "\nThe user configurable inputs are given in the following table.\n"
        fullMessage += "Those inputs that are of type 'component' are also listed in\n"
        fullMessage += "table of facilities below with additional information.\n"
        fullMessage += "To configure the parameters, enter the desired value in the\n"
        fullMessage += "input file using a property tag with name = to the name\n"
        fullMessage += "given in the table.\n"

        line  = "name".ljust(maxname,' ')+spc0+"type".ljust(maxtype,' ')
        line += spc0+"mandatory".ljust(maxman,' ')+spc0+"doc".ljust(maxdoc,' ')

        fullMessage += line + '\n'

        line = undername+spc0+undertype+spc0+underman+spc0+underdoc

        fullMessage += line + '\n'

        #make sure that there is something to print
        shallPrint = False
        instance.reformatDictionaryOfVariables()
        for x, y in collections.OrderedDict(sorted(instance.dictionaryOfVariables.items())).items():
            #skip the mandatory private. Those are parameters of Facilities that
            #are only used by the framework and the user should not know about 
            if y['mandatory'] and y['private']:
                    continue
            if x in instance.descriptionOfVariables:
                z = instance.descriptionOfVariables[x]['doc']
            elif x in instance._dictionaryOfFacilities and 'doc' in instance._dictionaryOfFacilities[x]:
                z = instance._dictionaryOfFacilities[x]['doc']
            else:
                z = 'N/A'
            shallPrint = True
            try:
                yt = str(y['type']).split("'")[1]
            except:
                yt = str(y['type'])

            lines = []
            self.cont_string = ''
            lines.append(self.columnate_words(x, maxname, self.cont_string))
            lines.append(self.columnate_words(yt, maxtype, self.cont_string))
            lines.append(self.columnate_words(str(y['mandatory']), maxman, self.cont_string))
            lines.append(self.columnate_words(z, maxdoc, self.cont_string))
            nlines = max(map(len,lines))
            for row in lines:
                row += [' ']*(nlines-len(row))
            for ll in range(nlines):
                fullMessage  += lines[0][ll].ljust(maxname,' ')
                fullMessage += spc0+lines[1][ll].ljust(maxtype,' ')
                fullMessage += spc0+lines[2][ll].ljust(maxman,' ')
                fullMessage += spc0+lines[3][ll].ljust(maxdoc,' ') + '\n'
#            line  = spc0+x.ljust(maxname)+spc0+yt.ljust(maxtype)
#            line += spc0+y[2].ljust(maxman)+spc0+z.ljust(maxdoc)
#            print(line)
        if(shallPrint):
            print(fullMessage)
        else:
            print("No help available\n")
        #only print the following if there are facilities
        if(instance._dictionaryOfFacilities.keys()):
            #maxname = max(len(n) for n in self._dictionaryOfFacilities.keys())
            maxname = 20
            undername = "="*maxname

    #        maxmod = max(
    #            len(x['factorymodule']) for x in
    #            self._dictionaryOfFacilities.values()
    #            )
            maxmod = 15
            undermod = "="*maxmod

    #        maxfac = max(
    #            len(x['factoryname']) for x in
    #            self._dictionaryOfFacilities.values()
    #            )
            maxfac = 17
            underfac = "="*maxfac

    #        maxarg = max(
    #            len(str(x['args'])) for x in self._dictionaryOfFacilities.values()
    #            )
            maxarg = 20
            underarg = "="*maxarg

    #        maxkwa = max(
    #            len(str(x['kwargs'])) for x in
    #            self._dictionaryOfFacilities.values()
    #            )
            maxkwa = 7
    #        underkwa = "="*max(maxkwa, 6)
            underkwa = "="*maxkwa
            spc = " "
            n = 1
            spc0 = spc*n
            firstTime = True
            for x, y in collections.OrderedDict(sorted(instance._dictionaryOfFacilities.items())).items():
                #skip the mandatory private. Those are parameters of Facilities that
                #are only used by the framework and the user should not know about
                if y['mandatory'] and y['private']:
                    continue
                #only print if there is something
                if firstTime:
                    firstTime = False
                    print()
                    print("The configurable facilities are given in the following table.")
                    print("Enter the component parameter values for any of these "+
                        "facilities in the")
                    print("input file using a component tag with name = to "+
                        "the name given in")
                    print("the table. The configurable parameters for a facility "+
                        "are entered with ")
                    print("property tags inside the component tag. Examples of the "+
                        "configurable")
                    print("parameters are available in the examples/inputs directory.")
                    print("For more help on a given facility run")
                    print("iscehelp.py -t type")
                    print("where type (if available) is the second entry in the table")
                    print()
        
                    line  = "name".ljust(maxname)+spc0+"type".ljust(maxmod)
        
                    print(line)
                    line  = " ".ljust(maxname)+spc0+" ".ljust(maxmod)
        
                    print(line)
                    line = undername+spc0+undermod
                    print(line)

                lines = []
                self.cont_string = ''
                lines.append(self.columnate_words(x, maxname, self.cont_string))
                z = self.columnate_words(self.getTypeFromFactory(y['factoryname']),maxmod, self.cont_string)
                lines.append(z)

                nlines = max(map(len,lines))
                for row in lines:
                    row += [' ']*(nlines-len(row))
                for ll in range(nlines):
                    out  = lines[0][ll].ljust(maxname)
                    out += spc0+lines[1][ll].ljust(maxmod)
                    print(out)
           
#            line  = spc0+x.ljust(maxname)+spc0+y['factorymodule'].ljust(maxmod)
#            line += spc0+y['factoryname'].ljust(maxfac)
#            line += spc0+str(y['args']).ljust(maxarg)
#            line += spc0+str(y['kwargs']).ljust(maxkwa)
#            print(line)

        return sys.exit(1)
    def columnate_words(self, s, n, cont='',onePerLine=False):
        """
        arguments = s (str), n (int), [cont (str)]
        s is a sentence
        n is the column width
        Returns an array of strings of width <= n.
        If any word is longer than n, then the word is split with
        continuation character cont at the end of each column
        """
        #Split the string s into a list of words
        a = s.split()

        #Check the first word as to whether it fits in n columns
        if a:
            if len(a[0]) > n:
                y = [x for x in self.nsplit(a[0]+" ", n, cont)]
            else:
                y = [a[0]]
            cnt = len(y[-1])
    
            for i in range(1, len(a)):
                cnt += len(a[i])+1
                if cnt <= n:
                    if not onePerLine:
                        y[-1] += " "+a[i]
                    else:
                        y.append(a[i])
                else:
                    y += self.nsplit(a[i], n, cont)
                    if not onePerLine:
                        cnt = len(y[-1])
                    else:
                        cnt = n+1
                
        else:
            y = ['']
        return y

    def nsplit(self, s, nc, cont=''):
        x = []
        ns = len(s)
        n = nc - len(cont)
        for i in range(int(ns/n)):
            x.append(s[i*n:(i+1)*n]+cont)
        if ns%n:
            x.append(s[int(ns/n)*n:])
        return x

    def typeNeedsNoArgs(self,type_):
        try:
            ret = False
            for k,v in self._registered[type_]['args'].items():
                #it's positional so it need the args
                if k.isdigit():
                    ret = True
                    break
                elif (not 'optional' in v) or (not  ('optional' in v and v['optional'])):
                    ret = True
                    break
        except Exception:
            ret = False
        return (not ret)
        
    def printInfo(self,type_,helpIfNoArg = False, steps=False):
        #try to print the info of the arguments necessary to instanciate the instance        
        try:
            sortedArgs = collections.OrderedDict(sorted(self._registered[type_]['args'].items()))
            maxname = 17
            undername = "="*maxname
            maxtype = 10
            undertype = "="*maxtype
            maxargtype = 10
            underargtype = "="*maxargtype
            maxman = 10
            underman = "="*maxman
            maxvals = 20
            undervals = "="*maxvals
            maxdef = 10
            underdef = "="*maxdef
            spc = " "
            n = 1
            spc0 = spc*n
            line  = "name".ljust(maxname,' ')+spc0+"type".ljust(maxtype,' ')+spc0+"argtype".ljust(maxargtype,' ')
            line += spc0+"mandatory".ljust(maxman,' ')+spc0+"values".ljust(maxvals,' ')+spc0+"default".ljust(maxdef,' ')

            fullMessage = line + '\n'

            line = undername+spc0+undertype+spc0+underargtype+spc0+underman+spc0+undervals+spc0+underdef
            shallPrint = False
            fullMessage += line + '\n'
            for arg,val in sortedArgs.items():
                try:                
                    type =  str(val['type'])
                except Exception:
                    type = 'N/A'
                if(arg.isdigit()):
                    argtype = 'positional'
                else:
                    argtype = 'keyword'
                try:
                    mandatory = 'False'  if val['optional'] else 'True'
                except Exception:
                    mandatory  = 'True'
                try:                
                    default =  str(val['default'])
                except Exception:
                    default = 'Not set'
               
                if isinstance(val['value'],list):
                    posarg = ' '.join(val['value'])                        
                elif isinstance(val['value'],str) and val['value']:
                    posarg = val['value']
                else:
                    posarg = ''
                
                lines = []
                self.cont_string = ''
                lines.append(self.columnate_words(arg, maxname, self.cont_string))
                lines.append(self.columnate_words(type, maxtype, self.cont_string))
                lines.append(self.columnate_words(argtype, maxargtype, self.cont_string))
                lines.append(self.columnate_words(mandatory, maxman, self.cont_string))
                lines.append(self.columnate_words(posarg, maxvals, self.cont_string,True))
                lines.append(self.columnate_words(default, maxdef, self.cont_string))

                nlines = max(map(len,lines))
                for row in lines:
                    try:
                        row += [' ']*(nlines-len(row))
                    except:
                        dummy = 1
                for ll in range(nlines):
                    fullMessage  += lines[0][ll].ljust(maxname,' ')
                    fullMessage += spc0+lines[1][ll].ljust(maxtype,' ')
                    fullMessage += spc0+lines[2][ll].ljust(maxargtype,' ')
                    fullMessage += spc0+lines[3][ll].ljust(maxman,' ')
                    fullMessage += spc0+lines[4][ll].ljust(maxvals,' ')
                    fullMessage += spc0+lines[5][ll].ljust(maxdef,' ') + '\n'
                shallPrint = True
#            line  = spc0+x.ljust(maxname)+spc0+yt.ljust(maxtype)
#            line += spc0+y[2].ljust(maxman)+spc0+z.ljust(maxdoc)
#            print(line)
            if(shallPrint):
                print("\nType ",type_, ": Constructor requires arguments described in the\n" + 
                      "table below. Use the -a option with the mandatory arguments\n"+
                       "to ask for more help. Run iscehelp.py -h for more info on the -a option.\n",sep="")

                print(fullMessage)
        except Exception:
            print("\nType ",type_, ": constructor requires no arguments",sep="")
    
        #try to see if one can create an instance and provide more help
        if helpIfNoArg:
            instance = self.getInstance(type_)
            self.askHelp(instance, self._inputs.steps)
           




    def printAll(self):
        for k in self._registered.keys():
            self.printInfo(k)


    def run(self):
        self.parse()
        sys.argv = [sys.argv[0]]

        noArgs = True
        for k,v in self._inputs._get_kwargs():
            if(v):
                noArgs = False
                break

        if self._inputs.info or noArgs:
            #if no arguments provided i.e. self._input has all the attributes = None
            #then print the list of all available helps
            self.printAll()
        elif self._inputs.type and not self._inputs.args:
            #if only -t type is provided print how to get help for that specific type
            self.printInfo(self._inputs.type,helpIfNoArg=self.typeNeedsNoArgs(self._inputs.type))
        elif self._inputs.type and (self._inputs.args):
            #if type and arguments are provided then provide help for that type
            if self._inputs.type in self._registered:
                instance = self.getInstance(self._inputs.type)
                self.askHelp(instance, self._inputs.steps)
            else:
                print("Help for",self._inputs.type,"is not available. Run iscehelp.py"+\
                      " with no options to see the list of available type of objects" +\
                      " one can get help for")
                sys.exit(1)
        elif self._inputs.type and self._inputs.steps and not self._inputs.args:
            #if only -t type is provided print how to get help for that specific type
            self.printInfo(self._inputs.type, helpIfNoArg=True,
                steps=self._inputs.steps)
        elif self._inputs.type and (self._inputs.args) and self._inputs.steps:
            #if type and arguments are provided then provide help for that type
            if self._inputs.type in self._registered:
                instance = self.getInstance(self._inputs.type)
                self.askHelp(instance, self._inputs.steps)
            else:
                print("Help for",self._inputs.type,"is not available. Run iscehelp.py"+\
                      " with -i (--info)  to see the list of available type of objects" +\
                      " one can get help for")
                sys.exit(1)



    def parse(self):
        epilog = 'Run iscehelp.py with no arguments or with -i option to list the available object\n'
        epilog += 'types for which help is provided\n'
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,epilog=epilog)
        parser.add_argument('-i','--info',dest='info',action='store_true',help='Provides the list of registered object types')
        parser.add_argument('-t','--type',dest='type',type=str,help='Specifies the object type for which help is sought')
        parser.add_argument('-a','--args',dest='args',type=str,nargs='+',help='Set of positional and keyword arguments '\
                                                                        +'that the factory of the object "type" takes.'\
                                                                        + 'The keyword arguments are specified as keyword=value with no spaces.')
        parser.add_argument('-s','--steps',dest='steps',action='store_true',help='Provides the list of steps in the help message')

        self._inputs = parser.parse_args()
    def __init__(self):
        import isce
        #the directory is defined in SConstruct
        self._helpDir = os.path.join(isce.__path__[0],'helper')
        self._registered = self.getRegistered()
        self._inputs = None

def main():
    hp = Helper()
    hp.run()
if __name__ == '__main__':
    main()
