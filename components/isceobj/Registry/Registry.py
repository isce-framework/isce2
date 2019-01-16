#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#  http://www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  
#  United States Government Sponsorship acknowledged. This software is subject to
#  U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
#  (No [Export] License Required except when exporting to an embargoed country,
#  end user, or in support of a prohibited end use). By downloading this software,
#  the user agrees to comply with all applicable U.S. export laws and regulations.
#  The user has the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting this software to any 'EAR99'
#  embargoed foreign country or citizen of those countries.
# 
#  Authors: Giangi Sacco, Eric Gurrola
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from collections import OrderedDict
import sys
class Registry:
    class __Registry:
        _registry = None
        _template = None
        def __init__(self):
            #define the keyword used to specify the actual filename
            self._filename = 'name'
            if self._registry is None:
                self._registry = OrderedDict()
            if self._template is None:
                self._template = OrderedDict()
        def _addToDict(self,dictIn,args):
            if(len(args) == 2):#reach the end
                #the last element is a string
                if(isinstance(args[0],str)):
                    dictIn[args[0]] = args[1]
            else:
                if not args[0] in dictIn:
                    dictIn[args[0]] = OrderedDict()
                self._addToDict(dictIn[args[0]],args[1:])

        def _toTuple(self,root,kwargs):
            ret = []
            for k in self._template[root]:
                for k1,v1 in kwargs.items():
                    if(k == k1):
                        ret.append(v1)
                        break
            return tuple(ret)
        def _getFromDict(self,dictIn,args):
            ret  = None
            if(len(args) == 1):
                ret = dictIn[args[0]]
            else:
                ret = self._getFromDict(dictIn[args[0]],args[1:])
            return ret
        def get(self,root,*args,**kwargs):
            ret = self._registry[root]
            if(args):
                ret = self._getFromDict(self._registry[root],args)
            #allow to get values using kwargs so order does not need to be respected
            elif(kwargs):
                argsNow = self._toTuple(root,kwargs)
                ret = self._getFromDict(self._registry[root],args)
            return ret

        def set(self,root,*args,**kwargs):
            #always need to specify the root node
            if not root in self._registry:
                self._registry[root] = OrderedDict()
            if(args):
                self._addToDict(self._registry[root],args)
            #allow to set values using kwargs so order does not need to be respected
            elif(kwargs):
                argsNow = self._toTuple(root,kwargs)
                self._addToDict(self._registry[root],argsNow)




    _instance = None
    def __new__(cls,*args):
        if not cls._instance:
            cls._instance = Registry.__Registry()
        #assume that if args is provided then it creates the template
        if(args):
            argsNow = list(args[1:])
            argsNow.append(cls._instance._filename)
            cls._instance._template[args[0]] = tuple(argsNow)



        return cls._instance
def printD(dictIn,tabn):
    for k,v in dictIn.items():
        print('\t'*tabn[0] + k)
        if(isinstance(v,OrderedDict)):
            tabn[0] += 1
            printD(v,tabn)
        else:
            if not v:
                print('\t'*(tabn[0] + 1) + 'not set yet\n')
            else:
                print('\t'*(tabn[0] + 1) + v + '\n')

    tabn[0] -= 1


def main():
    #create template
    rg = Registry('imageslc','sceneid','pol')
    #add node {'imageslc':{'alos1':{'hh':'image_alos1_hh'} using set
    rg.set('imageslc',pol='hh',sceneid='alos1',name='image_alos1_hh')
    tabn = [0]
    printD(rg._registry['imageslc'],tabn)
    pols = rg.get('imageslc','alos1')
    #add node  {'alos1':{'vv':'image_alos1_vv'} using dict syntax
    pols['vv'] = 'image_alos1_hh'
    tabn = [0]
    printD(rg.get('imageslc'),tabn)
    #add alos2 using positinal
    rg.set('imageslc','alos2','hh','image_alos2_hh')
    tabn = [0]
    printD(rg.get('imageslc'),tabn)
    #change value to test that also the underlying _registry changed
    pols['hh'] = 'I have been changed'
    tabn = [0]
    printD(rg.get('imageslc'),tabn)
    '''
    rg = Registry('imageslc','alos1','hh')
    tabn = [0]
    printD(rg,tabn)
    rg['alos1']['hh'] = 'slc_alos1_hh'
    tabn = [0]
    print('***********\n')
    printD(rg,tabn)
    rg = Registry('imageslc','alos1','hv')
    tabn = [0]
    print('***********\n')
    printD(rg,tabn)
    rg['alos1']['hv'] = 'slc_alos1_hv'
    tabn = [0]
    print('***********\n')
    printD(rg,tabn)
    rg = Registry('imageslc','alos2','hh')
    tabn = [0]
    print('***********\n')
    printD(rg,tabn)
    rg['alos2']['hh'] = 'slc_alos2_hh'
    tabn = [0]
    print('***********\n')
    printD(rg,tabn)
    rg = Registry('imageslc','alos2','hv')
    tabn = [0]
    print('***********\n')
    printD(rg,tabn)
    rg['alos2']['hv'] = 'slc_alos2_hv'
    tabn = [0]
    print('***********\n')
    printD(rg,tabn)
    '''
if __name__ == '__main__':
    sys.exit(main())
