#!/usr/bin/env python3
from __future__ import print_function
import logging
import numbers
import sys


class DictUtils:

   
    @staticmethod
    # if a value for a given key is "empty" (like '',[],{}, None etc, except for zero) then the pair is removed
    def cleanDictionary(dictIn):
        for k,v in list(dictIn.items()):
            if (not v) and not isinstance(v,numbers.Number):
               del dictIn[k]
            #keep going down the tree
            elif isinstance(v,dict):
                DictUtils.cleanDictionary(v)

        return dictIn # doesn't have to return it, but just in case one wants to use it this way instead of passing by ref

    @staticmethod
    def renormalizeKey(s):
        """
        staticmethod renormalizeKey(s):
        Apply renormalization to a dictionary key,
        i.e., transform key to a standard format,
        by removing all white space and canverting
        to lower case.
        """
        from isceobj.Util.StringUtils import StringUtils
        return StringUtils.lower_no_spaces(s)
        
    #renormalize all the keys in the dictionary
    @staticmethod
    def renormalizeKeys(dictNow):
        """
        staticmethod renormalizeKeys(d):
        renormalize all keys in dictionary d by
        applying renormalizeKey static method.
        """
        for k,v in list(dictNow.items()):
            kNow = DictUtils.renormalizeKey(k)
            if kNow != k:
                dictNow[kNow] = dictNow.pop(k)
            if isinstance(v,dict):
                DictUtils.renormalizeKeys(v)
        return dictNow
    #compares keys in dict with an input one. it's case and whitespace insensitive
    #if replace is true it also changes the equivalent key with k
    @staticmethod
    def keyIsIn(k,dictNow,replace = None):
        if(replace == None):
            replace = True
        ret = False
        for k1 in dictNow.keys():
            if (''.join(k1.split())).lower() == (''.join(k.split())).lower():
                if replace:
                    dictNow[k] = dictNow.pop(k1)
                ret =  True
                break
                
        return ret


    @staticmethod
    # update the dictionary dict1 by the value in dict2. 
    # If the key exists and replace = True, then the value is overwritten 
    # otherwise it is appended. 
    # If it does not exist a new node is created. 
    # When replace is True if spare (a list of key or single key) is defined the values of these 
    # keys will be appended if they are not already present. Use it only for str values, i.e. for doc string
    def updateDictionary(dict1,dict2,replace = None,spare = None):
        if replace is None:
            replace = False
        if spare:#if it's a single key, put it into a list
            if isinstance(spare,str):
                spare = [spare]
        else:
            spare = []
       
        # dict1 is the one to update
        for k2,v2 in dict(dict2).items():
            if DictUtils.keyIsIn(k2,dict1):
                if isinstance(v2,dict):#if is a dict keep going down the node
                    DictUtils.updateDictionary(dict1[k2],v2,replace,spare)
                else:
                    if replace:#replace the entry
                        append = False
                        if k2 in spare: #check if the key needs to be spared
                            append = True
                            if isinstance(dict1[k2],list):
                                if v2  in dict1[k2]: # if so then append the content
                                    append = False
                                    break
                            else:
                                if dict1[k2] == v2:
                                    append = False
                                    break
                            if not append:# same key but item already in. it will rewrite it. not a big deal
                                break
                        if append: #convert everything into a list
                            if not isinstance(v2,list):
                                v2 = [v2]
                            if not isinstance(dict1[k2],list):
                                dict1[k2] = [dict1[k2]]
                            #do not append if already there
                            for v22 in v2:
                                if v22 not in dict1[k2]:
                                    dict1[k2].append(v22)
                        else:    
                            dict1.update({k2:v2})
                    else:#update only if is not the same item or the item is not already present (if dict1[k2] is a list)
                        if isinstance(dict1[k2],list):
                            if v2 not in dict1[k2]: # if so then append the content
                                dict1[k2].append(v2) 
                        else:
                            if dict1[k2] != v2:
                                dict1[k2] = [dict1[k2],v2]
                    
            else:
                dict1.update({k2:v2})

    #probably need to create a class with some dictionary utils. put also some of the methods in Parser()
    # if we have a dict of dicts, keeping the structure, extract  a particular key
    # ex. {'n1':{n1_1:{'k1':v1},{'k2':v2},n1_2:{'k1':v11},{'k2':v22}}} extract the 'k2' the result is
    # {'n1':{n1_1:{'k2':v2},n1_2:{'k2':v22}}}. in this case k1 could be the 'doc' string and 'k2' the units
    
    @staticmethod
    def extractDict(dictIn,key):
        import copy
        #put everything i
        dictOut = copy.deepcopy(dictIn)
        DictUtils.searchKey(dictIn,dictOut,key)
        return dictOut
    
    @staticmethod 
    #just  wrapper of the _getDictWithey so the result can be returned instead of being an argument
    def getDictWithKey(dictIn,key,includeKey=True):
        dictOut = {}
        DictUtils._getDictWithKey(dictIn,dictOut,key,includeKey)
        return dictOut

    
    @staticmethod 
    #it returns the first occurance of {key,val} where val is the corresponding value for that key
    #if includeKey is True otherwise returns val
    def _getDictWithKey(dictIn,dictOut,key,includeKey=True):
        if(isinstance(dictIn,dict)):
            for k in dictIn.keys():
                if(k == key):
                    if includeKey:
                        dictOut.update({k:dictIn[k]})
                    else:
                        dictOut.update(dictIn[k])
                    break
                else:
                    DictUtils._getDictWithKey(dictIn[k],dictOut,key,includeKey)
    
    @staticmethod
    #returns a dictionary where all the keys are removed but key
    def searchKey(dictIn,dictOut,key):
        for k,v in dictIn.items():
            if(k == key):
                break
            if isinstance(v,dict):
                DictUtils.searchKey(v,dictOut[k],key)
                if dictOut[k] == {}:#if we removed everything in dictOut[k], then remove the branch
                    dictOut.pop(k)
            
            elif (key != k):#this is a simple pair (k,v) but the key is not the one we want
                dictOut.pop(k)


    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d
    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.iscesys.DictUtils')
    def __init__(self):
        self.logger = logging.getLogger('isce.iscesys.DictUtils')


