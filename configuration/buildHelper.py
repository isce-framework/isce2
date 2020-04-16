#!/usr/bin/env python3
import sys
import os
import json
tmpdump = 'tmpdump.json'

def createHelp(env,factoryFile,installDir):
    #jng: try to have scons handle all the creation but could not figure out how
    #     so handled dir creation manually
    try:
        os.makedirs(env['HELPER_BUILD_DIR'])
    except:
        # already exists
        pass
    try:
        #one could probably also use __import__ but needs to make sure the
        #the cwd is prepended to the sys.path otherwise if factoryFile = __init__.py
        #it will load the first one found
        moduleList = env['ISCEPATH']
        package = "."
        nameList = []
        for module in moduleList:
            if installDir.count(module):
                ls = installDir.replace(module,'').split("/")
                #remove empty element
                ls = [i for i in ls if i != '']
                package = ".".join(ls)
                #when module is the same as installDir package is empty
                if not package:
                    package = [i for i in installDir.split('/') if i != ''][-1]
                #Since scons at the moment is in python2 adn it calls createHelp
                #in the SCoscript the part that is now in the main
                #might not work since the loading of modules might import some
                #abi3.so modules which are not compatible.
                #To solve we system exec what is in the main using python3
                command = 'python3 ' + os.path.realpath(__file__).replace('.pyc','.py') + ' ' + os.path.join(os.getcwd(),factoryFile) + \
                          ' ' + package + ' ' + env['HELPER_BUILD_DIR']

                if not os.system(command):
                    nameList = json.load(open(tmpdump,'r'))
                    os.remove(tmpdump)
    except:
        nameList = []
    #because the code is run with python2 and 3 during compiling there was
    #RuntimeError: Bad magic number in .pyc file, so remove it
    try:
        os.remove(os.path.realpath(__file__) + 'c')
    except Exception:
        pass
    env.Install(env['HELPER_DIR'],nameList)
    env.Alias('install',env['HELPER_DIR'])
    return nameList,env['HELPER_DIR']

def hasSameContent(dict1,dict2):
    differ = False
    for k1,v1 in dict1.items():
        keyDiffer = True
        for k2,v2 in dict2.items():
            if k1 == k2:
                if isinstance(v1,dict) and isinstance(v2,dict):
                    if not hasSameContent(v1,v2):
                        differ = True
                        break
                else:
                    if isinstance(v1,list):
                        try:
                            if(len(set(v1) & set(v2)) != len(v1)):
                                differ = True
                                break
                        #they are not both lists
                        except Exception:
                            differ = True
                            break

                    elif v1 != v2:
                        differ = True
                        break
                keyDiffer = False
                break
        if differ:
            break
        if keyDiffer:
            differ = True
            break
    return not differ

def compareDict(dict1,dict2):
    if hasSameContent(dict1,dict2) and hasSameContent(dict2,dict1):
        ret = True
    else:
        ret = False
    return ret



def main(factoryFile,package,buildDir):
    ret = 0
#    import isce
    import filecmp
    try:
        from importlib import util
        factoryFile = os.path.abspath(factoryFile)
        mod = util.spec_from_file_location('.', factoryFile)
        factModule = mod.loader.load_module()
        factoriesInfo = factModule.getFactoriesInfo()
        nameList = []
        for k,v in factoriesInfo.items():
            name = os.path.join(buildDir,k + '.hlp')
            v["package"] = package
            if os.path.exists(name):
                toCmp = json.load(open(name))
                if not compareDict(toCmp,{k:v}):
                    json.dump({k:v},open(name,'w'),indent=4)
                    nameList.append(name)
            else:
                json.dump({k:v},open(name,'w'),indent=4)
                nameList.append(name)

        json.dump(nameList,open(tmpdump,'w'))
    except Exception as e:
        print(e)
        ret  = 1

    return ret

if __name__ == '__main__':
    sys.exit(main(sys.argv[1],sys.argv[2],sys.argv[3]))
