#!/usr/bin/env python3
import os, sys
from importlib import util as importlibutil
import argparse
import configparser


# Deals with Configuration file 
class ConfigParser:
  def __init__(self, configFile):
    self.configFile = configFile
    self.funcParams = {}
    self.funcSeq = {}
    self.optionSeq = []

  # Parse Config File
  def readConfig(self):
    # Open the file Cautiously
    with open(self.configFile) as config:
        content = config.readlines()

    # Setting up and reading config 
    Config = configparser.ConfigParser()
    Config.optionxform = str
    Config.read_file(content)

    # Reading the function sequence followed by input parameters 
    # followed by the function parameters
    self.optionSeq = Config.sections()[1:]
    for section in self.optionSeq:
      dictionary, seqOption = self.__readConfigSection(Config, section) 

      # Noting the function name and removing from dictionary
      funcName = seqOption[0]
      del dictionary[funcName]
      self.funcSeq[section] = funcName

      # Creating params for the function
      self.funcParams[section] = self.__dictToParams(funcName, dictionary)

    print("Completed Parsing the Configuration file")

  # Executes the command parsed from the configuaration file
  def runCmd(self):
    import subprocess as SP 
    for section in self.optionSeq:
      ifunc = self.funcSeq[section]
      print("Running: %s"%ifunc) 
      print(self.funcParams[section])
      func_modules = self.__import(ifunc)
      func_modules.main(self.funcParams[section])


  # Converts the dictionary from Config file to parameter list
  def __dictToParams(self, fileName, dictionary):
    params = []
    # Creating params with dictionary
    for key in dictionary.keys():
      if dictionary[key] == 'True':
        # For binary parameters
        params.append('--%s'%key)
      elif not dictionary[key]:
        continue
      elif dictionary[key] == 'False':
        continue
      else:
        params.append('--%s'%key)
        params.append(dictionary[key])
    
    return params


  # Maps each section to its arguments in a dictionary
  def __readConfigSection(self, Config, section):
    import collections
    dict1 = {}
    seqOptions = []
    options = collections.OrderedDict(Config.items(section))
    options = list(options.items())
    
    for option, ip in options:
      dict1[option] = ip
      seqOptions.append(option)

    return (dict1, seqOptions)


  # Importing the functions from the filename
  def __import(self, name, globals=None, locals=None, fromlist=None):
    # Fast path: see if the module has already been imported.
    try:
      return sys.modules[name]
    except KeyError:
      pass

    # If any of the following calls raises an exception,
    # there's a problem we can't handle -- let the caller handle it.
    spec = importlibutil.find_spec(name)

    try:
      return spec.loader.load_module()
    except ImportError:
      print('module {} not found'.format(name))

# Check existence of the input file
def  check_if_files_exist(Files, ftype='input'):
  for ifile in Files:
    if not os.path.exists(ifile):
      print("Error: specified %s file %s does not exist" % (ftype, ifile))
    else:
      print("Reading specified %s file: %s" %(ftype, ifile))

   
# Set up option parser and parse command-line args
def parse_args():

  parser = argparse.ArgumentParser( description='StripMap Processing Wrapper')
  parser.add_argument('-c', type=str, dest='config', default=None,
          help='Specify config file')
  parser.add_argument('-s', type=str, dest='start', default=None,
          help='Specify the start step in the config file. eg: -s Function-2')
  parser.add_argument('-e', type=str, dest='end', default=None,
          help='Specify the end step in the config file. eg: -e Function-3')

  return parser.parse_args()

def main(start = None, end = None):
  # config file creation or parsing
  config = configFile

  # Creating ConfigParser object
  cfgParser = ConfigParser(config)

  # Parse through the configuration file and convert them into terminal cmds
  cfgParser.readConfig()

  # #################################
  if not start is None and not end is None:
     if start in cfgParser.optionSeq and end in cfgParser.optionSeq:
       ind_start = cfgParser.optionSeq.index(start)
       ind_end = cfgParser.optionSeq.index(end)
       cfgParser.optionSeq  = cfgParser.optionSeq[ind_start:ind_end+1]
     else:
       print("Warning start and end was not found")  
  # Run the commands on the Terminal
  cfgParser.runCmd()

if __name__ == "__main__":

  # Parse the input arguments
  args = parse_args()
  configFile = args.config

  # Main engine
  main(args.start,args.end)

