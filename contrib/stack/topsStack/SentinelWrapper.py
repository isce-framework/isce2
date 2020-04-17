#!/usr/bin/env python3
import os, sys
from importlib import util as importlibutil
import argparse
import configparser

# Deals with Configuration file 
class ConfigParser:
  def __init__(self, configFile, fileList, common):
    self.configFile = configFile
    self.fileList = fileList
    self.funcParams = {}
    self.funcSeq = {}
    self.optionSeq = []
    self.common = common

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
    self.common, seqOption = self.__readConfigSection(Config, "Common")
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

  # Generating the config using the file list 
  def generateConfig(self, configFile):

    # Setting up reading config 
    Config = configparser.ConfigParser(delimiters=':')
    Config.optionxform = str

    # Create the section in the Config
    self.__generateConfigSection(Config, "Common", self.common)

    # Reading the parameters from each function in the list 
    for i, ifile in enumerate(self.fileList):
      section = "Function-%s"%(i+1)
      func_modules = self.__import(ifile)
      parser = func_modules.createParser()

      # Reading the arguments for the function
      args, types = self.__readParserArgs(parser) 
      
      # Appending the function name as the first argument
      args.insert(0, ifile)

      # Create the section in the Config
      self.__generateConfigSection(Config, section, args)

    # Writing our configuration file 
    with open(self.configFile, 'w') as configfile:
      Config.write(configfile)

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

  # Writes the arguments in each section
  def __generateConfigSection(self, config, section, subsection):
    config[section] = {}

    # Writing empty section to create the config file
    for isubsec in subsection:
      config[section][isubsec] = ''

  # Looks for string between $ sysmbols in the common subheading in config file
  def __parseString(self, iString):
    if iString is '':
      return iString
    elif isinstance(self.common, (dict)):
      # Case when "common" parameters are read from the configuration file
      for commonStr in self.common.keys():
        key = '$' + commonStr + '$'
        iString = iString.replace(key, self.common[commonStr])
      return iString
    else:
      return iString

  # Maps each section to its arguments in a dictionary
  def __readConfigSection(self, Config, section):
    import collections
    dict1 = {}
    seqOptions = []
    options = collections.OrderedDict(Config.items(section))
    options = list(options.items())
    for option, ip in options:
      dict1[option] = self.__parseString(ip)
      seqOptions.append(option)

    return (dict1, seqOptions)

  # Get attributes from the parser
  def __readParserArgs(self, parser):
    iargs = [] 
    types = []
    numArgs = len(parser._actions) - 1

    # Skipping the help argument
    for i in range(numArgs):
      options = parser._actions[i+1]
      iargs.append(options.option_strings[1][2:])
      types.append(options.type)

    return iargs, types

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

  parser = argparse.ArgumentParser( description='Sentinel Processing Wrapper')
  parser.add_argument('-s', type=str, dest='start', default=None,
          help='Specify the start step in the config file. eg: -s Function-4')
  parser.add_argument('-e', type=str, dest='end', default=None,
          help='Specify the end step in the config file. eg: -e Function-8')
  parser.add_argument('-c', type=str, dest='config', default=None,
          help='Specify config file other than sentinel.ini')
  parser.add_argument('-n', dest='createConfig', action='store_true', default=False,
          help='Create a config file')

  return parser.parse_args()

def main(start = None, end = None):
  # config file creation or parsing
  config = 'sentinel.ini' if configFile is None else configFile

  common = ['outputDir', \
            'masterDir', \
            'slaveDir', \
            'masterOrbit', \
            'slaveOrbit', \
            'dem', \
            'swathnum']

  fileList = ['Sentinel1A_TOPS', \
             'topo',\
             'geo2rdr',\
             'estimateOffsets_withDEM',\
             'derampSlave',\
             'resamp_withDEM',\
             'overlap_withDEM',\
             'estimateAzimuthMisreg',\
             'estimateOffsets_withDEM',\
             'resamp_withDEM',\
             'generateIgram',\
             'merge_withDEM']

  # Creating ConfigParser object
  cfgParser = ConfigParser(config, fileList, common)

  # Empty Configuration creation
  if createConfig is True:
    print("Creating Configuration File")
    cfgParser.generateConfig(config) 
    print("Configuration File Created: %s"%(config))
    return
    
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

  print ("Functions to be executed:")
  print (cfgParser.optionSeq)
  # Run the commands on the Terminal
  cfgParser.runCmd()

if __name__ == "__main__":

  # Parse the input arguments
  args = parse_args()
  configFile, createConfig = args.config, args.createConfig

  # Main engine
  main(args.start,args.end)

