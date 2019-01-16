#!/usr/bin/env python3
"""
   The main code. This code will look at the command line arguments. If
   an invalid number of arguments are given, it will return an error.
   Otherwise, it will read the commandline arguments. If one argument
   is given, the code will assume the class name is the same as the
   module name, and try to import the class. Otherwise, it will import
   the given class from the given module and try to make an instance
   of it.
   This code will first try to run ._parameters and ._facilities
   method of the instance. Then, it will check the dictionaryOfVariables
   of the Insar class to see what components may be required. If it is
   not empty, it will make a GUI with the following components:
        - Label to indicate the component name, and whether or not its optional
        - An entry box for the user to input the value for the component
        - Buttons for each facility to allow user to
          change the component of each one
        - A Save button to save the component values, as well as the components
          of the facilities that the user has saved
        - A button to switch between saving a single xml file or saving
          the xml file using multiple xml files
        - A Reset all button, which resets all the inputted data in program
        - A button to allow the user to use an existing xml file to change
          data
        - A quit button to quit the GUI

  Global Variables Used: parameters, dictionaryOfFacilities, facilityButtons,
                         facilityDirs, classInstance, description, allParams,
                         singleFile, directory, facilityParams

"""
import sys
import os
from StringIO import StringIO
import Tkinter as tk
import tkFileDialog, tkMessageBox, tkFont
import xml.etree.ElementTree as ElementTree

import isce
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
#from insarApp import Insar
import traceback
from xml.parsers.expat import ExpatError

"""
Global Definitions:

classInstance - The instance of Insar that is created. This is the instance
                which has the dictionaryOfVariables and dictionaryOfFacilities
                attributes.

allParams - A dictionary of dictionaries containing all the parameters that
            have been set so far.

parameters - a list containing class instances of class parameter, used to
             access the user entry and the name and whether or not it is
             optional in a clean manner.

description - a description of variables for parameters


facilityParams - a list containing instances of class parameter, used
                 to access the user entry for the facility's parameter
                 more easily, similar to global variable parameters.

dictionaryOfFaciliites - the dictionaryOfFacilities, contains the names
                         of all the facilities, as well as its factorymodule,
                         which is the path to the module containing its
                         factoryname, which creates an instance of the
                         facility

facilitiyButtons - The buttons, which causes a GUI for the facility to pop up
                   when pressed. They are disabled when a facility GUI is
                   already present.

facilityDirs - A dictionary containing the locations that the
               user saved the xml file for each key, which is the
               facility name.

root2 - The Tk instance for the second GUI, whcih should be the
        GUI for the facility's parameters.

rootName - The name that the component in the xml is saved under.
           This value is either the name of a facility or 'insarApp'.

directory - The directory at which the most recent file was saved.

singleFile - A boolean which indicates whether or not to save
             the final XML file as a single file or multiple XML in
             catalog format.
"""

class RefactorWarning(DeprecationWarning):
    """put in to alert uses that the code needs to be refactored.
    Take out the raising if you don't like it"""
    pass

class parameter:
    """Class parameter used to keep track of a parameter and its related objects

       Class Members:
           key:      The name of the parameter
           text:     The text widget used for inputting data of this parameter
           optional: Indicates whether or not this parameter is optional
           attrib:   The name this parameter has as an Insar class attribute
    """
    def __init__(self, key=None, text=None, optional=None, attrib = None):
        self.key = key
        self.text = text
        self.optional = optional
        self.attrib = attrib

def indent(elem, level=0):
    """Indent an XML ElementTree"""
    i = "\n" + level*"    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


##    Creates the Input XML file given the user's inputs.
##    If the user has missed a mandatory field in the current level GUI,
##    this will cause a pop-up box to appear and tell the user to
##    fill in the mandatory fields. Otherwise, it will ask the
##    user for a directory to save the xml file in and create the
##    xml file given their inputs. If making the final xml file,
##    i.e the input file for the insarApp, it will also add any
##    directories created by using a catalog.
##
##    global variables used - directory, facilityDirs, facilityButtons,
##                            singleFile, allParams
def createInputXML(parameters, rootName):
    """Creates the Input XML File given the user inputs
       Arguments:
          parameters - A list of parameters to be inputted into the xml file
          rootName - The name of the root
    """
    # Get necessary global variables
    global directory
    global facilityDirs
    global facilityButtons
    global facilityRequired
    global singleFile
    global allParams
    # Checks if any of the manadatory fields are blank.
    for param in parameters:
        if(not(param.optional) and param.text.get()==''):
            tkMessageBox.showerror('ERROR!', 'Mandatory Field(s) is blank!')
            return False
    # If rootName is insarApp, and it is in multi file XML mode,
    # then the user should have, by either loading an XML which is
    # in that form or creating multiple files, a file for each facility.
    if(rootName == 'insarApp' and not singleFile):
        for x in zip(facilityButtons,facilityRequired):
            button = x[0]
            req = x[1]
            try:
                if(facilityDirs[button.cget('text')]=='' and req):
                    raise KeyError
            except KeyError:
                tkMessageBox.showerror('ERROR!',
                                       'Facility parameters not saved in a file for:\n' +
                                        button.cget('text'))
                return False
    # If rootName is insarApp and it is in single file XML mode,
    # then the user should have, by either loading an XML file or
    # by inputting and saving, have data for each facility.
    elif(rootName == 'insarApp' and singleFile):
        for x in zip(facilityButtons,facilityRequired):
            button = x[0]
            req = x[1]
            try:
                if(allParams[button.cget('text')] == {} and req):
                    raise KeyError
            except KeyError:
                tkMessageBox.showerror('ERROR!',
                                       'Facility parameters not set in:\n' +
                                        button.cget('text'))
                return False
    # Get a directory from the user to save in if we are in multi file XML
    # mode and/or is saving the insarApp input file.
    if(not singleFile or rootName == 'insarApp'):
        directory = tkFileDialog.asksaveasfilename(initialfile=rootName+'.xml',
                                             title="Choose where to save:",
                                             defaultextension='.xml',
                                             filetypes=[('xml files', '.xml')])
        if(not directory):
            return False
        else:
            # Create the input xml file using ElementTree.
            top = ElementTree.Element(rootName)
            top.text='\n'
            root = ElementTree.SubElement(top,'component', {'name':rootName})
            for param in parameters:
                if(param.text.get()!=''):
                    property = ElementTree.SubElement(root,'property', {'name':param.key})
                    value = ElementTree.SubElement(property,'value')
                    value.text = param.text.get()
            # If this is the insarApp input file, we must put the
            # directory of all the input xml files for the facilities
            if(rootName == 'insarApp'):
                # If we are in sigleFile mode, write all the parameters
                # into the file that we were writing to.
                if singleFile:
                    for key in allParams.keys():
                        if allParams[key]:
                             facility = ElementTree.SubElement(root, 'component', {'name':key})
                             for paramKey in allParams[key].keys():
                                 if allParams[key][paramKey]:
                                     param = ElementTree.SubElement(facility, 'property',
                                                                    {'name':paramKey})
                                     value = ElementTree.SubElement(param, 'value')
                                     value.text = allParams[key][paramKey]
                # Otherwise, write the directory of each facility into
                # the file that we were writing to.
                else:
                    for key in facilityDirs.keys():
                        if facilityDirs[key]:
                            property = ElementTree.SubElement(root, 'component', {'name':key})
                            catalog = ElementTree.SubElement(property, 'catalog')
                            catalog.text = facilityDirs[key]
            # Write the file using ElementTree
            # If the file we are saving is the insarApp input file,
            # we want insarApp tag on top of it. Otherwise, just
            # put the data in to the xml file
            if(rootName == 'insarApp'):
                tempTree = ElementTree.ElementTree(root)
                indent(tempTree.getroot())
                tree = ElementTree.ElementTree(top)
            else:
                tree = ElementTree.ElementTree(root)
                indent(tree.getroot())
            tree.write(directory)
    # Since the user is saving a facility in the single file XML mode,
    # save the values in the global variable allParams
    else:
        allParams[rootName] = {}
        for param in parameters:
            allParams[rootName][param.key] = param.text.get()
    return True


##   Creates the input XML for a toplevel GUI, which
##   should be for the facility's components. After
##   saving the XML file, it will exit the toplevel
##   GUI and save the directory that it was saved to
##   in a dictionary with the key as the name of the
##   facility.
##
##   global variables used - facilityComponents, dir, rootName, facilityDirs
def facilityInputXML():
    """Creates an XML file for a facility's parameters"""
    global facilityParams
    global directory
    global rootName
    global facilityDirs
    # Create the XML using the facilityParameters
    # and the rootName, which was set as the facility name
    # when the facility GUI was made
    if(createInputXML(facilityParams, rootName)):
        facilityQuit()
    if(directory):
        facilityDirs[rootName] = directory
    return


##   Creates the input XML for insarApp, which is
##   at the root.
def componentInputXML():
    """Creates an XML file for the InsarApp"""
    global parameters
    global facilityDirs
    createInputXML(parameters, 'insarApp')

###The event that is called when a facilityButton is
##   pressed by the user. When the button is pressed,
##   the code will first try to create an instance of
##   the class using the argument given in the
##   dictionaryOfFacilities and the method given in it.
##   If it fails, it will return an error
##   message, indicating a matching argument for the method
##   was not found. If it succeeds, it will disable the facility
##   buttons, since we can only have one other GUI open at once.
##   Then, it will also disable the inputs to the components,
##   since those should not be changed, since the facility could
##   depend on the values. It will then proceed to make
##   a GUI with entries for each component found in the
##   attribute dictionaryOfVariables of the instance.
def facilityEvent(event):
    """Creates a pop-up GUI for inputting facility parameters"""
    # Load all the global variables used in this function
    global parameters
    global dictionaryOfFacilities
    global facilityButtons
    global facilityParams
    global rootName
    global root2
    global classInstance
    global singleFile
    global allParams
    global facilityDocs
    # Find which facility button the user pressed
    # through its text, and set it as the rootName
    text = event.widget.cget('text')
    rootName = text
    # Initiate instance as None
    instance = None
    # Initiate a StringIO and set it as stdout to
    # catch any error messages the factory
    # method produces
    temp = sys.stdout
    errorStr = StringIO('')
    sys.stdout = errorStr
    # Call the parameters method to restore the
    # default value of facilities
    try:
        classInstance._parameters()
    except:
        pass
    for param in parameters:
        if param.text.get():
#            exec 'classInstance.' + param.attrib + '= \'' + param.text.get() + '\''
            setattr(classInstance, param.attrib, eval('\'' + param.text.get() + '\''))

            pass
        pass
    try:
        classInstance._facilities()
    except:
        pass
    # Try to use the arguments in the dictionaryOfFacilities
    # to instantiate an instance of the facility
    try:
        args = dictionaryOfFacilities[text]['args']
        kwargs = dictionaryOfFacilities[text]['kwargs']
        # May need to be modified if a factory takes
        # the None argument
        modified = ['']*len(args)
        for i in range(0, len(args)):
            if(args[i] == None):
                modified[i] = 'None'
            else:
                modified[i] = args[i]
                pass
            pass
        modified = tuple(modified)
#        raise RefactorWarning("refactor with appy built-in")
        instance = eval(
            dictionaryOfFacilities[text]['factoryname']+'(*' + modified.__str__() + ', **' +
            kwargs.__str__() + ')'
            )
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        tkMessageBox.showerror('ERROR!', 'Unknown error occurred:\n'+errorStr.getvalue()+'\n%s' %e)
        return None
    # If the instance is still none, this means
    # that an error message was produced, and
    # that it failed to make an instance.
    # Print out the error message
    # produced, which is contained in the StringIO
    sys.stdout = temp
    if instance is None:
        tkMessageBox.showerror('ERROR!', 'Bad argument for: ' +
                               dictionaryOfFacilities[text]['factoryname'] +
                               '\n' + errorStr.getvalue())
        return
    # Try to run the ._parameters() and ._facilities()
    # methods of the instance, and then get its
    # dictionaryOfVariables
    try:
        instance._parameters()
    except:
        pass
    try:
        instance._facilities()
    except:
        pass
    dictionaryOfVariables = None
    try:
        dictionaryOfVariables = instance.dictionaryOfVariables
    except:
        pass
    # Check if the dictionaryOfVariables is empty or does not exist
    if (dictionaryOfVariables is None or dictionaryOfVariables == {}):
        # Create a Popup Error message
        sys.stdout = sys.stderr
        tkMessageBox.showerror('ERROR!', 'DictionaryOfVariables for ' +
                               text + ' is empty! Nothing to do...')
        return
    # Disable all the facilityButtons b/c multiple facility
    # GUI's are not supported
    for button in facilityButtons:
        button.config(state='disabled')
    for param in parameters:
        param.text.config(state='disabled')
    XMLButton.config(state='disabled')
    # Create the new facility GUI
    root2 = tk.Toplevel()
    root2.protocol("WM_DELETE_WINDOW",facilityQuit)
    root2.title('Facility '+text+ ' Component Editor')
    tempFont = ('Times New Roman', 14)
    # Create a font with underlines
    uFont = tkFont.Font(family='Times New Roman', size=14, underline=True)
    # First column gives the name
    nameLabel = tk.Label(root2, text='Name (Click a name for help)', font=uFont)
    # Second column allows user to input values for each attribute
    valueLabel = tk.Label(root2, text='Value', font=uFont)
    # The third column is for units
    unitsLabel = tk.Label(root2, text='Units', font=uFont)
    # The fourth column indicates to users whether or not an
    # attribute is optional or mandatory.
    requiredLabel = tk.Label(root2, text='Optional/Mandatory', font=uFont)
    # Put each label in respective locations
    nameLabel.grid(row=0, column=0)
    valueLabel.grid(row=0, column=1)
    unitsLabel.grid(row=0, column=2)
    requiredLabel.grid(row=0, column=3)
    r = 1
    # Reset facilityParams, since we are using a new
    # facility
    facilityParams = []
    try:
        units = instance.unitsOfVariables
    except:
        pass
    try:
        facilityDocs = instance.descriptionOfVariables
    except:
        pass
    for key in dictionaryOfVariables.keys():
        label = tk.Label(root2, text=key)
        label.grid(row=r, column=0)
        if(dictionaryOfVariables[key][2].lower() == 'optional'):
            opt = tk.Label(root2, text='Optional', fg='green')
            facilityParams.append(parameter(key, tk.Entry(root2), True))
        else:
            opt = tk.Label(root2, text='Mandatory', fg='red')
            facilityParams.append(parameter(key, tk.Entry(root2), False))
        try:
            label = tk.Label(root2, text=units[key])
            label.grid(row=r, column=2)
        except:
            pass
        button = tk.Button(root2, text=key, width=25)
        button.bind('<ButtonRelease>', facilityHelp)
        button.grid(row=r, column=0)
        opt.grid(row=r, column=3)
        facilityParams[r-1].text.grid(row=r, column=1)
        r = r + 1
    # Put the known arguments into the entry boxes before outputting
    # them, and also check for any "trash" values inside the dictionary
    # that could occur from loading an xml file with incorrect facility
    # parameters
    temp = {}
    temp[text] = {}
    for param in facilityParams:
        try:
            param.text.insert(0, allParams[text][param.key])
            temp[text][param.key] = allParams[text][param.key]
        except:
            pass
    allParams[text] = temp[text]
    # Create a quit and save button, as well as a dir button so
    # that the user can load a directory and use that as their
    # facility XML file
    quitButton = tk.Button(root2, text='Quit', command=facilityQuit)
    saveButton = tk.Button(root2, text='Save', command=facilityInputXML)
    dirButton = tk.Button(root2, text='Use An Existing\n XML File',
                          command=getFacilityDirectory)
    quitButton.grid(row=r, column=2)
    saveButton.grid(row=r, column=1)
    dirButton.grid(row=r, column=0)
    root2.mainloop()

def facilityHelp(event):
    """Creates help documentation for the facility GUI"""
    global facilityDocs
    text = event.widget.cget('text')
    if(text in facilityDocs.keys() and facilityDocs[text] != ''):
        tkMessageBox.showinfo(text+' documentation:', description[text])
    else:
        tkMessageBox.showerror('Documentation Not Found!', 'There is no documentation\nfor this parameter')


##   This method is called when the button for using an already existing
##   XML file is clicked on the facility GUI. The method tries to open
##   the xml file given, and stores the data in the global variable
##   allParams, as well as populate them in the GUI's entry boxes.
##
##   Global Variables Used: rootName, facilityDirs, facilityParams
def getFacilityDirectory():
    """Gets the directory for the xml used for the facility's parameter"""
    global rootName
    global facilityDirs
    global facilityParams
    directory = tkFileDialog.askopenfilename(title='Locate Your XML File for '
                                       + rootName, defaultextension='.xml',
                                       filetypes=[('xml files', '.xml')])
    if(directory):
        try:
            tree = ElementTree.parse(directory)
            value = ''
            name = ''
            for property in tree.findall('property'):
                name = property.attrib['name']
                value = property.find('value').text
                for param in facilityParams:
                    if param.key == name:
                        param.text.delete(0, tk.END)
                        param.text.insert(0, value)
                        allParams[rootName][param.key] = value
                        name = ''
                        break
                if name != '':
                    tkMessageBox.showerror('Error!', 'Invalid XML for'+
                                                     rootName +  ' facility!'
                                                     + '\nParameter ' + name +
                                                     ' does not exist in this facility!')
                    return
        except ExpatError:
            tkMessageBox.showerror('Error!', 'Invalid XML error! XML is ill formed!')
        except Exception:
            tkMessageBox.showerror('Error!', 'Invalid XML error! XML is ill formed for ' + rootName + '!')
        facilityDirs[rootName] = directory

##   This is the quit button event for the facility GUI. This
##   quits out of the for facility and reenables all the
##   buttons for the other facilities and entry boxes for
##   the components.
##
##   Global Variables Used: facilityButtons, components, root2, XMLButton
def facilityQuit():
    """The button event for Quit button on facility GUI. This destroys the
       facility GUI and restores disabled buttons on main GUI."""
    root2.destroy()
    for button in facilityButtons:
        button.config(state='normal')
    for param in parameters:
        param.text.config(state='normal')
    XMLButton.config(state='normal')

def showDoc(event):
    """Shows documentation for the parameter written on the button"""
    text = event.widget.cget('text')
    if(text in description.keys() and description[text] != ''):
        tkMessageBox.showinfo(text+' documentation:', description[text])
    else:
        tkMessageBox.showerror('Documentation Not Found!', 'There is no documentation\nfor this parameter')

def changeSave(event):
    """Changes the save from single file save to multiple and vice versa"""
    global singleFile
    global facilityDirs
    singleFile = not singleFile
    if(singleFile):
        event.widget.configure(text='Currently:\nSingle XML File Mode')
        facilityDirs = {}
    else:
        event.widget.configure(text = 'Currently:\nMultiple XML Mode')
    return

def loadXML():
    """Loads an XML file for the insarApp and stores the data"""
    global parameters
    global allParams
    global facilityDirs
    facilityDirs = {}
    # Get the directory from the user
    directory = ''
    directory = tkFileDialog.askopenfilename(title='Locate Your XML File:',
                                             defaultextension='.xml',
                                             filetypes=[('xml files', '.xml')])
    # If the user specified a directory, try loading it
    if directory:
        try:
            # Find the insarApp component which should have all the properties
            # and facilities
            tree = ElementTree.parse(directory).find('component')
            text = ''
            name = ''
            # First find all the parameters listed in the main GUI
            for property in tree.findall('property'):
                name = property.attrib['name']
                value = property.find('value').text
                for param in parameters:
                    if param.key == name:
                        param.text.delete(0, tk.END)
                        param.text.insert(0, value)
                        name = ''
                        break
                    pass
                if name:
                    tkMessageBox.showerror('Error!', 'Invalid xml for these parameters!\n'+
                                           'Parameter ' + name + ' does not exist!')
                    pass
                pass

            # Then find the parameters for the facilities
            for facility in tree.findall('component'):
                exists = False
                facilityName = facility.attrib['name']
                for button in facilityButtons:
                    if button.cget('text') == facilityName:
                        exists = True
                        pass
                    pass
                if not exists:
                    tkMessageBox.showerror('Error!',  'Invalid xml error! Facility '
                                           + facilityName + ' does not exist!')
                    return None
                # Check whether or not the xml is in catalog format or all-in-one
                # format
                catalog = None
                catalog = facility.find('catalog')
                allParams[facilityName] = {}
                # If there is a catalog, assume that the first component
                # contains every parameter of the facility
                if catalog is not None:
                    catalog = catalog.text
                    facilityDirs[facilityName] = catalog
                    facilityTree = ElementTree.parse(catalog)
                    for property in facilityTree.findall('property'):
                        name = property.attrib['name']
                        value = property.find('value').text
                        allParams[facilityName][name] = value
                        pass
                    pass
                # Otherwise, go through the facility and get the parameters
                else:
                    for property in facility.findall('property'):
                        name = property.attrib['name']
                        value = property.find('value').text
                        allParams[facilityName][name] = value
        except IOError:
            tkMessageBox.showerror('Error!', 'Invalid XML error! One or more XML does not exist!')
        except ExpatError:
            tkMessageBox.showerror('Error!', 'Invalid XML error! XML is ill formed!')
        except Exception:
            tkMessageBox.showerror('Error!', 'Invalid XML error! XML is valid for insarApp!')
    return



def reset():
    """After asking the user, resets everything in the code used for writing to an xml"""
    global allParams
    global facilityDirs
    global parameters
    global facilityButtons
    global root2
    # Ask the user if they want to reset everything
    answer = tkMessageBox.askyesno("Are you sure?", "Are you sure you want to reset all data?")
    if answer:
        # Delete all entries in the main GUI
        for param in parameters:
            param.text.delete(0, tk.END)
        # Erase all data stored for writing to XML's
            allParams = {}
        facilityDirs = {}
        # Make sure that all the main GUI buttons are enabled
        for button in facilityButtons:
            button.configure(state='normal')
            facilityDirs[button.cget('text')] = ''
            allParams[button.cget('text')] = {}
        XMLButton.config(state='normal')
        # If there is a facility GUI, get rid of it
        try:
            root2.destroy()
        except:
            pass
        pass
    pass


if __name__ == "__main__":
    """Builds the main GUI for making an XML input for given class"""
    # Get the global variable
    global parameters
    global dictionaryOfFacilities
    global facilityButtons
    global facilityRequired
    global facilityDirs
    global classInstance
    global description
    global allParams
    global singleFile
    global directory
    global facilityParams
    parameters = []
    facilityParams = []
    dictionaryOfFacilities = {}
    facilityButtons = []
    facilityRequired = []
    facilityDirs = {}
    root2 = None
    rootName = ''
    directory = ''
    allParams = {}

    # Create an instance of Insar to run the _parameters() and
    # _facilities() function, if they exist, to create the
    # dictionaryOfVariables.
    try:
        if(len(sys.argv) != 2 and len(sys.argv) != 3):
            print("Invalid commandline arguments:")
            print("Usage 1, Module and Class have same names: xmlGenerator Module")
            print("Usage 2, Module and Class names different: xmlGenerator Module Class")
            print("(Module name should not include the '.py')")
            sys.exit()
        elif(len(sys.argv) == 2):
            if 'help' in sys.argv[1]:
                print("'Invalid commandline arguments:\nUsage: xmlGenerator [Module (sans '.py'] [Class]")
#            raise RefactorWarning("refactor with __import__ built-in")
            print("Assuming module name and class name are both, ", sys.argv[1])
            exec('from ' + sys.argv[1] + ' import ' + sys.argv[1])
            classInstance = eval(sys.argv[1] + '()')
        else:
            print("importing class %s from module %s" % (sys.argv[1], sys.argv[2]))
#            raise RefactorWarning("refactor with __import__ built-in")
            exec('from ' + sys.argv[1] + ' import ' + sys.argv[2])
#            print sys.argv[2]
            classInstance = eval(sys.argv[2] + '()')
            pass
        pass
    except ImportError as e:
        print("Invalid arguments!")
        print("Either the given module or the given class does not exist,")
        print("or you have assumed they both have the same name and they do not.")
        sys.exit()
        pass
    try:
        classInstance._parameters()
        classInstance._facilities()
    except:
        pass
    dictionaryOfVariables = classInstance.dictionaryOfVariables
    try:
        dictionaryOfFacilities = classInstance._dictionaryOfFacilities
    except:
        pass

    # If the dictionaryOfVariables is not empty, create
    # the GUI
    if dictionaryOfVariables:

        # Since Frame class does not have scrollbars, use a
        # canvas to create a scrollbar in the y direction
        root = tk.Tk()
        root.title(sys.argv[1] + ' Input XML File Generator')
        verticalBar = tk.Scrollbar(root)
        verticalBar.grid(row=0, column=1, sticky='N'+'S')

        # Create the Canvas, which will have the scroll bar as
        # well as the frame. Change the width here to
        # change the starting width of the screen.
        canvas = tk.Canvas(root,
                           yscrollcommand=verticalBar.set,
                           width=1100, height=500)
        canvas.grid(row=0, column=0, sticky='N'+'S'+'E'+'W')
        verticalBar.config(command=canvas.yview)

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)


        frame = tk.Frame(canvas)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(1, weight=1)
        # Begin creating the GUI involved with input variables
        # Create a font with underlines
        uFont = tkFont.Font(family='Times New Roman', size=14, underline=True)
        # Create a parameters label
        paramLabel = tk.Label(frame, text='Parameters:',
                              font=("Times New Roman", 20, "bold"))
        # First column gives the name
        nameLabel = tk.Label(frame, text='Name (Click a name for help)', font=uFont)
        # Second column allows user to input values for each attribute
        valueLabel = tk.Label(frame, text='Value', font=uFont)
        # The third column is for units
        unitsLabel = tk.Label(frame, text='Units', font=uFont)
        # The fourth column indicates to users whether or not an
        # attribute is optional or mandatory.
        requiredLabel = tk.Label(frame, text='Optional/Mandatory', font=uFont)
        # Put each label in respective locations
        paramLabel.grid(row=0, column=0)
        nameLabel.grid(row=1, column=0, columnspan=2)
        valueLabel.grid(row=1, column=2)
        unitsLabel.grid(row=1, column=4)
        requiredLabel.grid(row=1, column=5)

        # Create a variable for the row
        r = 2
        try:
            description = classInstance.descriptionOfVariables
        except:
            pass
        units = {}
        try:
            units = classInstance.unitsOfVariables
        except:
            pass
        for key in dictionaryOfVariables.keys():
            val = dictionaryOfVariables[key]
            # Make the label from the keys in the dictionary
            # Change the wraplength here for the names if it is too short or long.
            # label = tk.Label(frame, text=key, anchor = tk.W, justify=tk.LEFT, wraplength=100)
            # label.grid(row=r,column=0)
            # Indicate whether the attribute is optional or mandatory
            if(val[2].lower() == ('optional')):
                required = tk.Label(frame, text='Optional', fg='green')
                parameters.append(parameter(key, tk.Entry(frame, width=50), True, val[0]))
            else:
                required = tk.Label(frame, text='Mandatory', fg='red')
                parameters.append(parameter(key, tk.Entry(frame, width=50), False, val[0]))
                pass
            try:
                doc = tk.Button(frame, text=key, anchor = tk.W, justify=tk.LEFT, width=50,
                                wraplength=348)
                doc.bind('<ButtonRelease>', showDoc)
                doc.grid(row=r, column=0, columnspan=2)
            except:
                pass
            try:
                unit = tk.Label(frame, text=units[key])
                unit.grid(row=r, column=2)
            except:
                pass
            required.grid(row=r,column=5)
            # Put the Entry in global variable, since it is needed
            # for saving inputted values into xml
            parameters[r-2].text.grid(row=r,column=2, columnspan=2)
            r = r + 1
            pass
        if dictionaryOfFacilities:
            # Add a label indicating that these buttons are facilities
            facilityLabel = tk.Label(frame, text='Facilities:',
                                     font=("Times New Roman", 20, "bold"),
                                     justify=tk.LEFT,
                                     anchor=tk.W)
            facilityLabel.grid(row=r, column=0)
            r = r + 1
            x = 0
            # Make the buttons to edit facility parameters and import
            # the required modules using the factorymodule
            for key in dictionaryOfFacilities.keys():
                facilityButtons.append(tk.Button(frame, text = key, width=50, justify=tk.LEFT,
                                                 anchor=tk.W, wraplength=348))
                facilityButtons[x].grid(row=r, column=0, columnspan=2)
                facilityButtons[x].bind('<ButtonRelease>', facilityEvent)
                facilityDirs[key] = ''
                allParams[key] = {}
                if dictionaryOfFacilities[key]['mandatory']:
                    facilityRequired.append(True)
                    required = tk.Label(frame, text='Mandatory', fg='red')
                    required.grid(row=r,column=5)
                else:
                    facilityRequired.append(False)
                    required = tk.Label(frame, text='Optional', fg='green')
                    required.grid(row=r,column=5)

                r = r + 1
                x = x + 1
                try:
                    exec ('from ' + dictionaryOfFacilities[key]['factorymodule'] +
                          ' import ' +  dictionaryOfFacilities[key]['factoryname'])
                    raise RefactorWarning("refactor with __import__ built-in")
                except:
                    pass
                pass
            pass
        # Buttons for saving the xml file, using an existing xml file,
        # changing the save settings, and quitting out of the program
        saveButton = tk.Button(frame, text="Save", command=componentInputXML)
        quitButton = tk.Button(frame, text="Quit", command=root.destroy)
        resetButton = tk.Button(frame, text='Reset All', command=reset)
        # The button for switching between multiple xml mode and single
        # mode. The default is multiple XML mode.
        singleFile = False
        singleFileButton = tk.Button(frame, text='Currently:\nMultiple XML Mode')
        singleFileButton.bind('<ButtonRelease>', changeSave)
        # The button used to get an existing XML file
        XMLButton = tk.Button(frame, text='Use an existing XML File', command=loadXML)
        saveButton.grid(row=r+1, column=2)
        quitButton.grid(row=r+1, column=3)
        resetButton.grid(row=r+1, column=4)
        singleFileButton.grid(row=r+1, column=5)
        XMLButton.grid(row=r+1, column=1)
        # Have the canvas create a window in the top left corner,
        # which is the frame with everything on it
        canvas.create_window(0, 0, anchor='nw', window=frame)
        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        root.mainloop()
    else:
        tkMessageBox.showerror('ERROR!', 'Dictionary of Variables Empty: Nothing to do')
        pass
    sys.exit()
