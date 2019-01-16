
import sys
import os
import fnmatch
import Tkinter, tkFileDialog
import xml.etree.ElementTree as ElementTree


class App(Tkinter.Frame):

    def __init__(self,master=None):
        Tkinter.Frame.__init__(self,master)
        self.master.title('ISSI Input File Generator')

        self.filterList = None
        self.filterX = Tkinter.IntVar()
        self.filterY = Tkinter.IntVar()
        self.tec = Tkinter.StringVar()
        self.fr = Tkinter.StringVar()
        self.phase = Tkinter.StringVar()

        self.grid()
        self._buildGUI()


    def findFiles(self,dir):
        """Find a list of the files needed for Faraday Rotation estimation"""
        filenames = {'leader': None,
                     'image': {}}
        # Look for files that start with IMG
        # note, this will only work with JAXA/ASF style CEOS files
        # ERSDAC file nameing structure is not supported
        for root,dirs,files in os.walk(dir):
            for file in files:
                # Find the leader file
                if (fnmatch.fnmatch(file,'LED*')):
                    leaderFile = os.path.join(root,file)
                    filenames['leader'] = leaderFile
                # Find the image files
                elif (fnmatch.fnmatch(file,'IMG*')):
                    polarity = file[4:6]
                    imageFile = os.path.join(root,file)
                    filenames['image'][polarity] = imageFile

        return filenames

    def createImageXML(self,files):
        """Create an XML input file from the dictionary of input files"""

        for polarity in ('HH','HV','VH','VV'):
            output = polarity + '.xml'
            root = ElementTree.Element('component')
            # Leader File
            leaderProperty = ElementTree.SubElement(root,'property')
            leaderName = ElementTree.SubElement(leaderProperty,'name')
            leaderValue = ElementTree.SubElement(leaderProperty,'value')
            leaderName.text = 'LEADERFILE'
            leaderValue.text = files['leader']
            # Image File
            imageProperty = ElementTree.SubElement(root,'property')
            imageName = ElementTree.SubElement(imageProperty,'name')
            imageValue = ElementTree.SubElement(imageProperty,'value')
            imageName.text = 'IMAGEFILE'
            imageValue.text = files['image'][polarity]

            tree = ElementTree.ElementTree(root)
            self.indent(tree.getroot())
            tree.write(output)

    def createAuxilliaryXML(self,output):
        """Create an input file with the default file names"""
        root = ElementTree.Element('component')
        for polarity in ('HH','HV','VH','VV'):
            filename = polarity + '.xml'

            property = ElementTree.SubElement(root,'property')
            name = ElementTree.SubElement(property,'name')
            factoryName = ElementTree.SubElement(property,'factoryname')
            factoryModule = ElementTree.SubElement(property,'factorymodule')
            value = ElementTree.SubElement(property,'value')
            name.text = polarity
            factoryName.text = 'createALOS'
            factoryModule.text = 'isceobj.Sensor'
            value.text = filename

        tree = ElementTree.ElementTree(root)
        self.indent(tree.getroot())
        tree.write(output)

    def createOutputXML(self,output):
        """Create the output xml file"""
        root = ElementTree.Element('component')
        products = {'FILTER': self.filterList.get(),
                    'FILTER_SIZE_X': str(self.filterX.get()),
                    'FILTER_SIZE_Y': str(self.filterY.get()),
                    'FARADAY_ROTATION': self.fr.get(),
                    'TEC': self.tec.get(),
                    'PHASE': self.phase.get()}
        for key in products:
            property = ElementTree.SubElement(root,'property')
            name = ElementTree.SubElement(property,'name')
            value = ElementTree.SubElement(property,'value')
            name.text = key
            value.text = products[key]

        tree = ElementTree.ElementTree(root)
        self.indent(tree.getroot())
        tree.write(output)


    def indent(self,elem, level=0):
        """Indent and XML ElementTree"""
        i = "\n" + level*" "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + " "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def chooseFiles(self):
        """Create a dialog box for the ALOS Quad-pol directory"""
        dir = tkFileDialog.askdirectory(parent=self,title="Choose a directory")
        if (dir):
            files = self.findFiles(dir)
            try:
                self.createImageXML(files)
                self.createAuxilliaryXML('FR.xml')
                self.createOutputXML('output.xml')
                print("XML Files Created")
            except Exception as strerr:
                print(strerr)
                print("No ALOS files found in %s" % (dir))

    def _buildGUI(self):
        """Create widgets and build the GUI"""
        filterLabel = Tkinter.Label(self,text='Choose Filter Type:')
        xSizeLabel = Tkinter.Label(self,text='Range Filter Size')
        ySizeLabel = Tkinter.Label(self,text='Azimuth Filter Size')
        tecLabel = Tkinter.Label(self,text='TEC Output Filename')
        frLabel = Tkinter.Label(self,text='Faraday Rotation Output Filename')
        phaseLabel = Tkinter.Label(self,text='Phase Correction Output Filename')

        self.filterList = Tkinter.Spinbox(self,values=('None','Mean','Median','Gaussian'))
        xSizeEntry = Tkinter.Entry(self,textvariable=self.filterX)
        ySizeEntry = Tkinter.Entry(self,textvariable=self.filterY)
        frEntry = Tkinter.Entry(self,textvariable=self.fr)
        tecEntry = Tkinter.Entry(self,textvariable=self.tec)
        phaseEntry = Tkinter.Entry(self,textvariable=self.phase)
        dirButton = Tkinter.Button(self,text="Choose Data Directory",command=self.chooseFiles)
        quitButton = Tkinter.Button(self,text="Quit",command=self.quit)

        filterLabel.grid(row=0,column=0)
        self.filterList.grid(row=0,column=1)
        xSizeLabel.grid(row=1,column=0)
        xSizeEntry.grid(row=1,column=1)
        ySizeLabel.grid(row=2,column=0)
        ySizeEntry.grid(row=2,column=1)
        frLabel.grid(row=3,column=0)
        frEntry.grid(row=3,column=1)
        tecLabel.grid(row=4,column=0)
        tecEntry.grid(row=4,column=1)
        phaseLabel.grid(row=5,column=0)
        phaseEntry.grid(row=5,column=1)
        dirButton.grid(row=6,column=0)
        quitButton.grid(row=6,column=1)

if __name__ == "__main__":
    """
    Simple example program for creating input files for ISSI.
    """
    app = App()
    app.mainloop()
