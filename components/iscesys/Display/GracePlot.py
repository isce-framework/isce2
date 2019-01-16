#!/usr/bin/env python3 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
from iscesys.Compatibility import Compatibility



class GracePlot(object):
  """
  GracePlot is a class to facilitate creation of Grace Plots in batch mode
  from data in memory.
  """

  def __init__(self):
    import time
    import random
    t = time.localtime()
    r = int(100000*random.random())
    self._defaultFile = "gracetemp_"+str(t[0])+\
        str(t[1])+\
        str(t[2])+\
        str(t[3])+\
        str(t[4])+\
        str(t[5])+\
        "_"+str(r)+".agr"
    self._filename = ''
    self._psfile =''
    self. _weekday = ["Mon",\
                        "Tue",\
                        "Wed",\
                        "Thu",\
                        "Fri",\
                        "Sat",\
                        "Sun"]
    self._month = ["",\
                     "Jan",\
                     "Feb",\
                     "Mar",\
                     "Apr",\
                     "May",\
                     "Jun",\
                     "Jul",\
                     "Aug",\
                     "Sep",\
                     "Oct",\
                     "Nov",\
                     "Dec"]
    
    self._fontsList = ["Times-Roman",\
                       "Times-Italic",\
                       "Times-Bold",\
                       "Times-BoldItalic",\
                       "Helvetica",\
                       "Helvetica-Oblique",\
                       "Helvetica-Bold",\
                       "Helvetica-BoldOblique",\
                       "Courier",\
                       "Courier-Oblique",\
                       "Courier-Bold",\
                       "Courier-BoldOblique",\
                       "Symbol",\
                       "ZapfDingbats"]

    self._fonts = {"Times-Roman":          0,\
                   "Times-Italic":         1,\
                   "Times-Bold":           2,\
                   "Times-BoldItalic":     3,\
                   "Helvetica":            4,\
                   "Helvetica-Oblique":    5,\
                   "Helvetica-Bold":       6,\
                   "Helvetica-BoldOblique":7,\
                   "Courier":              8,\
                   "Courier-Oblique":      9,\
                   "Courier-Bold":        10,\
                   "Courier-BoldOblique": 11,\
                   "Symbol":              12,\
                   "ZapfDingbats":        13}

    self._colorsList = ["white",\
                        "black",\
                        "red",\
                        "green",\
                        "blue",\
                        "yellow",\
                        "brown",\
                        "grey",\
                        "violet",\
                        "cyan",\
                        "magenta",\
                        "orange",\
                        "indigo",\
                        "maroon",\
                        "turquoise",\
                        "green4"]

    self._colors = {"white":    ( 0,(255, 255, 255)),\
                    "black":    ( 1,(  0,   0,   0)),\
                    "red":      ( 2,(255,   0,   0)),\
                    "green":    ( 3,(  0, 255,   0)),\
                    "blue":     ( 4,(  0,   0, 255)),\
                    "yellow":   ( 5,(255, 255,   0)),\
                    "brown":    ( 6,(188, 143, 143)),\
                    "grey":     ( 7,(220, 220, 220)),\
                    "violet":   ( 8,(148,   0, 211)),\
                    "cyan":     ( 9,(  0, 255, 255)),\
                    "magenta":  (10,(255,   0, 255)),\
                    "orange":   (11,(255, 165,   0)),\
                    "indigo":   (12,(114,  33, 188)),\
                    "maroon":   (13,(103,   7,  72)),\
                    "turquoise":(14,( 64, 224, 208)),\
                    "green4":   (15,(  0, 139,   0))}

    self._linestylesList = [ "none",\
                             "solid",\
                             "dot",\
                             "dash",\
                             "long-dash",\
                             "dash-dot",\
                             "long-dash-dot",\
                             "dash-dot-dot",\
                             "dash-dash-dot"
                           ]
    
    self._linestyles = { "none":          0,\
                         "solid":         1,\
                         "dot":           2,\
                         "dash":          3,\
                         "long-dash":     4,\
                         "dash-dot":      5,\
                         "long-dash-dot": 6,\
                         "dash-dot-dot":  7,\
                         "dash-dash-dot": 8
                       }

    self._typesettingList = [ ("switch to font x",               '\\\\f{x}'),\
                              ("switch to font number n",        '\\\\f{n}'),\
                              ("return to orginial font",        '\\\\f{}'),\
                              ("switch to color x",              '\\\\R{x}'),\
                              ("switch to color number n",       '\\\\R{n}'),\
                              ("switch to original color",       '\\\\R{}'),\
                              ("treat x as hex character codes", '\\\\#{x}'),\
                              ("apply transformation matrix",    '\\\\t{xx xy yx yy}'),\
                              ("reset transformation matrix",    '\\\\t{}'),\
                              ("zoom x times",                   '\\\\z{x}'),\
                              ("return to original zoom",        '\\\\z{}'),\
                              ("rotate by x degrees",            '\\\\r{x}'),\
                              ("slant by factor x",              '\\\\l{x}'),\
                              ("shift vertically by x",          '\\\\v{x}'),\
                              ("return to unshifted baseline",   '\\\\v{}'),\
                              ("shift baseline by x",            '\\\\V{x}'),\
                              ("reset baseline",                 '\\\\V{}'),\
                              ("horizontal shift by x",          '\\\\h{x}'),\
                              ("new line",                       '\\\\n'),\
                              ("begin underline",                '\\\\u'),\
                              ("stop underline",                 '\\\\U'),\
                              ("begin overline",                 '\\\\o'),\
                              ("stop overline",                  '\\\\O'),\
                              ("enable kerning",                 '\\\\Fk'),\
                              ("disable kerning",                '\\\\FK'),\
                              ("enable ligatures",               '\\\\Fl'),\
                              ("disable ligatures",              '\\\\FL'),\
                              ("mark current position as n",     '\\\\m{n}'),\
                              ("return to position n",           '\\\\M{n}'),\
                              ("LtoR substring direction",       '\\\\dl'),\
                              ("RtoL substring direction",       '\\\\dr'),\
                              ("LtoR text advancing",            '\\\\dL'),\
                              ("RtoL text advancing",            '\\\\dR'),\
                              ("switch to Symbol font",          '\\\\x'),\
                              ("increase size",                  '\\\\+'),\
                              ("decrease size",                  '\\\\='),\
                              ("begin subscript",                '\\\\s'),\
                              ("begin superscript",              '\\\\S'),\
                              ("absolute transformation matrix", '\\\\T{xx xy yx yy}'),\
                              ("absolute zoom x times",          '\\\\Z{x}'),\
                              ("make font oblique",              '\\\\q'),\
                              ("undo oblique",                   '\\\\Q'),\
                              ("return to normal style",         '\\\\N'),\
                              ("print \\",                       '\\\\')\
                            ]

    self._typesetting = { "switch to font x":               '\\\\f{x}',\
                          "switch to font number n":        '\\\\f{n}',\
                          "return to orginial font":        '\\\\f{}',\
                          "switch to color x":              '\\\\R{x}',\
                          "switch to color number n":       '\\\\R{n}',\
                          "switch to original color":       '\\\\R{}',\
                          "treat x as hex character codes": '\\\\#{x}',\
                          "apply transformation matrix":    '\\\\t{xx xy yx yy}',\
                          "reset transformation matrix":    '\\\\t{}',\
                          "zoom x times":                   '\\\\z{x}',\
                          "return to original zoom":        '\\\\z{}',\
                          "rotate by x degrees":            '\\\\r{x}',\
                          "slant by factor x":              '\\\\l{x}',\
                          "shift vertically by x":          '\\\\v{x}',\
                          "return to unshifted baseline":   '\\\\v{}',\
                          "shift baseline by x":            '\\\\V{x}',\
                          "reset baseline":                 '\\\\V{}',\
                          "horizontal shift by x":          '\\\\h{x}',\
                          "new line":                       '\\\\n',\
                          "begin underline":                '\\\\u',\
                          "stop underline":                 '\\\\U',\
                          "begin overline":                 '\\\\o',\
                          "stop overline":                  '\\\\O',\
                          "enable kerning":                 '\\\\Fk',\
                          "disable kerning":                '\\\\FK',\
                          "enable ligatures":               '\\\\Fl',\
                          "disable ligatures":              '\\\\FL',\
                          "mark current position as n":     '\\\\m{n}',\
                          "return to position n":           '\\\\M{n}',\
                          "LtoR substring direction":       '\\\\dl',\
                          "RtoL substring direction":       '\\\\dr',\
                          "LtoR text advancing":            '\\\\dL',\
                          "RtoL text advancing":            '\\\\dR',\
                          "switch to Symbol font":          '\\\\x',\
                          "increase size":                  '\\\\+',\
                          "decrease size":                  '\\\\=',\
                          "begin subscript":                '\\\\s',\
                          "begin superscript":              '\\\\S',\
                          "absolute transformation matrix": '\\\\T{xx xy yx yy}',\
                          "absolute zoom x times":          '\\\\Z{x}',\
                          "make font oblique":              '\\\\q',\
                          "undo oblique":                   '\\\\Q',\
                          "return to normal style":         '\\\\N',\
                          "print \\":                       '\\\\'\
                        }

    
    self._pageX = '800'
    self._pageY = '600'

    self._minX = ''
    self._maxX = ''
    self._minY = ''
    self._maxY = ''

    self._tickMajorDeltaX = '1.0'
    self._tickMajorSizeX = '1.0'
    self._tickMajorColorX = str(self.getColorNum("black"))
    self._tickMajorWidthX = '1.0'
    self._tickMajorStyleX = '1'
    self._tickMajorGridX = 'off'
    self._tickMinorNumX = '1'
    self._tickMinorColorX = str(self.getColorNum("black"))
    self._tickMinorWidthX = '1.0'
    self._tickMinorStyleX = '1'
    self._tickMinorGridX ='off'
    self._tickMinorSizeX = '0.5'

    self._tickLabelOnX = 'on'
    self._tickLabelPrecX = '5'
    self._tickLabelAngleX = '0'
    self._tickLabelSkipX = '0'
    self._tickLabelSizeX = '1.0'
    self._tickLabelFontX = str(self.getFontNum("Times-Roman"))
    self._tickLabelColorX = str(self.getColorNum("black"))
    
    self._tickMajorDeltaY = '1.0'
    self._tickMajorSizeY = '1.0'
    self._tickMajorColorY = str(self.getColorNum("black"))
    self._tickMajorWidthY = '1.0'
    self._tickMajorStyleY = '1'
    self._tickMajorGridY = 'off'

    self._tickMinorNumY = '1'
    self._tickMinorColorY = str(self.getColorNum("black"))
    self._tickMinorWidthY = '1.0'
    self._tickMinorStyleY = '1'
    self._tickMinorGridY ='off'
    self._tickMinorSizeY = '0.5'
    
    self._tickLabelOnY = 'on'
    self._tickLabelPrecY = '5'
    self._tickLabelAngleY = '0'
    self._tickLabelSkipY = '0'
    self._tickLabelSizeY = '1.0'
    self._tickLabelFontY = str(self.getFontNum("Times-Roman"))
    self._tickLabelColorY = str(self.getColorNum("black"))

    self._defaultFont = str(self.getFontNum("Times-Roman"))
    self._defaultColor = str(self.getColorNum("black"))
    self._defaultLineWidth = '1.0'
    self._defaultLineStyle = '1'
    self._defaultPattern = '1'
    self._defaultCharSize = '1.0'
    self._defaultSymbolSize = '1.0'
    self._defaultSFormat = '%.8g'

    self._backColor = str(self.getColorNum("white"))

    self._title = ''
    self._titleSize = '1.5'
    self._titleFont = str(self.getFontNum("Times-Bold"))
    self._titleColor = str(self.getColorNum("black"))
    self._subtitle = ''
    self._subtitleSize = '1.0'
    self._subtitleFont = str(self.getFontNum("Times-Roman"))
    self._subtitleColor = str(self.getColorNum("black"))

    self._labelX = ''
    self._labelSizeX = '1.0'
    self._labelFontX = str(self.getFontNum("Times-Roman"))
    self._labelColorX = str(self.getColorNum("black"))
    self._barColorX = str(self.getColorNum("black"))
    self._barWidthX = '1.0'
    self._barStyleX = '1'

    self._labelY = ''
    self._labelSizeY = '1.0'
    self._labelFontY = str(self.getFontNum("Times-Roman"))
    self._labelColorY = str(self.getColorNum("black"))
    self._barColorY = str(self.getColorNum("black"))
    self._barWidthY = '1.0'
    self._barStyleY = '1'

    self._numSetsFormat = 0
    self._lineWidth = ['1']
    self._lineColor = ['1']
    self._lineStyle = ['1']
    self._lineType = ['1']
    self._linePattern = ['1']

    self._timestamp = self._weekday[t[6]]+" "+\
                      self._month[t[1]]  +" "+\
                      str(t[2])          +" "+\
                      str(t[3])+":"+str(t[4])+":"+str(t[5])+" "+\
                      str(t[0])
    return
  # end __init__

  def page(self,x,y):
    self._pageX = x
    self._pageY = y
  #end setPage

  def title(self,a):
    self._title = a
  def titleFont(self,f):
    self._titleFont = str(self._fonts[f])
  def titleSize(self,s):
    self._titleSize = str(s)

  def subtitle(self,a):
    self._subtitle = a
  def subtitleFont(self,f):
    self._subtitleFont = str(self._fonts[f])
  def subtitleSize(self,s):
    self._subtitleSize = str(s)

  def minX(self,x):
    self._minX = str(x)
  def maxX(self,x):
    self._maxX = str(x)
  def deltaX(self,x):
    self._tickMajorDeltaX = str(x)
  def labelX(self,x):
    self._labelX = str(x)
  def labelFontX(self,x):
    self._labelFontX = str(self._fonts[x])
  def labelSizeX(self,x):
    self._labelSizeX = str(x)

  def minY(self,x):
    self._minY = str(x)
  def maxY(self,x):
    self._maxY = str(x)
  def deltaY(self,x):
    self._tickMajorDeltaY = str(x)
  def labelY(self,x):
    self._labelY = str(x)
  def labelFontY(self,x):
    self._labelFontY = str(self._fonts[x])
  def labelSizeY(self,x):
    self._labelSizeY = str(x)

  def thickerLines(self,sets=[]):
    if len(sets) == 0:
      for i in range(self._numSetsFormat):
        self._lineWidth[i] = str(int(self._lineWidth[i])+1)
      return

    for i in range(min(len(sets),self._numSetsFormat)):
      self._lineWidth[sets[i]] = str(int(self._lineWidth[i])+1)
  
  def thickLines(self,w=2,widths=[],sets=[]):
    """
    thickLines([w,[sets,[widths]]]):

      w      = default thickness to apply to formatted
               sets.  The default is overridden if a
               list of sets and corresponding widths are
               given.
      widths = widths to apply to list of sets..
               If empty the default width is used.
               If length is less than length of sets
               then the default width is applied to
               additional sets.
      sets   = list of sets to apply thickLines to.
               If empty or ommitted apply the default
               width to all formatted sets as determined
               by numSetsFormat.
    """

    if len(sets) == 0 and len(widths) == 0 and w == 0:
      return
    
    if len(sets) == 0 and len(widths) == 0:
      for i in range(self._numSetsFormat):
        self._lineWidth[i] = str(w)
      return
    
    if len(sets) == 0:
      for i in range(min(len(widths),self._numSetsFormat)):
        self._lineWidth[i] = str(widths[i])
      return
    
    if len(widths) == 0:
      for i in range(min(len(sets),self._numSetsFormat)):
        self._lineWidth[sets[i]] = str(w)
      return

    if w == 0:
      for i in range(min(len(widths),len(sets),self._numSetsFormat)):
        self._lineWidth[sets[i]] = str(widths[i])
      return
    
    for i in range(self._numSetsFormat):
      self._lineWidth[i] = str(w)
    for i in range(min(len(widths),len(sets),self._numSetsFormat)):
      self._lineWidth[sets[i]] = str(widths[i])

    return

  def thickAxes(self,n=2):
    self._barWidthX = str(n)
    self._barWidthY = str(n)
    self._tickMajorWidthX = str(n)
    self._tickMajorWidthY = str(n)
    self._tickMinorWidthX = str(n)
    self._tickMinorWidthY = str(n)
    return
  
  def numSetsFormat(self,num):
    """
    numSetsFormat(num):

      num = number of sets to receive line formatting
      
    Sets the number of data sets to have line formatting and
    also initialize the line formatting for the given number
    of sets to defaults.  The defaults can be overwritten by
    explicit calls to the following methods:

       setWidths()
       setColors()
       setStyles()

    There may be more data sets than those whose lines are formatted.
    The additional sets will receive grace default formatting. Those
    sets receiving special formatting must be the first num sets in
    the list of [x,y] pairs of lists.
    """
    self._numSetsFormat = num
    self._lineWidth = []
    self._lineColor = []
    self._lineType  = []
    self._lineStyle = []
    self._linePattern = []
    nc = len(self._colorsList)
    ns = len(self._linestylesList)
    for i in range(num):
      self._lineWidth.append(self._defaultLineWidth)
      self._lineColor.append(str((i%nc) + 1))
      self._lineStyle.append(str(((i/ns)%ns) + 1))
      self._lineType.append('1')
      self._linePattern.append('1')
    return

  
  def setColors(self,colors=[],sets=[]):
    """
    setColors(colors,[sets]):

      colors = list of colors to be applied
               to formatted sets
      sets   = optional list of sets to which
               to apply colors

    If the length of the sets list is 0 or if
    no list of sets is given, then the colors
    are applied sequentially to the formatted
    sets up to the number of sets receiving
    formatting as determined by previously calling
    numSetsFormat.  If the length of sets is
    not 0, then the colors are applied to the
    sets in the list.  If the length of sets
    is larger than the length of colors then
    the last color in the list is used for
    the remaining sets.
    """
    
    if len(colors) == 0:
      return
    
    if len(sets) == 0:
      for i in range(min(len(colors),self._numSetsFormat)):
        self._lineColor[i] = str(self.getColorNum(colors[i]))
      return
    
    for i in range(min(len(sets),self._numSetsFormat)):
      self._lineColor[sets[i]] = str(self.getColorNum(colors[min(i,len(colors)-1)]))
    
    return

  def setStyles(self,styles=[],sets=[]):
    """
    """
    
    if len(styles) == 0:
      return

    if len(sets) == 0:
      for i in range(min(len(styles),self._numSetsFormat)):
        self._lineStyle[i] = str(self.getLineStyleNum(styles[i]))
      return
    
    for i in range(min(len(sets),self._numSetsFormat)):
      self._lineStyle[sets[i]] = str(self.getLineStyleNum(styles[min(i,len(styles)-1)]))

    return
  
  def toScreen(self,dat):
    import os
    if len(self._filename) == 0:
      self.toGraceFile(dat)
      os.system("xmgrace "+self._defaultFile)
      os.system("rm "+self._defaultFile)
    else:
      os.system("xmgrace "+self._filename)
      
    return
  
  def toPS(self,dat,psfile):
    import os
    if len(psfile) == 0:
      print("No psfile name given")
      return
    self._psfile = psfile
    
    if len(self._filename) == 0:
      self.toGraceFile(dat)
      os.system("gracebat "+self._defaultFile+" -printfile "+psfile)
      os.system("rm "+self._defaultFile)
    else:
      os.system("gracebat "+self._filename+" -printfile "+psfile)

    return
  
  def toEPS(self,dat,epsfile):
    import os
    if len(epsfile) == 0:
      print("No epsfile name given")
      return

    psf  = epsfile.strip('.eps')+'.ps'

    if len(self._filename) == 0:
      self.toGraceFile(dat)
      if len(self._psfile) == 0:
        os.system("gracebat "+self._defaultFile+" -printfile "+psf)
      os.system("ps2epsi "+psf+" "+epsfile)
      os.system("rm "+self._defaultFile)
      if len(self._psfile) == 0:
        os.system("rm "+psf)
    else:
      if self._psfile != psf:
       os.system("gracebat "+self._filename+" -printfile "+psf)
      os.system("ps2epsi "+psf+" "+epsfile)
      if len(self._psfile) == 0:
        os.system("rm "+psf)

    return

  def toPDF(self,dat,pdffile):
    import os
    if len(pdffile) == 0:
      print("No pdffile name given")
      return

    psf  = pdffile.strip('.pdf')+'.ps'

    if len(self._filename) == 0:
      self.toGraceFile(dat)
      if self._psfile != psf:
        os.system("gracebat "+self._defaultFile+" -printfile "+psf)
      os.system("ps2pdf "+psf+" "+pdffile)
      os.system("rm "+self._defaultFile)
      if self._psfile != psf:
        os.system("rm "+psf)
    else:
      if self._psfile != psf:
        os.system("gracebat "+self._filename+" -printfile "+psf)
      os.system("ps2pdf "+psf+" "+pdffile)
      if self._psfile != psf:
        os.system("rm "+psf)
      
    return

  def toGraceFile(self,dat,filename=''):
    """
    toGraceFile(dat,[filename])

      dat = [[x1,y1],[x2,y2],...]
      xi = list of x values for ith set
      yi = list of y values for ith set

      filename = name of grace file to write to.

    If no filename is given then a temporary name is constructed
    based on the time of file creation and has a random number
    encoded into the filename to ensure uniqueness.
    """

# If no filename is given use the one created
# when the GracePlot object was instantiated

    if len(filename) != 0:
      self._filename = filename
      fname = filename
    else:
      fname = self._defaultFile

    FIL = open(fname,'w')
    FIL.write(self.graceHdr())

    for iset in range(len(dat)):
      FIL.write('@target G0.S'+str(iset)+'\n')
      FIL.write('@type xy\n')
      for ix in range(len(dat[iset][0])):
        FIL.write(str(dat[iset][0][ix])+' '+str(dat[iset][1][ix])+'\n')

      FIL.write('&\n')

    FIL.close()
    return

  def graceHdr(self):
    hdr = '# Grace project file\n'
    hdr = hdr + '#\n'
    if len(self._minX) != 0:
      hdr = hdr + '@version 50110\n'

    if len(self._pageX) != 0 and len(self._pageY) != 0:
      hdr = hdr + '@page size '+\
                  self._pageX+', '+\
                  self._pageY+'\n'

    hdr = hdr + '@page scroll 5%\n'
    hdr = hdr + '@page inout 5%\n'
    hdr = hdr + '@link page off\n'
    hdr = hdr + '@map font 0 to "' +self.getFont(0) +'", "'+self.getFont(0) +'"\n'
    hdr = hdr + '@map font 1 to "' +self.getFont(1) +'", "'+self.getFont(1) +'"\n'
    hdr = hdr + '@map font 2 to "' +self.getFont(2) +'", "'+self.getFont(2) +'"\n'
    hdr = hdr + '@map font 3 to "' +self.getFont(3) +'", "'+self.getFont(3) +'"\n'
    hdr = hdr + '@map font 4 to "' +self.getFont(4) +'", "'+self.getFont(4) +'"\n'
    hdr = hdr + '@map font 5 to "' +self.getFont(5) +'", "'+self.getFont(5) +'"\n'
    hdr = hdr + '@map font 6 to "' +self.getFont(6) +'", "'+self.getFont(6) +'"\n'
    hdr = hdr + '@map font 7 to "' +self.getFont(7) +'", "'+self.getFont(7) +'"\n'
    hdr = hdr + '@map font 8 to "' +self.getFont(8) +'", "'+self.getFont(8) +'"\n'
    hdr = hdr + '@map font 9 to "' +self.getFont(9) +'", "'+self.getFont(9) +'"\n'
    hdr = hdr + '@map font 10 to "'+self.getFont(10)+'", "'+self.getFont(10)+'"\n'
    hdr = hdr + '@map font 11 to "'+self.getFont(11)+'", "'+self.getFont(11)+'"\n'
    hdr = hdr + '@map font 12 to "'+self.getFont(12)+'", "'+self.getFont(12)+'"\n'
    hdr = hdr + '@map font 13 to "'+self.getFont(13)+'", "'+self.getFont(13)+'"\n'
    hdr = hdr + '@map color 0 to ' +str(self.getRGB(0)) +', "'+self.getColor(0) +'"\n'
    hdr = hdr + '@map color 1 to ' +str(self.getRGB(1)) +', "'+self.getColor(1) +'"\n'
    hdr = hdr + '@map color 2 to ' +str(self.getRGB(2)) +', "'+self.getColor(2) +'"\n'
    hdr = hdr + '@map color 3 to ' +str(self.getRGB(3)) +', "'+self.getColor(3) +'"\n'
    hdr = hdr + '@map color 4 to ' +str(self.getRGB(4)) +', "'+self.getColor(4) +'"\n'
    hdr = hdr + '@map color 5 to ' +str(self.getRGB(5)) +', "'+self.getColor(5) +'"\n'
    hdr = hdr + '@map color 6 to ' +str(self.getRGB(6)) +', "'+self.getColor(6) +'"\n'
    hdr = hdr + '@map color 7 to ' +str(self.getRGB(7)) +', "'+self.getColor(7) +'"\n'
    hdr = hdr + '@map color 8 to ' +str(self.getRGB(8)) +', "'+self.getColor(8) +'"\n'
    hdr = hdr + '@map color 9 to ' +str(self.getRGB(9)) +', "'+self.getColor(9) +'"\n'
    hdr = hdr + '@map color 10 to '+str(self.getRGB(10))+', "'+self.getColor(10)+'"\n'
    hdr = hdr + '@map color 11 to '+str(self.getRGB(11))+', "'+self.getColor(11)+'"\n'
    hdr = hdr + '@map color 12 to '+str(self.getRGB(12))+', "'+self.getColor(12)+'"\n'
    hdr = hdr + '@map color 13 to '+str(self.getRGB(13))+', "'+self.getColor(13)+'"\n'
    hdr = hdr + '@map color 14 to '+str(self.getRGB(14))+', "'+self.getColor(14)+'"\n'
    hdr = hdr + '@map color 15 to '+str(self.getRGB(15))+', "'+self.getColor(15)+'"\n'
    hdr = hdr + '@reference date 0\n'
    hdr = hdr + '@date wrap off\n'
    hdr = hdr + '@date wrap year 1950\n'
    hdr = hdr + '@default linewidth '+self._defaultLineWidth+'\n'
    hdr = hdr + '@default linestyle '+self._defaultLineStyle+'\n'
    hdr = hdr + '@default color '+self._defaultColor+'\n'
    hdr = hdr + '@default pattern '+self._defaultPattern+'\n'
    hdr = hdr + '@default font '+self._defaultFont+'\n'
    hdr = hdr + '@default char size '+self._defaultCharSize+'\n'
    hdr = hdr + '@default symbol size '+self._defaultSymbolSize+'\n'
    hdr = hdr + '@default sformat "'+self._defaultSFormat+'"\n'
    hdr = hdr + '@background color '+self._backColor+'\n'
    hdr = hdr + '@page background fill on\n'
    hdr = hdr + '@timestamp off\n'
    hdr = hdr + '@timestamp 0.03, 0.03\n'
    hdr = hdr + '@timestamp color 1\n'
    hdr = hdr + '@timestamp rot 0\n'
    hdr = hdr + '@timestamp font 0\n'
    hdr = hdr + '@timestamp char size 1.000000\n'
    hdr = hdr + '@timestamp def "'+self._timestamp+'"\n'
    hdr = hdr + '@r0 off\n'
    hdr = hdr + '@link r0 to g0\n'
    hdr = hdr + '@r0 type above\n'
    hdr = hdr + '@r0 linestyle 1\n'
    hdr = hdr + '@r0 linewidth 1.0\n'
    hdr = hdr + '@r0 color 1\n'
    hdr = hdr + '@r0 line 0, 0, 0, 0\n'
    hdr = hdr + '@r1 off\n'
    hdr = hdr + '@link r1 to g0\n'
    hdr = hdr + '@r1 type above\n'
    hdr = hdr + '@r1 linestyle 1\n'
    hdr = hdr + '@r1 linewidth 1.0\n'
    hdr = hdr + '@r1 color 1\n'
    hdr = hdr + '@r1 line 0, 0, 0, 0\n'
    hdr = hdr + '@r2 off\n'
    hdr = hdr + '@link r2 to g0\n'
    hdr = hdr + '@r2 type above\n'
    hdr = hdr + '@r2 linestyle 1\n'
    hdr = hdr + '@r2 linewidth 1.0\n'
    hdr = hdr + '@r2 color 1\n'
    hdr = hdr + '@r2 line 0, 0, 0, 0\n'
    hdr = hdr + '@r3 off\n'
    hdr = hdr + '@link r3 to g0\n'
    hdr = hdr + '@r3 type above\n'
    hdr = hdr + '@r3 linestyle 1\n'
    hdr = hdr + '@r3 linewidth 1.0\n'
    hdr = hdr + '@r3 color 1\n'
    hdr = hdr + '@r3 line 0, 0, 0, 0\n'
    hdr = hdr + '@r4 off\n'
    hdr = hdr + '@link r4 to g0\n'
    hdr = hdr + '@r4 type above\n'
    hdr = hdr + '@r4 linestyle 1\n'
    hdr = hdr + '@r4 linewidth 1.0\n'
    hdr = hdr + '@r4 color 1\n'
    hdr = hdr + '@r4 line 0, 0, 0, 0\n'
    hdr = hdr + '@g0 on\n'
    hdr = hdr + '@g0 hidden false\n'
    hdr = hdr + '@g0 type XY\n'
    hdr = hdr + '@g0 stacked false\n'
    hdr = hdr + '@g0 bar hgap 0.000000\n'
    hdr = hdr + '@g0 fixedpoint off\n'
    hdr = hdr + '@g0 fixedpoint type 0\n'
    hdr = hdr + '@g0 fixedpoint xy 0.000000, 0.000000\n'
    hdr = hdr + '@g0 fixedpoint format general general\n'
    hdr = hdr + '@g0 fixedpoint prec 6, 6\n'
    hdr = hdr + '@with g0\n'
    if len(self._minX) != 0:
      hdr = hdr + '@    world xmin '+self._minX+'\n'
      
    if len(self._maxX) != 0:
      hdr = hdr + '@    world xmax '+self._maxX+'\n'

    if len(self._minY) != 0:
      hdr = hdr + '@    world ymin '+self._minY+'\n'
    if len(self._maxY) != 0:
      hdr = hdr + '@    world ymax '+self._maxY+'\n'

    hdr = hdr + '@    stack world 0, 0, 0, 0\n'
    hdr = hdr + '@    znorm 1\n'
    hdr = hdr + '@    view xmin 0.150000\n'
    hdr = hdr + '@    view xmax 1.150000\n'
    hdr = hdr + '@    view ymin 0.150000\n'
    hdr = hdr + '@    view ymax 0.850000\n'
    hdr = hdr + '@    title "'+self._title+'"\n'
    hdr = hdr + '@    title font '+self._titleFont+'\n'
    hdr = hdr + '@    title size '+self._titleSize+'\n'
    hdr = hdr + '@    title color '+self._titleColor+'\n'
    hdr = hdr + '@    subtitle "'+self._subtitle+'"\n'
    hdr = hdr + '@    subtitle font '+self._subtitleFont+'\n'
    hdr = hdr + '@    subtitle size '+self._subtitleSize+'\n'
    hdr = hdr + '@    subtitle color '+self._subtitleColor+'\n'
    hdr = hdr + '@    xaxes scale Normal\n'
    hdr = hdr + '@    yaxes scale Normal\n'
    hdr = hdr + '@    xaxes invert off\n'
    hdr = hdr + '@    yaxes invert off\n'
    hdr = hdr + '@    xaxis  on\n'
    hdr = hdr + '@    xaxis  type zero false\n'
    hdr = hdr + '@    xaxis  offset 0.000000 , 0.000000\n'
    hdr = hdr + '@    xaxis  bar on\n'
    hdr = hdr + '@    xaxis  bar color '+self._barColorX+'\n'
    hdr = hdr + '@    xaxis  bar linestyle '+self._barStyleX+'\n'
    hdr = hdr + '@    xaxis  bar linewidth '+self._barWidthX+'\n'
    hdr = hdr + '@    xaxis  label "'+self._labelX+'"\n'
    hdr = hdr + '@    xaxis  label layout para\n'
    hdr = hdr + '@    xaxis  label place auto\n'
    hdr = hdr + '@    xaxis  label char size '+self._labelSizeX+'\n'
    hdr = hdr + '@    xaxis  label font '+self._labelFontX+'\n'
    hdr = hdr + '@    xaxis  label color '+self._labelColorX+'\n'
    hdr = hdr + '@    xaxis  label place normal\n'
    hdr = hdr + '@    xaxis  tick on\n'
    hdr = hdr + '@    xaxis  tick major '+self._tickMajorDeltaX+'\n'
    hdr = hdr + '@    xaxis  tick minor ticks '+self._tickMinorNumX+'\n'
    hdr = hdr + '@    xaxis  tick default 6\n'
    hdr = hdr + '@    xaxis  tick place rounded true\n'
    hdr = hdr + '@    xaxis  tick in\n'
    hdr = hdr + '@    xaxis  tick major size '+self._tickMajorSizeX+'\n'
    hdr = hdr + '@    xaxis  tick major color '+self._tickMajorColorX+'\n'
    hdr = hdr + '@    xaxis  tick major linewidth '+self._tickMajorWidthX+'\n'
    hdr = hdr + '@    xaxis  tick major linestyle '+self._tickMajorStyleX+'\n'
    hdr = hdr + '@    xaxis  tick major grid '+self._tickMajorGridX+'\n'
    hdr = hdr + '@    xaxis  tick minor color '+self._tickMinorColorX+'\n'
    hdr = hdr + '@    xaxis  tick minor linewidth '+self._tickMinorWidthX+'\n'
    hdr = hdr + '@    xaxis  tick minor linestyle '+self._tickMinorStyleX+'\n'
    hdr = hdr + '@    xaxis  tick minor grid '+self._tickMinorGridX+'\n'
    hdr = hdr + '@    xaxis  tick minor size '+self._tickMinorSizeX+'\n'
    hdr = hdr + '@    xaxis  ticklabel '+self._tickLabelOnX+'\n'
    hdr = hdr + '@    xaxis  ticklabel format general\n'
    hdr = hdr + '@    xaxis  ticklabel prec '+self._tickLabelPrecX+'\n'
    hdr = hdr + '@    xaxis  ticklabel formula ""\n'
    hdr = hdr + '@    xaxis  ticklabel append ""\n'
    hdr = hdr + '@    xaxis  ticklabel prepend ""\n'
    hdr = hdr + '@    xaxis  ticklabel angle '+self._tickLabelAngleX+'\n'
    hdr = hdr + '@    xaxis  ticklabel skip '+self._tickLabelSkipX+'\n'
    hdr = hdr + '@    xaxis  ticklabel stagger 0\n'
    hdr = hdr + '@    xaxis  ticklabel place normal\n'
    hdr = hdr + '@    xaxis  ticklabel offset auto\n'
    hdr = hdr + '@    xaxis  ticklabel offset 0.000000 , 0.010000\n'
    hdr = hdr + '@    xaxis  ticklabel start type auto\n'
    hdr = hdr + '@    xaxis  ticklabel start 0.000000\n'
    hdr = hdr + '@    xaxis  ticklabel stop type auto\n'
    hdr = hdr + '@    xaxis  ticklabel stop 0.000000\n'
    hdr = hdr + '@    xaxis  ticklabel char size '+self._tickLabelSizeX+'\n'
    hdr = hdr + '@    xaxis  ticklabel font '+self._tickLabelFontX+'\n'
    hdr = hdr + '@    xaxis  ticklabel color '+self._tickLabelColorX+'\n'
    hdr = hdr + '@    xaxis  tick place both\n'
    hdr = hdr + '@    xaxis  tick spec type none\n'
    hdr = hdr + '@    yaxis  on\n'
    hdr = hdr + '@    yaxis  type zero false\n'
    hdr = hdr + '@    yaxis  offset 0.000000 , 0.000000\n'
    hdr = hdr + '@    yaxis  bar on\n'
    hdr = hdr + '@    yaxis  bar color '+self._barColorY+'\n'
    hdr = hdr + '@    yaxis  bar linestyle '+self._barStyleY+'\n'
    hdr = hdr + '@    yaxis  bar linewidth '+self._barWidthY+'\n'
    hdr = hdr + '@    yaxis  label "'+self._labelY+'"\n'
    hdr = hdr + '@    yaxis  label layout para\n'
    hdr = hdr + '@    yaxis  label place auto\n'
    hdr = hdr + '@    yaxis  label char size '+self._labelSizeY+'\n'
    hdr = hdr + '@    yaxis  label font '+self._labelFontY+'\n'
    hdr = hdr + '@    yaxis  label color '+self._labelColorY+'\n'
    hdr = hdr + '@    yaxis  label place normal\n'
    hdr = hdr + '@    yaxis  tick on\n'
    hdr = hdr + '@    yaxis  tick major '+self._tickMajorDeltaY+'\n'
    hdr = hdr + '@    yaxis  tick minor ticks '+self._tickMinorNumY+'\n'
    hdr = hdr + '@    yaxis  tick default 6\n'
    hdr = hdr + '@    yaxis  tick place rounded true\n'
    hdr = hdr + '@    yaxis  tick in\n'
    hdr = hdr + '@    yaxis  tick major size '+self._tickMajorSizeY+'\n'
    hdr = hdr + '@    yaxis  tick major color '+self._tickMajorColorY+'\n'
    hdr = hdr + '@    yaxis  tick major linewidth '+self._tickMajorWidthY+'\n'
    hdr = hdr + '@    yaxis  tick major linestyle '+self._tickMajorStyleY+'\n'
    hdr = hdr + '@    yaxis  tick major grid '+self._tickMajorGridY+'\n'
    hdr = hdr + '@    yaxis  tick minor color '+self._tickMinorColorY+'\n'
    hdr = hdr + '@    yaxis  tick minor linewidth '+self._tickMinorWidthY+'\n'
    hdr = hdr + '@    yaxis  tick minor linestyle '+self._tickMinorStyleY+'\n'
    hdr = hdr + '@    yaxis  tick minor grid '+self._tickMinorGridY+'\n'
    hdr = hdr + '@    yaxis  tick minor size '+self._tickMinorSizeY+'\n'
    hdr = hdr + '@    yaxis  ticklabel '+self._tickLabelOnY+'\n'
    hdr = hdr + '@    yaxis  ticklabel format general\n'
    hdr = hdr + '@    yaxis  ticklabel prec '+self._tickLabelPrecY+'\n'
    hdr = hdr + '@    yaxis  ticklabel formula ""\n'
    hdr = hdr + '@    yaxis  ticklabel append ""\n'
    hdr = hdr + '@    yaxis  ticklabel prepend ""\n'
    hdr = hdr + '@    yaxis  ticklabel angle '+self._tickLabelAngleY+'\n'
    hdr = hdr + '@    yaxis  ticklabel skip '+self._tickLabelSkipY+'\n'
    hdr = hdr + '@    yaxis  ticklabel stagger 0\n'
    hdr = hdr + '@    yaxis  ticklabel place normal\n'
    hdr = hdr + '@    yaxis  ticklabel offset auto\n'
    hdr = hdr + '@    yaxis  ticklabel offset 0.000000 , 0.010000\n'
    hdr = hdr + '@    yaxis  ticklabel start type auto\n'
    hdr = hdr + '@    yaxis  ticklabel start 0.000000\n'
    hdr = hdr + '@    yaxis  ticklabel stop type auto\n'
    hdr = hdr + '@    yaxis  ticklabel stop 0.000000\n'
    hdr = hdr + '@    yaxis  ticklabel char size '+self._tickLabelSizeY+'\n'
    hdr = hdr + '@    yaxis  ticklabel font '+self._tickLabelFontY+'\n'
    hdr = hdr + '@    yaxis  ticklabel color '+self._tickLabelColorY+'\n'
    hdr = hdr + '@    yaxis  tick place both\n'
    hdr = hdr + '@    yaxis  tick spec type none\n'
    hdr = hdr + '@    altxaxis  off\n'
    hdr = hdr + '@    altyaxis  off\n'
    hdr = hdr + '@    legend on\n'
    hdr = hdr + '@    legend loctype view\n'
    hdr = hdr + '@    legend 0.85, 0.8\n'
    hdr = hdr + '@    legend box color '+str(self.getColorNum("black"))+'\n'
    hdr = hdr + '@    legend box pattern 1\n'
    hdr = hdr + '@    legend box linewidth 1.0\n'
    hdr = hdr + '@    legend box linestyle 1\n'
    hdr = hdr + '@    legend box fill color '+str(self.getColorNum("white"))+'\n'
    hdr = hdr + '@    legend box fill pattern 1\n'
    hdr = hdr + '@    legend font '+str(self.getFontNum("Times-Roman"))+'\n'
    hdr = hdr + '@    legend char size 1.000000\n'
    hdr = hdr + '@    legend color '+str(self.getColorNum("black"))+'\n'
    hdr = hdr + '@    legend length 4\n'
    hdr = hdr + '@    legend vgap 1\n'
    hdr = hdr + '@    legend hgap 1\n'
    hdr = hdr + '@    legend invert false\n'
    hdr = hdr + '@    frame type 0\n'
    hdr = hdr + '@    frame linestyle 1\n'
    hdr = hdr + '@    frame linewidth 1.0\n'
    hdr = hdr + '@    frame color '+str(self.getColorNum("black"))+'\n'
    hdr = hdr + '@    frame pattern 1\n'
    hdr = hdr + '@    frame background color '+str(self.getColorNum("white"))+'\n'
    hdr = hdr + '@    frame background pattern 0\n'
    for i in range(self._numSetsFormat):
      hdr = hdr + '@    s'+str(i)+' hidden false\n'
      hdr = hdr + '@    s'+str(i)+' type xy\n'
      hdr = hdr + '@    s'+str(i)+' symbol 0\n'
      hdr = hdr + '@    s'+str(i)+' symbol size 1.000000\n'
      hdr = hdr + '@    s'+str(i)+' symbol color '+str(self.getColorNum("black"))+'\n'
      hdr = hdr + '@    s'+str(i)+' symbol pattern 1\n'
      hdr = hdr + '@    s'+str(i)+' symbol fill color '+str(self.getColorNum("black"))+'\n'
      hdr = hdr + '@    s'+str(i)+' symbol fill pattern 0\n'
      hdr = hdr + '@    s'+str(i)+' symbol linewidth 1.0\n'
      hdr = hdr + '@    s'+str(i)+' symbol linestyle 1\n'
      hdr = hdr + '@    s'+str(i)+' symbol char 65\n'
      hdr = hdr + '@    s'+str(i)+' symbol char font '+str(self.getFontNum("Times-Roman"))+'\n'
      hdr = hdr + '@    s'+str(i)+' symbol skip 0\n'
      hdr = hdr + '@    s'+str(i)+' line type '+self._lineType[i]+'\n'
      hdr = hdr + '@    s'+str(i)+' line linestyle '+self._lineStyle[i]+'\n'
      hdr = hdr + '@    s'+str(i)+' line linewidth '+self._lineWidth[i]+'\n'
      hdr = hdr + '@    s'+str(i)+' line color '+self._lineColor[i]+'\n'
      hdr = hdr + '@    s'+str(i)+' line pattern '+self._linePattern[i]+'\n'
      hdr = hdr + '@    s'+str(i)+' baseline type 0\n'
      hdr = hdr + '@    s'+str(i)+' baseline off\n'
      hdr = hdr + '@    s'+str(i)+' dropline off\n'
      hdr = hdr + '@    s'+str(i)+' fill type 0\n'
      hdr = hdr + '@    s'+str(i)+' fill rule 0\n'
      hdr = hdr + '@    s'+str(i)+' fill color '+str(self.getColorNum("black"))+'\n'
      hdr = hdr + '@    s'+str(i)+' fill pattern 1\n'
      hdr = hdr + '@    s'+str(i)+' avalue off\n'
      hdr = hdr + '@    s'+str(i)+' avalue type 2\n'
      hdr = hdr + '@    s'+str(i)+' avalue char size 1.000000\n'
      hdr = hdr + '@    s'+str(i)+' avalue font '+str(self.getFontNum("Times-Roman"))+'\n'
      hdr = hdr + '@    s'+str(i)+' avalue color '+str(self.getColorNum("black"))+'\n'
      hdr = hdr + '@    s'+str(i)+' avalue rot 0\n'
      hdr = hdr + '@    s'+str(i)+' avalue format general\n'
      hdr = hdr + '@    s'+str(i)+' avalue prec 3\n'
      hdr = hdr + '@    s'+str(i)+' avalue prepend ""\n'
      hdr = hdr + '@    s'+str(i)+' avalue append ""\n'
      hdr = hdr + '@    s'+str(i)+' avalue offset 0.000000 , 0.000000\n'
      hdr = hdr + '@    s'+str(i)+' errorbar on\n'
      hdr = hdr + '@    s'+str(i)+' errorbar place both\n'
      hdr = hdr + '@    s'+str(i)+' errorbar color '+str(self.getColorNum("black"))+'\n'
      hdr = hdr + '@    s'+str(i)+' errorbar pattern 1\n'
      hdr = hdr + '@    s'+str(i)+' errorbar size 1.000000\n'
      hdr = hdr + '@    s'+str(i)+' errorbar linewidth 1.0\n'
      hdr = hdr + '@    s'+str(i)+' errorbar linestyle 1\n'
      hdr = hdr + '@    s'+str(i)+' errorbar riser linewidth 1.0\n'
      hdr = hdr + '@    s'+str(i)+' errorbar riser linestyle 1\n'
      hdr = hdr + '@    s'+str(i)+' errorbar riser clip off\n'
      hdr = hdr + '@    s'+str(i)+' errorbar riser clip length 0.100000\n'
      hdr = hdr + '@    s'+str(i)+' comment ""\n'
      hdr = hdr + '@    s'+str(i)+' legend  ""\n'
    return hdr

  def getFontTable(self):
    print(" Index      Font")
    print("===============================")
    for i in range(len(self._fonts)):
      print("  %2d     %s" % (i,self._fontsList[i]))
  def getFont(self,num):
    """
    Get the font name associated with an index.
    """
    if num < len(self._fonts):
      return self._fontsList[num]
    else:
      return 0
  def getFontNum(self,font):
    return self._fonts[font]

  def getColorTable(self):
    print(" Index   Color          RGB")
    print("===================================")
    for i in range(len(self._colors)):
      fmt = "  %2d     %s"
      for j in range(10-len(self._colorsList[i])):
        fmt = fmt + " "
      fmt = fmt + " %s"
      print(fmt % (i,self._colorsList[i],str(self._colors[self._colorsList[i]][1])))
  def getColor(self,num):
    return self._colorsList[num]
  def getColorNum(self,col):
    return self._colors[col][0]
  def getRGB(self,num):
    return self._colors[self.getColor(num)][1]
  def getColorRGB(self,col):
    return self._colors[col][1]

  def getLineStyleTable(self):
    print(" Index   LineStyle")
    print("=======================")
    for i in range(len(self._linestylesList)):
      print("  %2d      %s" % (i,self._linestylesList[i]))
  def getLineStyle(self,num):
    return self._linestylesList[num]
  def getLineStyleNum(self,sty):
    return self._linestyles[sty]
  
  def getTypesettingTable(self):
    print("Typesetting                        Control Sequence")
    print("===================================================")
    for i in range(len(self._typesettingList)):
      fmt = "%s"
      for j in range(35-len(self._typesettingList[i][0])):
        fmt = fmt + " "
      fmt = fmt + "%s"
      print(fmt % (self._typesettingList[i][0],self._typesettingList[i][1]))
  def getLineStyle(self,num):
    return self._linestylesList[num]

  def greek(self,s):
    """
    greek(s):

      s = a string
      
    Return Grace encoding to convert the input
    string to symbol font, i.e., greek font
    """
    return '\\x'+s+'\\f{}'  

  def big(self,s):
    """
    big(s):

      s = a string

    Return Grace encoding to increase the size of
    the input string by a factor of sqrt(sqrt(2))
    """
    return '\\+'+s+'\\-'

  def sub(self,s):
    return '\\s'+s+'\\N'

  def sup(self,s):
    return '\\S'+s+'\\N'

  def font(self,f,s):
    return '\\f{'+f+'}'+s+'\\f{}'

if __name__ == '__main__':
  
  from GracePlot import GracePlot

# First example, simple

  gp = GracePlot()
  xdat = [0,1,2,3]
  ydat = [0.,10.,20.,30.]
  gp.toScreen([[xdat,ydat]])


# Second example, more bells and whistles

  ax = range(100); ay = [0.]*100
  bx = range(100); by = [0.]*100
  import math
  for i in range(100):
    ax[i] *= 0.02; bx[i] *= 0.02
    ay[i] = math.exp(-4.*(ax[i]-1.)**2)*math.cos(ax[i]*2.*math.pi)
    by[i] = -1. + 2.*math.exp(-bx[i]**2)

  gg = GracePlot()  

# Nonsense titles and labels to illustrate options

  gg.title(gg.big('V-+')+gg.big(' | ')+'E'+gg.sub('V')+gg.big(' | ')+'Elevation Profile, '+
          gg.greek('f')+' = 90'+gg.sup('o'))
  gg.subtitle('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
  gg.subtitleFont('Symbol')

  gg.minX(0)
  gg.maxX(2)
  gg.deltaX(.2)
  gg.minY(-1)
  gg.maxY(1)
  gg.deltaY(0.2)
  gg.thickAxes()
  gg.numSetsFormat(2)
  gg.thickLines()
  gg.thickerLines([1])
  gg.setColors(['red','green4'])
  gg.setStyles(['solid','long-dash'])
  gg.labelX(gg.greek('abcdefghijklmnopqrstuvwxyz'))
  gg.labelY(gg.font('Helvetica','La')+'ble'+gg.sub(gg.big(gg.font('Times-Roman','Y'))))
  gg.labelFontY('Helvetica-Oblique')

# Options to display to screen and create different file formats

  a = [[ax,ay],[bx,by]]
  gg.toScreen(a)
  gg.toPS(a,"newtest.ps")
  gg.toEPS(a,"newtest.eps")
  gg.toPDF(a,"newtest.pdf")
  


# version
# $Id: GracePlot.py,v 1.1 2004/11/03 16:34:49 ericmg Exp $

# End of file
