==================
Python Terminology
==================

In Python terminology, a module is the basic block of code that can be imported by some other code. There are three main types of modules: packages, pure Python modules and external modules.

A **package** is a module that contains other modules. It is basically a directory in the filesystem, distinguished from other directories by the presence of a file *__init__.py*. That file might be empty but it can also execute initialization code for the package.

.. note:: Even empty, the file *__init.py__* is required for a directory to be treated as a containing package. Otherwise, it is considered as a normal directory in the filesystem. For example, the folder *bin* in the ISCE tree is not a Python package.

A **pure Python module** (or a pure module) is written in Python and contained in a single *.py* file. For example, the package *applications* in the ISCE tree contains only pure modules. Since ISCE is object-oriented, many of its pure modules implement classes of objects with their attributes and methods. Whenever possible, classes are also shown in the following diagrams.

Finally, an **external module** contains code written in languages other than Python (e.g. C/C++, Fortran, Java...) and is typically packed in a single dynamically loadable file (e.g. a shared object *.so* or a dynamic-link library *.dll*).

===============
Module Diagrams
===============

The diagrams shown in this section reflect the structure of ISCE, as of July 1, 2012.

The first figure (:ref:`overallstructure`) gives an overview of ISCE packages and modules. The root package of ISCE contains 3 packages: *applications*, *components* and *library*. The package *components* holds 5 subpackages that are detailed in the other figures:

* :ref:`isceobj`
* :ref:`iscesys`
* :ref:`stdproc`
* :ref:`contrib`
* :ref:`mroipac`

.. _overallstructure:

.. figure:: ISCE_structure01.png
   :align: center
   :width: 1000 px

   Overall structure of ISCE

.. _isceobj:

.. figure:: ISCE_structure02.png
   :align: center
   :width: 1000 px

   ISCEOBJ package

.. figure:: ISCE_structure03.png
   :align: center
   :width: 1000 px

   ISCEOBJ package (2/7)

.. figure:: ISCE_structure04.png
   :align: center
   :width: 1000 px

   ISCEOBJ package (3/7)

.. figure:: ISCE_structure05.png
   :align: center
   :width: 1000 px

   ISCEOBJ package (4/7)

.. figure:: ISCE_structure06.png
   :align: center
   :width: 1000 px

   ISCEOBJ package (5/7)

.. figure:: ISCE_structure07.png
   :align: center
   :width: 1000 px

   ISCEOBJ package (6/7)

.. figure:: ISCE_structure08.png
   :align: center
   :width: 1000 px

   ISCEOBJ package (7/7)

.. _iscesys:

.. figure:: ISCE_structure09.png
   :align: center
   :width: 1000 px

   ISCESYS package

.. figure:: ISCE_structure10.png
   :align: center
   :width: 1000 px

   ISCESYS package (2/3)

.. figure:: ISCE_structure11.png
   :align: center
   :width: 1000 px

   ISCESYS package (3/3)

.. _stdproc:

.. figure:: ISCE_structure12.png
   :align: center
   :width: 1000 px

   STDPROC package

.. figure:: ISCE_structure13.png
   :align: center
   :width: 1000 px

   STDPROC package (2/6)

.. figure:: ISCE_structure14.png
   :align: center
   :width: 1000 px

   STDPROC package (3/6)

.. figure:: ISCE_structure15.png
   :align: center
   :width: 1000 px

   STDPROC package (4/6)

.. figure:: ISCE_structure16.png
   :align: center
   :width: 1000 px

   STDPROC package (5/6)

.. figure:: ISCE_structure17.png
   :align: center
   :width: 1000 px

   STDPROC package (6/6)

.. _contrib:

.. figure:: ISCE_structure18.png
   :align: center
   :width: 1000 px

   CONTRIB package

.. _mroipac:

.. figure:: ISCE_structure19.png
   :align: center
   :width: 1000 px

   MROIPAC package

.. figure:: ISCE_structure20.png
   :align: center
   :width: 1000 px

   MROIPAC package (2/3)

.. figure:: ISCE_structure21.png
   :align: center
   :width: 1000 px

   MROIPAC package (3/3)
