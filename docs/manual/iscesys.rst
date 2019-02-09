===========
ISCE System
===========

Module Descriptions
*******************

MathModule
----------

.. py:module:: isceobj.Util.mathModule

.. py:class:: MathModule

   A class for some common mathematical functions.

   .. py:staticmethod:: multiplyMatrices(mat1,mat2)
   .. py:staticmethod:: invertMatrix(mat)
   .. py:staticmethod:: matrixTranspose(mat)
   .. py:staticmethod:: matrixVectorProduct(mat,vec)
   .. py:staticmethod:: crossProduct(v1,v2)
   .. py:staticmethod:: normalizeVector(v1)
   .. py:staticmethod:: norm(v1)
   .. py:staticmethod:: dotProduct(v1,v2)
   .. py:staticmethod:: median(list)
   .. py:staticmethod:: mean(list)
   .. py:staticmethod:: linearFit(x,y)
   .. py:staticmethod:: quadraticFit(x,y)

DateTimeUtil
------------

.. py:module:: iscesys.DateTimeUtil.DateTimeUtil

.. py:class:: DateTimeUtil

   A class containing some useful, and common, date manipulations.

   .. py:staticmethod:: timeDeltaToSeconds(td)
   .. py:staticmethod:: secondsSinceMidnight(dt)
   .. py:staticmethod:: dateTimeToDecimalYear(dt)

Component
---------

.. py:module:: iscesys.Component.Component

.. py:class:: Port(name=None,method=None,doc=None)

   .. py:method:: get/setName()
   .. py:method:: get/setMethod()
   .. py:method:: get/setObject()

.. py:class:: PortIterator()

   .. py:method:: add(port)
   .. py:method:: getPort(name=None)
   .. py:method:: hasPort(name=None)

.. py:class:: InputPorts()

.. py:class:: OutputPorts()

.. py:class:: Component()

   .. py:method:: wireInputPort(name=None, object=None)
   .. py:method:: listInputPorts()
   .. py:method:: getInputPort(name=None)
   .. py:method:: activateInputPorts()

   .. py:method:: wireOuputPort(name=None, object=None)
   .. py:method:: listOutputPorts()
   .. py:method:: getOutputPort(name=None)
   .. py:method:: activateOutputPorts()
