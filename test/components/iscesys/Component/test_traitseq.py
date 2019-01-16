#!/usr/bin/env python3
#
# Author: Ravi Lanka
# Copyright 2016
#

import isce
import unittest
import datetime, random
from iscesys.Component.Component import Component
from isceobj.Orbit.Orbit import Orbit, StateVector
from iscesys.Component.ProductManager import ProductManager
from iscesys.Component.TraitSeq import TraitSeq
from iscesys.Traits.Datetime import datetimeType
import numpy as np

def createParam():
  return datetime.datetime.now()

#Global variables for statevector time testing
svcount = 0
t0 = datetime.datetime.now()
dt = datetime.timedelta(seconds=1.000001)

def createFacility():
  # StateVector
  global svcount, t0, dt
  svcount += 1
  sv = StateVector()
  sv.configure()
  sv.setPosition(list(np.random.randint(10, size=(3,))))
  sv.setVelocity(list(np.random.randint(10, size=(3,))))
  t = t0 + (svcount-1)*dt
  #Force microseconds=0 on some statevectors
  if svcount%2 ==0:
    t = datetime.datetime(t.year,t.month,t.day,t.hour,t.minute,t.second)
  sv.setTime(t)
  return sv

class TestTraitSeq(unittest.TestCase):
  def setUp(self):
    self.stateVectors = TraitSeq()
    self.stateVectors.configure()

    pass

  def tearDown(self):
    pass

  def testDump(self):
    '''
    Test Dump and Load
    '''

    # Orbit class instance
    #print('.Test Dump and Load')

    # Add StateVectors to orbit
    for i in range(10):
      self.stateVectors.append(createFacility())

    # Create Product Manager and dump orbit
    pm = ProductManager()
    pm.configure()
    pm.dumpProduct(self.stateVectors, 'test.xml')

    # Load it back and compare it with the older instance
    newStateVec = pm.loadProduct('test.xml')
    self.assertEqual(self.stateVectors, newStateVec)

  def process(self, obj):
    # Create Product Manager and dump orbit
    pm = ProductManager()
    pm.configure()
    pm.dumpProduct(obj, 'test.xml')

    return pm.loadProduct('test.xml')

  def testAdd(self):
    '''
    Test the add
    '''
    # Orbit class instance
    #print('Test Add ')

    otherStateVecs = TraitSeq()
    otherStateVecs.configure()
    cummStateVecs = TraitSeq()
    cummStateVecs.configure()

    for i in range(10):
      stateVec = createFacility()
      if (i < 5):
        self.stateVectors.append(stateVec)
      else:
        otherStateVecs.append(stateVec)

      cummStateVecs.append(stateVec)

    self.assertEqual(cummStateVecs, self.stateVectors + otherStateVecs)
    return

  def testContains(self):
    '''
    Test if the Trait Sequence contains a particular element
    '''
    #print('Test contains ')

    # Add StateVectors to orbit
    for i in range(10):
      sv = createFacility()
      self.stateVectors.append(sv)

    random = createFacility()
    self.assertIn(sv, self.stateVectors)
    self.assertNotIn(random, self.stateVectors)
    return

  def testDelete(self):
    '''
    Test the delete
    '''
    #print('Test delete ')

    otherStateVecs = TraitSeq()
    otherStateVecs.configure()

    skip = random.randint(0,9)
    for i in range(10):
      sv = createFacility()
      if (i != skip):
        self.stateVectors.append(sv)
      otherStateVecs.append(sv)

    del otherStateVecs[skip]

    self.stateVectors = self.process(self.stateVectors)
    otherStateVecs = self.process(otherStateVecs)
    self.assertEqual(self.stateVectors, otherStateVecs)

    return

  def testGet(self):
    '''
    Test the Get
    '''
    #print('Test get ')

    svList = list()
    test = random.randint(0,9)
    for i in range(10):
      sv = createFacility()
      svList.append(sv)
      self.stateVectors.append(sv)

    self.stateVectors = self.process(self.stateVectors)
    self.assertEqual(svList[test], self.stateVectors[test])

  def testLen(self):
    '''
    Test the Length
    '''
    #print('Test Length ')
    test = random.randint(1,10)
    for i in range(test):
      sv = createFacility()
      self.stateVectors.append(sv)

    self.stateVectors = self.process(self.stateVectors)
    self.assertEqual(test, len(self.stateVectors))

  def testSet(self):
    '''
    Test the Length
    '''
    #print('Test Set ')
    skip = random.randint(0,9)
    otherStateVecs = TraitSeq()
    otherStateVecs.configure()
    testsv = createFacility()
    for i in range(10):
      sv = createFacility()
      self.stateVectors.append(sv)
      if skip == i:
        otherStateVecs.append(testsv)
      else:
        otherStateVecs.append(sv)

    self.stateVectors[skip] = testsv
    self.stateVectors = self.process(self.stateVectors)
    otherStateVecs = self.process(otherStateVecs)
    self.assertEqual(self.stateVectors, otherStateVecs)
    return

  def testClear(self):
    '''
    Test the clear
    '''
    #print('Test clear ')
    otherStateVecs = TraitSeq()
    otherStateVecs.configure()

    for i in range(10):
      sv = createFacility()
      self.stateVectors.append(sv)
    otherStateVecs.append(sv)

    self.stateVectors.clear()
    otherStateVecs.clear()
    self.assertEqual(self.stateVectors, otherStateVecs)
    return

  def testCount(self):
    '''
    Test the count
    '''
    #print('Test Count ')
    sv = createFacility()
    svTest = sv
    for i in range(10):
      self.stateVectors.append(sv)
      sv = createFacility()

    self.stateVectors.append(svTest)
    self.stateVectors = self.process(self.stateVectors)
    self.assertEqual(self.stateVectors.count(svTest), 2)
    return

  def testIndex(self):
    '''
    Test the Length
    '''
    #print('Test Length ')
    test = random.randint(0,9)
    for i in range(10):
      sv = createFacility()
      self.stateVectors.append(sv)
      if test == i:
        svTest = sv

    self.stateVectors = self.process(self.stateVectors)
    self.assertEqual(self.stateVectors.index(svTest), test)
    return

  def testInsert(self):
    #print('Test Insert ')
    otherStateVecs = TraitSeq()
    otherStateVecs.configure()

    test = random.randint(0,9)
    for i in range(10):
      sv = createFacility()
      self.stateVectors.append(sv)
      if test == i:
        svTest = sv
      else:
        otherStateVecs.append(sv)

    otherStateVecs.insert(test, svTest)
    self.stateVectors = self.process(self.stateVectors)
    otherStateVecs = self.process(otherStateVecs)
    self.assertEqual(self.stateVectors, otherStateVecs)
    return

  def testPop(self):
    '''
    Test the Pop
    '''
    #print('Test Pop ')
    otherStateVecs = TraitSeq()
    otherStateVecs.configure()

    for i in range(10):
      sv = createFacility()
      self.stateVectors.append(sv)
      otherStateVecs.append(sv)

    sv = createFacility()
    self.stateVectors.append(sv)

    # Pop
    self.stateVectors.pop()
    self.stateVectors = self.process(self.stateVectors)
    otherStateVecs = self.process(otherStateVecs)
    self.assertEqual(self.stateVectors, otherStateVecs)
    return

  def testRemove(self):
    '''
    Test Remove
    '''
    #print('Test Remove ')
    otherStateVecs = TraitSeq()
    otherStateVecs.configure()

    test = random.randint(0,9)
    for i in range(10):
      sv = createFacility()
      self.stateVectors.append(sv)
      if test == i:
        svTest = sv
      else:
        otherStateVecs.append(sv)

    self.stateVectors.remove(svTest)
    self.stateVectors = self.process(self.stateVectors)
    otherStateVecs = self.process(otherStateVecs)
    self.assertEqual(self.stateVectors, otherStateVecs)
    return

  def testReverse(self):
    '''
    Test Reverse
    '''
    #print('Test Reverse ')
    otherStateVecs = TraitSeq()
    otherStateVecs.configure()

    svList = []
    for i in range(10):
      sv = createFacility()
      self.stateVectors.append(sv)
      svList.append(sv)

    for sv in svList[::-1]:
      otherStateVecs.append(sv)

    self.stateVectors.reverse()
    self.stateVectors = self.process(self.stateVectors)
    otherStateVecs = self.process(otherStateVecs)
    self.assertEqual(self.stateVectors, otherStateVecs)
    return

  def testSort(self):
    '''
    Test Sort
    '''
    #print('Test Sort')
    import random
    otherStateVecs = TraitSeq()
    otherStateVecs.configure()

    svList = []
    for i in range(10):
      sv = createFacility()
      svList.append(sv)
      otherStateVecs.append(sv)

    random.shuffle(svList)
    for sv in svList:
      self.stateVectors.append(sv)

    self.stateVectors.sort(key=lambda sv: sv.time)
    self.stateVectors = self.process(self.stateVectors)
    otherStateVecs = self.process(otherStateVecs)
    self.assertEqual(self.stateVectors, otherStateVecs)
    return

def getSuite():
  suite = unittest.TestSuite()
  suite.addTest(TestTraitSeq('testDump'))
  suite.addTest(TestTraitSeq('testAdd'))
  return suite

if __name__ == "__main__":
    suite = unittest.TestSuite()
    tests = ['testDump', 'testAdd', 'testContains', \
             'testDelete', 'testGet', 'testLen', \
             'testSet', 'testClear', 'testCount', \
             'testIndex', 'testInsert', 'testPop', \
             'testRemove', 'testReverse', 'testSort']

    for T in tests:
      suite.addTest(TestTraitSeq(T))
#    unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.TextTestRunner(verbosity=0).run(suite)
