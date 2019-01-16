#!/usr/bin/env python
#
# Author: Eric Gurrola
# Copyright 2016
#

from __future__ import print_function
import subprocess

def print_test_banner():
    print("\n--- Testing...1,2,3 ---")

def print_entering_banner(tpackage):
    print("+++ entering, {}".format(tpackage))

def run_tests_and_print(testFiles):
    for t in testFiles:
        x = run_test(t)
        print("{0}: {1}".format(t,x))
    return

def run_tests_no_print(listFiles):
    r = []
    for t in listFiles:
        r.append(run_test(t))
    return r

def run_test(t):
    p = subprocess.Popen(['python3', t], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    x = p.communicate()[0].replace(b'\n',b' ').replace(b'-',b'')
    return x.decode(encoding='UTF-8')

def cleanup(cleanup_list):
  import os
  for f in cleanup_list:
      if os.path.isfile(f):
          os.remove(f)
