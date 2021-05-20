# -*- coding: utf-8 -*-
"""
The REsource eXtraction tool (rex)
"""
from __future__ import print_function, division, absolute_import
import os

from rex.joint_pd import *
from rex.rechunk_h5 import *
from rex.resource_extraction import *
from rex.temporal_stats import *
from rex.utilities import *

from rex.multi_file_resource import *
from rex.multi_time_resource import *
from rex.multi_year_resource import *
from rex.renewable_resource import *
from rex.resource import Resource
from rex.version import __version__

__author__ = """Michael Rossol"""
__email__ = "michael.rossol@nrel.gov"

REXDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(REXDIR), 'tests', 'data')
TREEDIR = os.path.join(os.path.dirname(REXDIR), 'bin', 'trees')
