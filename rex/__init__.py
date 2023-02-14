# -*- coding: utf-8 -*-
"""
The REsource eXtraction tool (rex)
"""
from __future__ import print_function, division, absolute_import
import os

from rex.joint_pd import JointPD
from rex.multi_file_resource import (MultiFileNSRDB, MultiFileResource,
                                     MultiFileWTK)
from rex.multi_time_resource import (MultiTimeNSRDB, MultiTimeResource,
                                     MultiTimeWaveResource,
                                     MultiTimeWindResource)
from rex.multi_year_resource import (MultiYearNSRDB, MultiYearResource,
                                     MultiYearWaveResource,
                                     MultiYearWindResource)
from rex.multi_res_resource import MultiResolutionResource
from rex.rechunk_h5 import (ArrayChunkSize, TimeseriesChunkSize, CombineH5,
                            RechunkH5, get_dataset_attributes)
from rex.renewable_resource import (NSRDB, WaveResource, WindResource,
                                    GeothermalResource)
from rex.resource_extraction import (ResourceX, MultiFileResourceX,
                                     MultiTimeResourceX, MultiYearResourceX,
                                     NSRDBX, MultiFileNSRDBX,
                                     MultiTimeNSRDBX, MultiYearNSRDBX,
                                     WindX, MultiFileWindX,
                                     MultiTimeWindX, MultiYearWindX,
                                     WaveX, MultiTimeWaveX, MultiYearWaveX)
from rex.temporal_stats import TemporalStats
from rex.utilities import (SpawnProcessPool, SLURM, init_logger, init_mult,
                           setup_logger, log_mem, log_versions, LOGGERS,
                           SolarPosition, safe_json_load, jsonify_dict,
                           parse_year, check_res_file, parse_table,
                           check_eval_str, to_records_array)
from rex.resource import Resource
from rex.outputs import Outputs
from rex.version import __version__

__author__ = """Michael Rossol"""
__email__ = "michael.rossol@nrel.gov"

REXDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(REXDIR), 'tests', 'data')
TREEDIR = os.path.join(os.path.dirname(REXDIR), 'bin', 'trees')
