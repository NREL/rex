# -*- coding: utf-8 -*-
"""
rex utilities.
"""
from .fun_utils import (has_class, get_class, is_standalone_fun,
                        is_instance_method, get_fun_str, get_arg_str,
                        get_fun_call_str)
from .execution import SpawnProcessPool
from .hpc import SLURM, PBS
from .loggers import (init_logger, init_mult, setup_logger, log_mem,
                      log_versions, LOGGERS)
from .solar_position import SolarPosition
from .toml_parser import TOMLParser
from .utilities import *
