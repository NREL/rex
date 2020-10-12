# -*- coding: utf-8 -*-
"""
rex utilities.
"""
from .execution import SpawnProcessPool
from .hpc import SLURM, PBS
from .loggers import init_logger, init_mult, setup_logger, log_mem
from .solar_position import SolarPosition
from .toml_parser import TOMLParser
from .utilities import (safe_json_load, jsonify_dict, parse_year,
                        check_res_file, parse_table, Retry, check_eval_str)
