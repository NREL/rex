# -*- coding: utf-8 -*-
"""
Custom Exceptions and Errors
"""


class FileInputError(Exception):
    """
    Error during input file checks.
    """


class JSONError(Exception):
    """
    Error reading json file.
    """


class ExecutionError(Exception):
    """
    Error for execution failure
    """


class ResourceKeyError(Exception):
    """
    KeyError for Resources
    """


class ResourceRuntimeError(Exception):
    """
    RuntimeError for Resources
    """


class ResourceValueError(Exception):
    """
    ValueError for Resources
    """


class ResourceError(Exception):
    """
    Error for poorly formatted resource.
    """


class ParallelExecutionWarning(Warning):
    """
    Warning for parallel job execution.
    """


class SlurmWarning(Warning):
    """
    Warning for SLURM errors/warnings
    """


class ResourceWarning(Warning):
    """
    Warning during .h5 handling
    """


class SAMInputError(Exception):
    """
    Input error for SAM simulations
    """
