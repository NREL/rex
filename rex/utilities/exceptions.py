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


class HpcError(Exception):
    """
    Error for HPC failure
    """


class SlurmError(Exception):
    """
    Error for SLURM failure
    """


class ExtrapolationWarning(Warning):
    """
    Warning for when value will be extrapolated
    """


class MoninObukhovExtrapolationError(Exception):
    """
    Custom error when WindResource._monin_obhukov_extrapolation fails
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


class RetryError(Exception):
    """
    Custom Error for Retry Decorator
    """


class ParallelExecutionWarning(Warning):
    """
    Warning for parallel job execution.
    """


class PbsWarning(Warning):
    """
    Warning for PBS errors/warnings
    """


class SlurmWarning(Warning):
    """
    Warning for SLURM errors/warnings
    """


class ResourceWarning(Warning):
    """
    Warning during .h5 handling
    """


class RetryWarning(Warning):
    """
    Warning that Retry decorator is trying again
    """


class SAMInputWarning(Warning):
    """
    Warning for bad SAM inputs
    """
