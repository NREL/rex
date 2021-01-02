# -*- coding: utf-8 -*-
"""
Logging wrapper
"""
from copy import deepcopy
import logging
import os
import psutil
import sys
import time
from warnings import warn

from rex.utilities.exceptions import LoggerWarning

FORMAT = '%(levelname)s - %(asctime)s [%(filename)s:%(lineno)d] : %(message)s'
LOG_LEVEL = {'INFO': logging.INFO,
             'DEBUG': logging.DEBUG,
             'WARNING': logging.WARNING,
             'ERROR': logging.ERROR,
             'CRITICAL': logging.CRITICAL}


def create_dirs(dir_path):
    """
    Wrapper for checking and making directory to handle parallel processing

    Parameters
    ----------
    dir_path : str
        Directory path to create if it doesn't exist
    """
    i = 0
    while True:
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            break
        except FileExistsError:
            time.sleep(0.01)
        except Exception as ex:
            i += 1
            if i == 2:
                msg = 'Cannot create {}: {}'.format(dir_path, ex)
                raise FileExistsError(msg) from ex


def get_handler(log_level="INFO", log_file=None, log_format=FORMAT):
    """
    get logger handler

    Parameters
    ----------
    log_level : str
        handler-specific logging level, must be key in LOG_LEVEL.
    log_file : str
        path to the log file
    log_format : str
        format string to use with the logging package

    Returns
    -------
    handler : logging.FileHandler | logging.StreamHandler
        handler to add to logger
    """
    if log_file:
        # file handler with mode "a"
        log_file = os.path.normpath(log_file)
        log_dir = os.path.dirname(log_file)
        if os.path.exists(log_dir):
            name = log_file
            handler = logging.FileHandler(log_file, mode='a')
        else:
            warn('{} does not exist, FileHandler will be converted to a '
                 'StreamHandler'.format(log_dir), LoggerWarning)
            name = 'stream'
            handler = logging.StreamHandler(sys.stdout)
    else:
        # stream handler to system stdout
        name = 'stream'
        handler = logging.StreamHandler(sys.stdout)

    if log_format:
        logformat = logging.Formatter(log_format)
        handler.setFormatter(logformat)

    # Set a handler-specific logging level (root logger should be at debug)
    handler.setLevel(LOG_LEVEL[log_level.upper()])
    handler.set_name(name)

    return handler


def add_handlers(logger, handlers):
    """
    Add handlers to logger ensuring they do not already exist

    Parameters
    ----------
    logger : logging.logger
        Logger to add handlers to
    handlers : list
        Handlers to add to logger

    Returns
    -------
    logger : logging.logger
        Logger with updated handlers
    """
    current_handlers = {h.name: h for h in logger.handlers}
    for handler in handlers:
        name = handler.name
        if name not in current_handlers:
            logger.addHandler(handler)
            current_handlers.update({name: handler})
        else:
            h = current_handlers[name]
            if handler.level < h.level:
                h.setLevel(handler.level)

    return logger


def setup_logger(logger_name, stream=True, log_level="INFO", log_file=None,
                 log_format=FORMAT):
    """
    Setup logging instance with given name and attributes

    Parameters
    ----------
    logger_name : str
        Name of logger
    stream : bool, optional
        Add a StreamHandler along with FileHandler, by default True
    log_level : str, optional
        Level of logging to capture, must be key in LOG_LEVEL. If multiple
        handlers/log_files are requested in a single call of this function,
        the specified logging level will be applied to all requested handlers,
        by default "INFO"
    log_file : str | list, optional
        Path to file to use for logging, if None use a StreamHandler
        list of multiple handlers is permitted, by default None
    log_format : str, optional
        Format for loggings, by default FORMAT

    Returns
    -------
    logger : logging.logger
        instance of logger for given name, with given level and added handler
    """
    logger = logging.getLogger(logger_name)
    # Set root logger to debug, handlers will control levels above debug
    level = logger.level
    set_level = LOG_LEVEL[log_level.upper()]
    if level == 0 or set_level < level:
        logger.setLevel(LOG_LEVEL[log_level])

    handlers = []
    if isinstance(log_file, list):
        for h in log_file:
            handlers.append(get_handler(log_level=log_level, log_file=h,
                                        log_format=log_format))
    else:
        handlers.append(get_handler(log_level=log_level, log_file=log_file,
                                    log_format=log_format))

    if stream:
        if log_file is not None:
            handlers.append(get_handler())

    logger = add_handlers(logger, handlers)

    return logger


def clear_handlers(logger):
    """
    Clear all handlers from logger

    Parameters
    ----------
    logger : logging.logger
        Logger to remove all handlers from

    Returns
    -------
    logger : logging.logger
        Logger with all handlers removed
    """
    handlers = logger.handlers.copy()
    for handler in handlers:
        # Copied from `logging.shutdown`.
        try:
            handler.acquire()
            handler.flush()
            handler.close()
        except (OSError, ValueError):
            pass
        finally:
            handler.release()

        logger.removeHandler(handler)

    logger.handlers.clear()

    return logger


class LoggingAttributes:
    """
    Class to store and pass logging attributes to modules
    """
    def __init__(self):
        self._loggers = {}

    def __repr__(self):
        msg = ("{} containing {} loggers"
               .format(self.__class__.__name__, self.logger_names))

        return msg

    def __setitem__(self, logger_name, attributes):
        log_attrs = self[logger_name]
        log_attrs = self._update_attrs(log_attrs, attributes)
        self._loggers[logger_name] = log_attrs

    def __getitem__(self, logger_name):
        return self._loggers.get(logger_name, {}).copy()

    def __contains__(self, logger_name):
        return logger_name in self.logger_names

    @property
    def loggers(self):
        """
        Available loggers

        Returns
        -------
        dict
        """
        return self._loggers

    @property
    def logger_names(self):
        """
        Logger names

        Returns
        -------
        list
        """
        return sorted(self.loggers.keys())

    @staticmethod
    def _check_file_handlers(handlers, new_handlers):
        """
        Check to see if new file handlers should be added to the logger

        Parameters
        ----------
        handlers : list
            None or list of existing file handlers
        new_handlers : list
            List of new file handlers to add to logger

        Returns
        -------
        handlers : list
           Updated list of valid log files to add to handler
        """
        if not isinstance(new_handlers, (list, tuple)):
            new_handlers = [new_handlers]
            new_handlers = [os.path.normpath(h) for h in new_handlers
                            if h is not None]

        for h in new_handlers:
            if h not in handlers:
                log_dir = os.path.dirname(h)
                if os.path.exists(log_dir):
                    # check if each handler has been previously set
                    handlers.append(h)
                else:
                    warn('{} does not exist, FileHandler will be '
                         'converted to a StreamHandler'
                         .format(log_dir), LoggerWarning)

        return handlers

    @classmethod
    def _update_attrs(cls, log_attrs, new_attrs):
        """
        Update logger attributes with new attributes
        - Add any new log files
        - Reduce log level

        Parameters
        ----------
        log_attrs : dict
            Existing logger attributes
        new_attrs : dict
            New logger attributes

        Returns
        -------
        log_attrs
            upated logger attributes
        """
        for attr, value in new_attrs.items():
            if attr == 'log_file' and value:
                if value is not None:
                    handlers = log_attrs.get('log_file', None)
                    if handlers is None:
                        handlers = []

                    handlers = cls._check_file_handlers(handlers, value)
                    if not handlers:
                        handlers = None
                else:
                    handlers = None

                log_attrs[attr] = handlers
            elif attr == 'log_level':
                log_value = LOG_LEVEL[log_attrs.get('log_level', 'INFO')]
                attr_value = LOG_LEVEL[value.upper()]
                if attr_value < log_value:
                    log_attrs[attr] = value.upper()
            else:
                log_attrs[attr] = value

        return log_attrs

    def _cleanup(self):
        """
        Cleanup loggers and attributes by combining dependent and parent
        loggers
        """
        loggers = deepcopy(self.loggers)
        parent = None
        parent_attrs = {}
        for name in self.logger_names:
            attrs = self.loggers[name]
            if parent is None:
                if "__main__" not in name:
                    p = name.split('.')[0]
                    if p == name:
                        parent = name
                        parent_attrs = attrs
            elif name.startswith(parent + '.'):
                # Remove child logger from internal record
                parent_attrs = self._update_attrs(parent_attrs, attrs)
                del loggers[name]
                # Remove any handlers from child loggers to prevent duplicate
                # logging
                clear_handlers(logging.getLogger(name))

        if parent is not None:
            loggers[parent] = parent_attrs

        self._loggers = loggers

    def _check_for_parent(self, logger_name):
        """
        Check for existing parent loggers

        Parameters
        ----------
        logger_name : str
            Name of logger to initialize

        Returns
        -------
        parent : str
            Name of parent logger to initialize, if no existing parent None
        parent_attrs : dict
            Parent logger attributes
        """
        parent = None
        parent_attrs = {}
        for name, attrs in self.loggers.items():
            if logger_name.startswith(name):
                parent = name
                parent_attrs = attrs

        return parent, parent_attrs

    def set_logger(self, logger_name, stream=True, log_level="INFO",
                   log_file=None, log_format=FORMAT):
        """
        Setup logging instance with given name and attributes

        Parameters
        ----------
        logger_name : str
            Name of logger
        stream : bool, optional
            Add a StreamHandler along with FileHandler, by default True
        log_level : str, optional
            Level of logging to capture, must be key in LOG_LEVEL. If multiple
            handlers/log_files are requested in a single call of this function,
            the specified logging level will be applied to all requested
            handlers, by default "INFO"
        log_file : str | list, optional
            Path to file to use for logging, if None use a StreamHandler
            list of multiple handlers is permitted, by default None
        log_format : str, optional
            Format for loggings, by default FORMAT

        Returns
        -------
        logger : logging.logger
            instance of logger for given name, with given level and added
            handlers(s)
        """
        attrs = {"log_level": log_level, "log_file": log_file,
                 "log_format": log_format, 'stream': stream}
        log_attrs = self[logger_name]
        if logger_name not in self:
            parent, parent_attrs = self._check_for_parent(logger_name)
            if parent and attrs != parent_attrs:
                logger_name = parent
                log_attrs = parent_attrs

        attrs = self._update_attrs(log_attrs, attrs)

        self._loggers[logger_name] = attrs
        self._cleanup()

        return setup_logger(logger_name, **attrs)

    def init_loggers(self, loggers):
        """
        Extract logger attributes and initialize logger

        Parameters
        ----------
        loggers : str | list
            Logger names to initialize
        """
        if not isinstance(loggers, (list, tuple)):
            loggers = [loggers]

        for logger_name in loggers:
            if logger_name in self:
                attrs = self[logger_name]
                setup_logger(logger_name, **attrs)

    def clear(self):
        """
        Clear all log handlers
        """
        for name, logger in logging.Logger.manager.loggerDict.items():
            if isinstance(logger, logging.Logger):
                for p_name in self.logger_names:
                    if name.startswith(p_name):
                        clear_handlers(logger)
                        break

        self._loggers = {}


LOGGERS = LoggingAttributes()


def init_logger(logger_name, stream=True, log_level="INFO", log_file=None,
                log_format=FORMAT, prune=True):
    """
    Starts logging instance and adds logging attributes to LOGGERS

    Parameters
    ----------
    logger_name : str
        Name of logger to initialize
    stream : bool, optional
        Add a StreamHandler along with FileHandler, by default True
    log_level : str, optional
        Level of logging to capture, must be key in LOG_LEVEL. If multiple
        handlers/log_files are requested in a single call of this function,
        the specified logging level will be applied to all requested handlers,
        by default "INFO"
    log_file : str | list, optional
        Path to file to use for logging, if None use a StreamHandler
        list of multiple handlers is permitted, by default None
    log_format : str, optional
        Format for loggings, by default FORMAT
    prune : bool, optional
        Remove child logger handlers if parent logger is added, parent will
        inherit child's handlers, by default True

    Returns
    -------
    logger : logging.logger
        logging instance that was initialized
    """
    kwargs = {"log_level": log_level, "log_file": log_file,
              "log_format": log_format, 'stream': stream}
    if prune:
        logger = LOGGERS.set_logger(logger_name, **kwargs)
    else:
        LOGGERS[logger_name] = kwargs
        logger = setup_logger(logger_name, **kwargs)

    return logger


def init_mult(name, logdir, modules, verbose=False, node=False):
    """Init multiple loggers to a single file or stdout.

    Parameters
    ----------
    name : str
        Job name; name of log file.
    logdir : str
        Target directory to save .log files.
    modules : list | tuple
        List of reV modules to initialize loggers for.
    verbose : bool
        Option to turn on debug logging.
    node : bool
        Flag for whether this is a node-level logger. If this is a node logger,
        and the log level is info, the log_file will be None (sent to stdout).

    Returns
    -------
    loggers : list
        List of logging instances that were initialized.
    """

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    if logdir is not None and not os.path.exists(logdir):
        os.makedirs(logdir)

    loggers = []
    for module in modules:
        if logdir is not None:
            log_file = os.path.join(logdir, '{}.log'.format(name))
        else:
            log_file = None

        # check for redundant loggers in the LOGGERS singleton
        logger = LOGGERS[module]

        if node and log_level == 'INFO':
            # Node level info loggers only go to STDOUT/STDERR files
            logger = init_logger(module, log_level=log_level, log_file=None)
        else:
            logger = init_logger(module, log_level=log_level,
                                 log_file=log_file)

        loggers.append(logger)

    return loggers


def log_mem(logger, log_level='DEBUG'):
    """Log the memory usage to the input logger object.

    Parameters
    ----------
    logger : logging.Logger
        Logger object to log memory message to.
    log_level : str
        DEBUG or INFO for different log levels for this log message.

    Returns
    -------
    msg : str
        Memory utilization log message string.
    """
    mem = psutil.virtual_memory()
    msg = ('Memory utilization is {0:.3f} GB out of '
           '{1:.3f} GB total ({2:.1f}% used)'
           .format(mem.used / 1e9, mem.total / 1e9,
                   100 * mem.used / mem.total))
    if log_level.upper() == 'DEBUG':
        logger.debug('\t- {}'.format(msg))
    else:
        logger.info(msg)

    return msg
