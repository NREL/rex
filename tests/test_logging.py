# -*- coding: utf-8 -*-
"""
pytests for logging methodology
"""
import os
import pytest
import tempfile

from rex.utilities.exceptions import LoggerWarning
from rex.utilities.loggers import (init_logger, LOGGERS, add_handlers,
                                   get_handler)


def test_loggers():
    """
    Test logger initilization and handling
    """
    with tempfile.TemporaryDirectory() as td:
        logger = init_logger('rex.test')
        assert len(logger.handlers) == 1
        assert len(LOGGERS.loggers) == 1

        # Add file handler
        log_file = os.path.join(td, 'test.log')
        handler = get_handler(log_file=log_file)
        logger = add_handlers(logger, [handler])
        assert len(logger.handlers) == 2

        # update stream handler
        handler = get_handler(log_level='debug')
        logger = add_handlers(logger, [handler])
        assert len(logger.handlers) == 2

        # add second file_handler
        log_file = os.path.join(td, 'log.log')
        handler = get_handler(log_level='debug', log_file=log_file)
        logger = add_handlers(logger, [handler])
        assert len(logger.handlers) == 3

        # re-initilize 'rex.test'
        LOGGERS.clear()

        logger.handlers.clear()
        log_file = os.path.join(td, 'test.log')
        logger = init_logger('rex.test', log_file=log_file, log_level='DEBUG')
        assert len(logger.handlers) == 2
        assert len(LOGGERS.loggers) == 1
        for h in logger.handlers:
            h.flush()
            h.close()

        # Add parent logger removing 'rex.test' but inheriting handlers and
        # level
        logger = init_logger('rex')
        assert len(logger.handlers) == 2
        assert logger.level == 10
        assert len(LOGGERS.loggers) == 1

    LOGGERS.clear()


def test_bad_log_dir():
    """
    test bad log dir warning and converstion to stream logger
    """
    with pytest.warns(LoggerWarning):
        log_file = '/abc/log.log'
        logger = init_logger(__name__, log_file=log_file)
        assert len(logger.handlers) == 1
        assert logger.handlers[0].name == 'stream'

    LOGGERS.clear()


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
