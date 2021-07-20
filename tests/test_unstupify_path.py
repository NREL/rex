# -*- coding: utf-8 -*-
"""
pytests for unstupify_path utility
"""
import os
import pytest

from rex import TESTDATADIR, REXDIR
from rex.utilities.utilities import unstupify_path

HERE = os.path.realpath(__file__)
DIR, FILE = os.path.split(HERE)


def test_unstupify_home_path():
    """
    Test unstupify path logic for a path relative to home
    NOTE: This will not pass locally and is setup for Github Actions!!!
    """
    test = unstupify_path('~/rex/')
    assert test == os.path.dirname(REXDIR)


def test_unstupify_relative_file():
    """
    Test unstupify path logic for a relative file
    """
    test = unstupify_path(FILE)
    assert test == HERE


@pytest.mark.parametrize("path", ['./data',
                                  '.'])
def test_unstupify_relative_dir(path):
    """
    Test unstupify path logic for a relative directory
    """
    if path.endswith('data'):
        truth = TESTDATADIR
    else:
        truth = DIR

    test = unstupify_path(path)
    assert test == truth


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
