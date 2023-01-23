# -*- coding: utf-8 -*-
"""
pytests for unstupify_path utility
"""
import os
import pytest

from rex import TESTDATADIR
from rex.utilities.utilities import unstupify_path

HERE = os.path.realpath(__file__)
DIR, FILE = os.path.split(HERE)
os.chdir(DIR)  # rex assumes that all tests are run from /tests/


def test_unstupify_home_path():
    """
    Test unstupify path logic for a path relative to home
    """
    home = os.path.expanduser('~')
    file = sorted(os.listdir(home))[0]
    truth = os.path.join(home, file)
    test = unstupify_path('~/' + file)
    assert os.path.samefile(test, truth)


def test_unstupify_relative_file():
    """
    Test unstupify path logic for a relative file
    """
    test = unstupify_path(FILE)
    assert os.path.samefile(test, HERE)


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
    assert os.path.samefile(test, truth)


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
