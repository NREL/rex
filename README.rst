***
rex
***

.. image:: https://github.com/NREL/rex/workflows/Documentation/badge.svg
    :target: https://nrel.github.io/rex/

.. image:: https://github.com/NREL/rex/workflows/Pytests/badge.svg
    :target: https://github.com/NREL/rex/actions?query=workflow%3A%22Pytests%22

.. image:: https://img.shields.io/pypi/pyversions/NREL-rex.svg
    :target: https://pypi.org/project/NREL-rex/

.. image:: https://badge.fury.io/py/NREL-rex.svg
    :target: https://badge.fury.io/py/NREL-rex

.. image:: https://anaconda.org/nrel/nrel-rex/badges/version.svg
    :target: https://anaconda.org/nrel/nrel-rex

.. image:: https://anaconda.org/nrel/nrel-rex/badges/platforms.svg
    :target: https://anaconda.org/nrel/nrel-rex

.. image:: https://anaconda.org/nrel/nrel-rex/badges/license.svg
    :target: https://anaconda.org/nrel/nrel-rex

The REsource eXtraction (rex) tool

.. inclusion-intro

rex command line tools
======================

- `rex <https://nrel.github.io/rex/rex/rex.resource_cli.html#rex>`_
- `NSRDB <https://nrel.github.io/rex/rex/rex.solar_cli.html#nsrdb>`_
- `WIND <https://nrel.github.io/rex/rex/rex.wind_cli.html#wind>`_
- `rechunk <https://nrel.github.io/rex/rex/rex.rechunk_h5.rechunk_cli.html#rechunk>`_

Using Eagle Module
==================

If you would like to run reV on Eagle (NREL's HPC) you can use a pre-compiled
module:

.. code-block:: bash

    module use /datasets/modulefiles
    module load rex

Installing rex
==============

Option 1: Install from PIP or Conda (recommended for analysts):
---------------------------------------------------------------

1. Create a new environment:
    ``conda create --name rex``

2. Activate directory:
    ``conda activate rex``

3. Install rex:
    1) ``pip install NREL-rex`` or
    2) ``conda install nrel-rex --channel=nrel``

       - NOTE: If you conda install and want to you HSDS you'll also need to
         install h5pyd manually: ``pip install h5pyd``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone https://github.com/NREL/rex.git``
    1) enter github username
    2) enter github password

2. Install reV environment and modules (using conda)
    1) cd into reV repo cloned above
    2) cd into ``bin/$OS/``
    3) run the command: ``conda env create -f rex.yml``. If conda can't find
       any packages, try removing them from the yml file.

    4) run the command: ``conda activate rex``
    5) prior to running ``pip`` below, make sure the branch is correct (install
       from master!)

    6) cd back to the rex repo (where setup.py is located)
    7) install pre-commit: ``pre-commit install``
    8) run ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)

3. Check that rev was installed successfully
    1) From any directory, run the following commands. This should return the
       help pages for the CLI's.

        - ``rex``
        - ``NSRDB``
        - ``WIND``
        - ``rechunk``
