**********************************************
Welcome to The REsource eXtraction (rex) tool!
**********************************************

.. image:: https://github.com/NREL/rex/workflows/Documentation/badge.svg
    :target: https://nrel.github.io/rex/

.. image:: https://github.com/NREL/rex/workflows/Pytests/badge.svg
    :target: https://github.com/NREL/rex/actions?query=workflow%3A%22Pytests%22

.. image:: https://github.com/NREL/rex/workflows/Lint%20Code%20Base/badge.svg
    :target: https://github.com/NREL/rex/actions?query=workflow%3A%22Lint+Code+Base%22

.. image:: https://img.shields.io/pypi/pyversions/NREL-rex.svg
    :target: https://pypi.org/project/NREL-rex/

.. image:: https://badge.fury.io/py/NREL-rex.svg
    :target: https://badge.fury.io/py/NREL-rex

.. image:: https://anaconda.org/nrel/nrel-rex/badges/version.svg
    :target: https://anaconda.org/nrel/nrel-rex

.. image:: https://anaconda.org/nrel/nrel-rex/badges/license.svg
    :target: https://anaconda.org/nrel/nrel-rex

.. image:: https://codecov.io/gh/nrel/rex/branch/main/graph/badge.svg?token=WQ95L11SRS
    :target: https://codecov.io/gh/nrel/rex

.. image:: https://zenodo.org/badge/253541811.svg
   :target: https://zenodo.org/badge/latestdoi/253541811

.. inclusion-intro

rex command line tools
======================

- `rex <https://nrel.github.io/rex/_cli/rex.html#rex>`_
- `NSRDBX <https://nrel.github.io/rex/_cli/NSRDBX.html#NSRDBX>`_
- `WINDX <https://nrel.github.io/rex/_cli/WINDX.html#WINDX>`_
- `US-wave <https://nrel.github.io/rex/_cli/US-wave.html#US-wave>`_
- `WaveX <https://nrel.github.io/rex/_cli/WaveX.html#Wavex>`_
- `MultiYearX <https://nrel.github.io/rex/_cli/MultiYearX.html#MultiYearX>`_
- `rechunk <https://nrel.github.io/rex/_cli/rechunk.html#rechunk>`_
- `temporal-stats <https://nrel.github.io/rex/_cli/temporal-stats.html#temporal-stats>`_
- `wind-rose <https://nrel.github.io/rex/_cli/wind-rose.html#wind-rose>`_

Using Eagle Env
===============

If you would like to run `rex` on Eagle (NREL's HPC) you can use a pre-compiled
conda env:

.. code-block:: bash

    conda activate /shared-projects/rev/modulefiles/conda/envs/rev/

or

.. code-block:: bash

    source activate /shared-projects/rev/modulefiles/conda/envs/rev/

Installing rex
==============

NOTE: The installation instruction below assume that you have python installed
on your machine and are using `conda <https://docs.conda.io/en/latest/index.html>`_
as your package/environment manager.

Option 1: Install from PIP or Conda (recommended for analysts):
---------------------------------------------------------------

1. Create a new environment:
    ``conda create --name rex``

2. Activate directory:
    ``conda activate rex``

3. Install rex:
    1) ``pip install NREL-rex`` or
    2) ``conda install nrel-rex --channel=nrel``

       - NOTE: If you install using conda and want to use `HSDS <https://github.com/NREL/hsds-examples>`_
         you will also need to install h5pyd manually: ``pip install h5pyd``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone git@github.com:NREL/rex.git``

2. Create ``rex`` environment and install package
    1) Create a conda env: ``conda create -n rex``
    2) Run the command: ``conda activate rex``
    3) cd into the repo cloned in 1.
    4) prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    5) Install ``rex`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)

3. Check that ``rex`` was installed successfully
    1) From any directory, run the following commands. This should return the
       help pages for the CLI's.

        - ``rex``
        - ``NSRDBX``
        - ``WINDX``
        - ``US-wave``

Recommended Citation
====================

Update with current version and DOI:

Michael Rossol, Grant Buster. The REsource Extraction Tool (rex).
https://github.com/NREL/rex (version v0.2.43), 2021.
https://doi.org/10.5281/zenodo.4499033.
