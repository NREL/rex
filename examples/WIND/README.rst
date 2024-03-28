NREL Wind Datasets
==================

This page describes some of the unique attributes of NREL wind datasets. For
instructions on how to access the data, see the docs page `here
<https://nrel.github.io/rex/misc/examples.nrel_data.html>`_.

WIND Toolkit
------------

The WIND Toolkit for North America was produced using the `Weather Research
and Forecasting Model (WRF)
<https://www.mmm.ucar.edu/weather-research-and-forecasting-model>`_.
The WRF model was initialized with the European Centre for Medium Range Weather
Forecasts Interim Reanalysis (ERA-Interm) data set with an initial grid spacing
of 54 km.  Three internal nested domains were used to refine the spatial
resolution to 18, 6, and finally 2 km.  The WRF model was run for years 2007 to
2014. While outputs were extracted from WRF at 5 minute time-steps, due to
storage limitations instantaneous hourly time-step are provided for all
variables while full 5 min resolution data is provided for wind speed and wind
direction only.

The following variables were extracted from the WRF model data:

- Wind Speed at 10, 40, 60, 80, 100, 120, 140, 160, 200 m
- Wind Direction at 10, 40, 60, 80, 100, 120, 140, 160, 200 m
- Temperature at 2, 10, 40, 60, 80, 100, 120, 140, 160, 200 m
- Pressure at 0, 100, 200 m
- Surface Precipitation Rate
- Surface Relative Humidity
- Inverse Monin Obukhov Length

The original WIND Toolkit dataset was produce using three distinct WRF domains
shown below. The CONUS domain for 2007-2013 was run by 3Tier while 2014 as well
as all years of the Canada and Mexico domains were run under NARIS. The data is
provided in three sets of files:

- CONUS: Extracted exclusively from the CONUS domain
- Canada: Combined data from the Canada and CONUS domains
- Mexico: Combined data from the Mexico and CONUS domains

WINDX CLI
---------

The `WINDX <https://nrel.github.io/rex/rex/rex.resource_extraction.wind_cli.html#windx>`_
command line utility provides the following options and commands:

.. code-block:: bash

  WINDX --help

  Usage: WINDX [OPTIONS] COMMAND [ARGS]...

    WindX Command Line Interface

  Options:
    -h5, --wind_h5 PATH  Path to Resource .h5 file  [required]
    -o, --out_dir PATH   Directory to dump output files  [required]
    -v, --verbose        Flag to turn on debug logging. Default is not verbose.
    --help               Show this message and exit.

  Commands:
    dataset     Extract a single dataset
    multi-site  Extract multiple sites given in '--sites' .csv or .json as...
    sam-file    Extract all datasets at the given hub height needed for SAM...

References
----------

For more information about the WIND Toolkit please see the `website. <https://www.nrel.gov/grid/wind-toolkit.html>`_
Users of the WIND Toolkit should use the following citations:

- `Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. Overview and Meteorological Validation of the Wind Integration National Dataset Toolkit (Technical Report, NREL/TP-5000-61740). Golden, CO: National Renewable Energy Laboratory. <https://www.nrel.gov/docs/fy15osti/61740.pdf>`_
- `Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. "The Wind Integration National Dataset (WIND) Toolkit." Applied Energy 151: 355366. <https://www.sciencedirect.com/science/article/pii/S0306261915004237?via%3Dihub>`_
- `Lieberman-Cribbin, W., C. Draxl, and A. Clifton. 2014. Guide to Using the WIND Toolkit Validation Code (Technical Report, NREL/TP-5000-62595). Golden, CO: National Renewable Energy Laboratory. <https://www.nrel.gov/docs/fy15osti/62595.pdf>`_
- `King, J., A. Clifton, and B.M. Hodge. 2014. Validation of Power Output for the WIND Toolkit (Technical Report, NREL/TP-5D00-61714). Golden, CO: National Renewable Energy Laboratory. <https://www.nrel.gov/docs/fy14osti/61714.pdf>`_
