NREL Wind Datasets
==================

This page describes some of the unique attributes of NREL wind datasets. For
instructions on how to access the data, see the docs page `here
<https://nrel.github.io/rex/misc/examples.nrel_data.html>`_.

WIND Toolkit v1.0.0
-------------------

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

Note that the WIND Toolkit version 1.0.0 (described above) includes the 2007-2013 years of the files located at the following hsds domain: `/nrel/wtk/conus/`

The next generation WIND Toolkit version 1.2.0 includes years 2018-2020 in the same HSDS directory `/nrel/wtk/conus/` but with different meta data. 


2023 National Offshore Wind data set (NOW-23)
---------------------------------------------
 
The 2023 National Offshore Wind data set (NOW-23) is the latest wind resource data set for offshore regions in the United States, which supersedes, for its offshore component, the Wind Integration National Dataset (WIND) Toolkit v1.0.0, which was published about a decade ago and is currently one of the primary resources for stakeholders conducting wind resource assessments in the continental United States.

The NOW-23 data set was produced using the Weather Research and Forecasting Model (WRF) version 4.2.1. A regional approach was used: for each offshore region, the WRF setup was selected based on validation against available observations. The WRF model was initialized with the European Centre for Medium Range Weather Forecasts 5 Reanalysis (ERA-5) data set, using a 6-hour refresh rate. The model is configured with an initial horizontal grid spacing of 6 km and an internal nested domain that refined the spatial resolution to 2 km. The model is run with 61 vertical levels, with 12 levels in the lower 300m of the atmosphere, stretching from 5 m to 45 m in height. The MYNN planetary boundary layer and surface layer schemes were used the North Atlantic, Mid Atlantic, Great Lakes, Hawaii, and North Pacific regions. On the other hand, using the YSU planetary boundary layer and MM5 surface layer schemes resulted in a better skill in the South Atlantic, Gulf of Mexico, and South Pacific regions. A more detailed description of the WRF model setup can be found in the WRF namelist files linked at the bottom of this page.

For all regions, the NOW-23 data set coverage starts on January 1, 2000. For Hawaii and the North Pacific regions, NOW-23 goes until December 31, 2019. For the South Pacific region, the model goes until 31 December, 2022. For all other regions, the model covers until December 31, 2020. Outputs are available at 5 minute resolution, and for all regions we have also included output files at hourly resolution.
 
The following variables are available:
 
Planetary boundary layer height (m)
Pressure at 0m, 100m, 200m, and 300m (Pa)
Temperature at 2m, 10m, 20-m intervals between 20m and 300m, 400m, and 500m (°C)
Wind direction at 10m, 20-m intervals between 20m and 300m, 400m, and 500m (° from N)
Wind speed at 10m, 20-m intervals between 20m and 300m, 400m, and 500m (m s-1)
Friction velocity at 2m (m s-1)
Sea surface temperature (°C)
Skin Temperature (°C)
Surface heat flux (W m-2)
Relative humidity at 2m (%)
Inverse Monin-Obukhov length (m-1)
Roughness length (m)

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
