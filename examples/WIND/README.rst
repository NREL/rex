Wind Integration National Dataset (WIND) Toolkit
================================================

Model
-----

Wind resource data for North America was produced using the `Weather Research and Forecasting Model (WRF) <https://www.mmm.ucar.edu/weather-research-and-forecasting-model>`_.
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

Domains
-------

The wind resource was produce using three distinct WRF domains shown below. The
CONUS domain for 2007-2013 was run by 3Tier while 2014 as well as all years of
the Canada and Mexico domains were run under NARIS. The data is provided in
three sets of files:

- CONUS: Extracted exclusively from the CONUS domain
- Canada: Combined data from the Canada and CONUS domains
- Mexico: Combined data from the Mexico and CONUS domains

Data Format
-----------

The data is provided in high density data file (.h5) separated by year. The
variables mentioned above are provided in 2 dimensional time-series arrays with
dimensions (time x location). The temporal axis is defined by the
``time_index`` dataset, while the positional axis is defined by the ``meta``
dataset. For storage efficiency each variable has been scaled and stored as an
integer. The scale-factor is provided in the ``scale-factor`` attribute.  The
units for the variable data is also provided as an attribute (``units``).

Data Access Examples
--------------------

Example scripts to extract wave resource data using the command line or python
are provided below:

The easiest way to access and extract data is by using the Resource eXtraction
tool `rex <https://nrel.github.io/rex/>`_

To use ``rex`` with `HSDS
<https://www.hdfgroup.org/solutions/highly-scalable-data-service-hsds/>`_ you
will need to install ``h5pyd`` with ``pip install h5pyd`` and then run
``hsconfigure`` as described in the `NREL HSDS Examples
<https://github.com/NREL/hsds-examples>`_.

*Please note that our HSDS service is for demonstration purposes only, if you
would like to use HSDS for production runs of reV please setup your own
service: https://github.com/HDFGroup/hsds and point it to our public HSDS
bucket: s3://nrel-pds-hsds**

WINDX CLI
+++++++++

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

WindX python class
++++++++++++++++++

.. code-block:: python

  from rex import WindX

  wtk_file = '/nrel/wtk/conus/wtk_conus_2014.h5'
  with WindX(wtk_file, hsds=True) as f:
      meta = f.meta
      time_index = f.time_index
      wspd_100m = f['windspeed_100m', :, ::1000]

Note: `WindX` will automatically interpolate to the desired hub-height:

.. code-block:: python

  from rex import WindX

  wtk_file = '/nrel/wtk/conus/wtk_conus_2014.h5'
  with WindX(wtk_file, hsds=True) as f:
      print(f.datasets)  # not 90m is not a valid dataset
      wspd_90m = f['windspeed_90m', :, ::1000]

`WindX` also allows easy extraction of the nearest site to a desired (lat, lon)
location:

.. code-block:: python

  from rex import WindX

  wtk_file = '/nrel/wtk/conus/wtk_conus_2014.h5'
  nwtc = (39.913561, -105.222422)
  with WindX(wtk_file, hsds=True) as f:
      nwtc_wspd = f.get_lat_lon_df('windspeed_100m', nwtc)


or to extract all sites in a given region:

.. code-block:: python

  from rex import WindX

  wtk_file = '/nrel/wtk/conus/wtk_conus_2014.h5'
  state = 'Colorado'
  with WindX(wtk_file, hsds=True) as f:
      date = '2014-07-04 18:00:00'
      wspd_map = f.get_timestep_map('windspeed_100m', date, region=region,
                                    region_col='state')

Lastly, `WindX` can be used to extract all variables needed to run SAM at a
given location:

.. code-block:: python

  from rex import WindX

  wtk_file = '/nrel/wtk/conus/wtk_conus_2014.h5'
  nwtc = (39.913561, -105.222422)
  with WindX(wtk_file, hsds=True) as f:
      nwtc_sam_vars = f.get_SAM_lat_lon(nwtc)


References
----------

For more information about the WIND Toolkit please see the `website. <https://www.nrel.gov/grid/wind-toolkit.html>`_
Users of the WIND Toolkit should use the following citations:

- `Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. Overview and Meteorological Validation of the Wind Integration National Dataset Toolkit (Technical Report, NREL/TP-5000-61740). Golden, CO: National Renewable Energy Laboratory. <https://www.nrel.gov/docs/fy15osti/61740.pdf>`_
- `Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. "The Wind Integration National Dataset (WIND) Toolkit." Applied Energy 151: 355366. <https://www.sciencedirect.com/science/article/pii/S0306261915004237?via%3Dihub>`_
- `Lieberman-Cribbin, W., C. Draxl, and A. Clifton. 2014. Guide to Using the WIND Toolkit Validation Code (Technical Report, NREL/TP-5000-62595). Golden, CO: National Renewable Energy Laboratory. <https://www.nrel.gov/docs/fy15osti/62595.pdf>`_
- `King, J., A. Clifton, and B.M. Hodge. 2014. Validation of Power Output for the WIND Toolkit (Technical Report, NREL/TP-5D00-61714). Golden, CO: National Renewable Energy Laboratory. <https://www.nrel.gov/docs/fy14osti/61714.pdf>`_
