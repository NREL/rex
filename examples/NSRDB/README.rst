National Solar Radiation Database (NSRDB)
=========================================

This page describes some of the unique attributes of the NSRDB. For
instructions on how to access the data, see the docs page `here
<https://nrel.github.io/rex/misc/examples.nrel_data.html>`_.

Description
-----------

The National Solar Radiation Database (NSRDB) is a serially complete
collection of meteorological and solar irradiance data sets for the
United States and a growing list of international locations for 1998-2017. The
NSRDB provides foundational information to support U.S. Department of Energy
programs, research, and the general public.

The NSRDB provides time-series data at 30 minute resolution of resource
averaged over surface cells of 0.038 degrees in both latitude and longitude,
or nominally 4 km in size. The solar radiation values represent the resource
available to solar energy systems. The data was created using cloud properties
which are generated using the AVHRR Pathfinder Atmospheres-Extended (PATMOS-x)
algorithms developed by the University of Wisconsin. Fast all-sky radiation
model for solar applications (FARMS) in conjunction with the cloud properties,
and aerosol optical depth (AOD) and precipitable water vapor (PWV) from
ancillary source are used to estimate solar irradiance (GHI, DNI, and DHI).
The Global Horizontal Irradiance (GHI) is computed for clear skies using the
REST2 model. For cloud scenes identified by the cloud mask, FARMS is used to
compute GHI. The Direct Normal Irradiance (DNI) for cloud scenes is then
computed using the DISC model. The PATMOS-X model uses half-hourly radiance
images in visible and infrared channels from the GOES series of geostationary
weather satellites.  Ancillary variables needed to run REST2 and FARMS (e.g.,
aerosol optical depth, precipitable water vapor, and albedo) are derived from
the the Modern Era-Retrospective Analysis (MERRA-2) dataset. Temperature and
wind speed data are also derived from MERRA-2 and provided for use in SAM to
compute PV generation.

Note that this guidance applies to the original NSRDB data in the western
hemisphere and also all of the new international data derived from the Meteosat
and Himawari satellites.

The following variables are provided by the NSRDB:

- Irradiance:

    - Global Horizontal (ghi)
    - Direct Normal (dni)
    - Diffuse (dhi)

- Clear-sky Irradiance
- Cloud Type
- Dew Point
- Temperature
- Surface Albedo
- Pressure
- Relative Humidity
- Solar Zenith Angle
- Precipitable Water
- Wind Direction
- Wind Speed
- Fill Flag
- Angstrom wavelength exponent (alpha)
- Aerosol optical depth (aod)
- Aerosol asymmetry parameter (asymmetry)
- Cloud optical depth (cld_opd_dcomp)
- Cloud effective radius (cld_ref_dcomp)
- cloud_press_acha
- Reduced ozone vertical pathlength (ozone)
- Aerosol single-scatter albedo (ssa)


NSRDB CLI
---------

The `NSRDBX <https://nrel.github.io/rex/rex/rex.resource_extaction.nsrdb_cli.html#nsrdbx>`_
command line utility provides the following options and commands:

.. code-block:: bash

  NSRDBX --help

  Usage: NSRDBX [OPTIONS] COMMAND [ARGS]...

    NSRDBX Command Line Interface

  Options:
    -h5, --solar_h5 PATH  Path to Resource .h5 file  [required]
    -o, --out_dir PATH    Directory to dump output files  [required]
    -v, --verbose         Flag to turn on debug logging. Default is not verbose.
    --help                Show this message and exit.

  Commands:
    dataset     Extract a single dataset
    multi-site  Extract multiple sites given in '--sites' .csv or .json as...
    sam-file    Extract all datasets needed for SAM for the nearest pixel to...


References
----------

For more information about the NSRDB please see the `website <https://nsrdb.nrel.gov/>`_
Users of the NSRDB should please cite:

- `Sengupta, M., Y. Xie, A. Lopez, A. Habte, G. Maclaurin, and J. Shelby. 2018. "The National Solar Radiation Data Base (NSRDB)." Renewable and Sustainable Energy Reviews  89 (June): 51-60. <https://www.sciencedirect.com/science/article/pii/S136403211830087X?via%3Dihub>`_
