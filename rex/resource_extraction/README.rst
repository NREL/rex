***
Resource Extraction Tools
***

This sub-package of `rex` contains wrappers on the Resource handlers that
further facilitate the extraction of resource data for analysts. Namely
extracting:

- The nearest site to a desired set of coordinates (latitude, longitude)
- All sites in a given region (meta data field, i.e., state, county, country)
- SAM variables and formatting of SAM .csv files
- Maps for a given timestep and region


Command Line Interfaces (CLIs)
==============================

They extraction handlers are also available from the command line:

- `rex <https://nrel.github.io/rex/rex/rex.resource_extraction.resource_cli.html#rex>`_
- `NSRDBX <https://nrel.github.io/rex/rex/rex.resource_extraction.nsrdb_cli.html#nsrdbx>`_
- `WINDX <https://nrel.github.io/rex/rex/rex.resource_extraction.wind_cli.html#windx>`_
- `MultiYearX <https://nrel.github.io/rex/rex/rex.resource_extraction.multi_year_resource_cli.html#multiyearx>`_
