NREL Renewable Energy Resource Data
===================================

Welcome to the docs page for NREL's renewable energy resource datasets! These
docs apply to all of the NREL spatiotemporal meteorological datasets stored in
HDF5 files including data for solar, wind, wave, and temperature variables. For
example, these docs apply to these NREL data products (not an exhaustive
list!):

 - The National Solar Radiation Database (NSRDB)
 - The Wind Integration National Dataset Toolkit (WIND Toolkit)
 - Other wind data stored in the WIND Toolkit AWS bucket (e.g., NOW-23, PR-100, Sup3rWind, international wind data, etc...)
 - High-resolution downscaled climate change data (Sup3rCC)
 - High Resolution Ocean Surface Wave Hindcast (US Wave) Data
 - Other spatiotemporal meteorological data from NREL!

Definitions
-----------

 - ``attributes`` - Meta data associated with an NREL h5 file or a dataset within that h5 file. This can be information about how the file was created, the software versions used to create the data, physical units of datasets, scale factors for compressed integer storage, or something else. ``attributes`` are stored in namespaces similar to python dictionaries for every h5 file and every dataset in every h5 file. This is not typically spatial meta data and is not related to the ``meta`` dataset. For more information, see the `h5py attributes docs <https://docs.h5py.org/en/stable/high/attr.html>`_.
 - ``chunks`` - Data arrays in an h5 dataset are stored in ``chunks`` which are subsets of the data array stored sequentially on disk. When reading an h5 file, you only have to read one chunk of data at a time, so if a file has a 1TB dataset with shape (8760, N) but the chunk shape is (8760, 100), you don't have to read the full 1TB of data to access a single ``gid``, you only have to read the single chunk of data (in this case a 8760x100 array). For more details, see the `h5py chunks docs <https://docs.h5py.org/en/stable/high/dataset.html?#chunked-storage>`_.
 - ``CLI`` - Command Line Interface (CLI). A program you can run from a command line call in a shell e.g., ``hsds``, ``hsls``, etc...
 - ``datasets`` - Named arrays (e.g., "windspeed_100m", "ghi", "temperature_2m", etc...) stored in an h5 file. These are frequently 2D arrays with dimensions (time, space) and can be sliced with a ``[idy, idx]`` syntax. See the `h5py dataset docs <https://docs.h5py.org/en/stable/high/dataset.html>`_ for details. We also refer to all our NREL data products as "datasets" so sorry for the confusion!
 - ``gid`` - We commonly refer to locations in a spatiotemporal NREL dataset by the location's ``gid`` which is the spatial index of the location of interest (zero-indexed). For example, in a 2D dataset with shape (time, space), ``gid=99`` (zero-indexed) would be the 100th column (1-indexed) in the 2D array.
 - ``h5`` - File extension for the heirarchical data format (e.g., "HDF5") that is widely used for spatiotemporal data at NREL. See the `h5py <https://docs.h5py.org/en/stable/>`_ library for more details.
 - ``h5pyd`` - The python library that provides the HDF REST interface to NREL data hosted on the cloud. This allows for the public to access small parts of large cloud-hosted datasets. See the `h5pyd <https://github.com/HDFGroup/h5pyd>`_ library for more details.
 - ``hsds`` - The highly scalable data service (HSDS) that we recommend to access small chunks of very large cloud-hosted NREL datasets. See the `hsds <https://github.com/HDFGroup/hsds>`_ library for more details.
 - ``meta`` - The ``dataset`` in an NREL h5 file that contains information about the spatial axis. This is typically a `pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ with columns such as "latitude", "longitude", "state", etc... The DataFrame is typically converted to a records array for storage in an h5 ``dataset``. The length of the meta data should match the length of axis 1 of a 2D spatiotemporal ``dataset``.
 - ``S3`` - Amazon Simple Storage Service (S3) is a basic cloud file storage system we use to store raw .h5 files in their full volume. Downloading files directly from S3 may not be the easiest way to access the data because each file tends to be multiple terabytes. Instead, you can stream small chunks of the files via HSDS.
 - ``scale_factor`` - We frequently scale data by a multiplicative factor, round the data to integer precision, and store the data in integer arrays. The ``scale_factor`` is an attribute associated with the relevant h5 ``dataset`` that defines the factor required to unscale the data from integer storage to the original physical units. The data should be divided by the ``scale_factor`` to scale back from integer to physical units.
 - ``time_index`` - The ``dataset`` in an NREL h5 file that contains information about the temporal axis. This is typically a `pandas DatetimeIndex <https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html>`_ that has been converted to a string array for storage in an h5 ``dataset``. The length of this ``dataset`` should match the length of axis 0 of a 2D spatiotemporal ``dataset``.

Data Format
-----------

NREL data is frequently provided in heirarchical data format (HDF5 or .h5).
Each file contains many datasets, with each ``dataset`` representing a physical
variable or meta data. Datasets are commonly 2 dimensional time-series arrays
with dimensions (time, space). The temporal axis is defined by ``time_index``,
while the spatial axis is defined by ``meta``. For storage efficiency, we
commonly scale each ``dataset`` by a multiplicative factor and store as an
integer. The scale_factor is provided in the ``scale_factor`` attribute. The
units for each variable are also commonly provided as an attribute called
``units``.


Many NREL tools have been developed based on the assumption that the data format
will follow a pseudo-standard definition. Data creators can adhere to the
following specifications for seamless integration into
`SAM <https://sam.nrel.gov>`_,
`reV <https://www.nrel.gov/gis/renewable-energy-potential>`_,
`rex <https://github.com/NREL/rex/blob/main/README.rst>`_,
and the data download APIs for
`solar <https://developer.nrel.gov/docs/solar/nsrdb/>`_,
`wind <https://developer.nrel.gov/docs/wind/wind-toolkit/>`_,
`wave <https://developer.nrel.gov/docs/wave/>`_, and
`climate <https://developer.nrel.gov/docs/climate/ncdb/>`_ data, to name a few.

- The domain is defined at the data product level. The domain naming convention
  is defined in a product agnostic way and follows the format:
  /``nrel``/``resource type``/``product name``/``optional data subclass(es)``/``data version``/``product_name``\ \_\ ``year.h5``.

    - Common subclasses include spatial regions, satellite variants, and
      underlying model variations. E.g. ``/nrel/nsrdb/GOES/conus/v4.0.0/``
- Each data product/domain will be represented by a single endpoint in the
  download APIs. E.g. the domain above is made accessible via
  `https://developer.nrel.gov/docs/solar/nsrdb/nsrdb-GOES-conus-v4-0-0-download <https://developer.nrel.gov/docs/solar/nsrdb/nsrdb-GOES-conus-v4-0-0-download/>`__
- Each data product/domain contains a set of HDF5 files, each holding a single
  year of data. - E.g.

+---------------------------------------------------+
| /nrel/nsrdb/GOES/conus/v4.0.0/                    |
+===================================================+
| /nrel/nsrdb/GOES/conus/v4.0.0/nsrdb_conus_2018.h5 |
+---------------------------------------------------+
| /nrel/nsrdb/GOES/conus/v4.0.0/nsrdb_conus_2019.h5 |
+---------------------------------------------------+
| /nrel/nsrdb/GOES/conus/v4.0.0/nsrdb_conus_2020.h5 |
+---------------------------------------------------+
| /nrel/nsrdb/GOES/conus/v4.0.0/nsrdb_conus_2021.h5 |
+---------------------------------------------------+
| /nrel/nsrdb/GOES/conus/v4.0.0/nsrdb_conus_2022.h5 |
+---------------------------------------------------+

- Each file has a top level attribute named ``version`` that defines the model version. In the example above this attribute would have a value of ``v4.0.0``. This is the version that is included in the files users
  will download.
- Each file includes 2 specific datasets that are tied to The API functionality. They are:

  - ``meta`` which contains a table of location specific metadata for each pixel/point/grid-cell of data. There are a few required
    metadata values, and no limit to additional metadata that can be
    included as deemed useful for the specific model. The required
    values are

    - latitude - either of the actual point or the centroid
    - longitude - either of the actual point or the centroid
    - timezone

      - The ``meta`` dataset will be 1 a dimensional array that must
        have the same length as the spatial dimension of the datasets
  - ``time_index`` which contains UTC timestamps in ISO 8601
    (``2022-01-01 00:45:00+00:00``) defining the temporal value of each
    data step.

    - There is a special use case for TMY data where ``time_index`` is not
      mandatory and may be superceded by ``tmy_year``. In this case the month,
      day, hour, and minute will be inferred from the data array position and
      the year will be supplied from this ``tmy_year`` dataset

        - The ``time_index`` dataset will be a 1 dimensional array that must
          have the same length as the temporal dimension of the datasets.

- Each file includes any number of datasets that include the model
  outputs. Each dataset is a variable e.g. “wind_speed” or “ghi”.

  - Each of these datasets will be a 2 dimensional array with (time, space)
    dimensions
  - The number of locations must be identical within every yearly file of the
    same domain
  - The number of timesteps must represent a consistent temporal
    resolution (hourly, 5 minute, etc), though there can be variations
    in size for leap years
  - Each of these datasets can optionally include any number of attributes. The
    following attribtues are used by the APIs to determine output formatting.

    - *scale_factor*: In cases where floats have been converted to
      integers for storage efficiency the *scale_factor* is the value to
      divide the raw data by to restore the original value. E.g.
      ``h5_val / scale_factor = actual_val``
    - *fill_value*: The value that is used to represent NULL. *\* NOTE
      that empty values in an HDF5 file are technically allowable but will
      result in errors during data post-processing by NREL tools, hence it is
      important to include a fill value in datasets where NULLs are possible*
    - *units*: The string representation of the units that apply to the
      raw data. Only necessary when you want values in Kelvin to be
      converted to Celcius for output. Otherwise included in output file
      metadata for the benefit of users

Notes on Chunking
~~~~~~~~~~~~~~~~~

Chunking is vital to achieving the best performance when reading the
data out of H5. Chunking divides up a potentially massive file into
discreet blocks on disc. When stored to disc HDF5 stores these chunks in
physically adjacent blocks for best I/O performance. In the cloud each
chunk is a discreet object. A bad chunking strategy on a typical data
product can create enough I/O latency to render the data absolutely
unusable by our tools due to excessive thread locking, network timeouts,
memory failures, etc.

The strategy is to identify the use cases for reading the data out of
H5, and then assign chunks that are aligned with that strategy. In
addition it has been observed that tiny chunks will cause slow read time
due to the I/O latency of reading many little chunks, while huge chunks
will cause slow read time and excessive memory usage because the process
is going to have to load a huge amount of data into memory in order to
slice out the small percentage that is relevant to the user. The worst
possible case would be to chunk the data opposite the manner that it
will be fetched. For example, if we imaging the data as a table, and
users typically fetch the data one column at a time, but you chunk the
data row-wise, then a single columnar fetch will require loading ALL of
the chunks. Imagine this same table however if the data is chunked
column-wise and also fetched column-wise the fetch will read only the
necessary data off disc thus minimizing I/O and maximizing performance.

For example, the APIs always read out a single pixel of data for an entire year
at a time, the obvious chunking strategy would appear to be to chunk [8760: 1]
(for hourly data with 8760 time steps). However this actually results in too
many small chunks and is not the ideal solution. Through testing we have found
that the ideal chunk for a year of hourly data is [8760: 1000]

There have been instances where a chunking strategy had to be adopted
that was a compromise between non-complementary use cases. One example
is where some users wanted to read data in spatially adjacent blocks for
time spans of only a week or a month at a time. This contradicts The API
use case of always reading out a year of data for a single pixel at a
time. The solution was to identify a chunk size that was a satisfacory
compromise for both. We came up with [2190: 1000]. Essentially this
breaks up the year into 4 quarters, keeping the 1000 adjacent locations
in each chunk.

A secondary note about chunking is that it makes logical sense for the
chunks to be a factor of the whole. E.g. if hourly data has 8760 time
steps then good chunk options would inclue 8760, (8760/2)=4380, or
(8760/4)=2190.

Common Data Formatting Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These are some common data formatting errors that we see when integrating new
data products into our tools. These are not limitations of the HDF5 format, but
rather are conventions NREL tools have adopted.

- NaNs in H5s
- Array sizes between different years of the same resource and/or different
  datasets within the same file that don't align on the *x* or *y* axis. Most
  often this is caused by something like the first year missing the first month
  due to the details of when the underlying source data began to be collected
  (not every weather station goes online at midnight on January 1st UTC time).
  *NOTE: time_index length will vary in leap years*
- Files that contain less than a year of data. Yearly data files are large, and
  it is often more convenient to work in monthly batches. However for final
  upload yearly is required.
- Scale factors that are applied by multiplication instead of division. We can
  handle this in code, just let us know how your scale factor is intended to be
  used.
- Confusion about units. If the datasets don't include the units attribute,
  please provide documentation.
- Excessive precision. Data at levels of precision beyond the model's accuracy,
  or margin of error can dramatically bloat the size of the data products and
  are very damaging to performance at every stage of the process.
- Extra dimensions in data. A good example is wind speed at multiple elevations.
  These values could easily be represented as a 3D array, however the supported
  way would be to create multiple datasets for each variable at each elevation.
  E.g. *wind_speed_40m*, *wind_speed_60M*, *wind_speed_80m*, etc.

Data Location - NREL Users
--------------------------

If you are at NREL, the easiest way to access this data is on the NREL
high-performance computing system (HPC). Go to the `NREL HPC website
<https://www.nrel.gov/hpc/>`_ and request access via an NREL project with an
HPC allocation. Once you are on the HPC, you can find that datasets in the
``/datasets/`` directory (e.g., run the linux command ``$ ls /datasets/``). Go
through the directory tree until you find the .h5 files you are looking for.
This datasets directory should not be confused with a ``dataset`` from an h5
file.

When using the ``rex`` examples below, update the file paths with the relevant
NREL HPC file paths in ``/datasets/``.

Data Location - External Users
------------------------------

If you are not at NREL, you can't just download these files. They are massive
and downloading the full files would crash your computer. The easiest way to
access this data is probably with ``fsspec``, which allows you to access files
directly on S3 with only one additional installation and no server setup.
However, this method is slow. The most performant method is via ``HSDS``.
``HSDS`` provides a solution to stream small chunks of the data to your laptop
or server for just the time or space domain you're interested in.

See `this docs page <https://nrel.github.io/rex/misc/examples.fsspec.html>`_
for easy (but slow) access of the source .h5 files on s3 with ``fsspec`` that
requires basically zero setup. To find relevant S3 files, you can explore the
S3 directory structure on `OEDI <https://openei.org/wiki/Main_Page>`_ or
with the `AWS CLI <https://aws.amazon.com/cli/>`_

See `this docs page <https://nrel.github.io/rex/misc/examples.hsds.html>`_ for
instructions on how to set up HSDS for more performant data access that
requires a bit of setup. To find relevant HSDS files, you can use HSDS and
h5pyd to explore the NREL public data directory listings. For example, if you
are running an HSDS local server, you can use the CLI utility ``hsls``, for
example, run: ``$ hsls /nrel/`` or ``$ hsls /nrel/nsrdb/v3/``. You can also use
h5pyd to do the same thing. In a python kernel, ``import h5pyd`` and then run:
``print(list(h5pyd.Folder('/nrel/')))`` to list the ``/nrel/`` directory.

There is also an experiment with using `zarr
<https://nrel.github.io/rex/misc/examples.zarr.html>`_, but the examples below
may not work with these utilities and the zarr example is not regularly tested.

The `Open Energy Data Initiative (OEDI) <https://openei.org/wiki/Main_Page>`_
is also invaluable for finding the source s3 filepaths and for finding
energy-relevant public datasets that are not necessarily spatiotemporal h5
data.


Data Access Examples
--------------------

If you are on the NREL HPC, update the file paths with the relevant NREL HPC
file paths in ``/datasets/``.

If you are not at NREL, see the "Data Location - External Users" section above
for S3 instructions or for how to setup HSDS and how to find the files that
you're interested in. Then update the file paths to the files you want either
on HSDS or S3.


The rex Resource Class
~~~~~~~~~~~~~~~~~~~~~~

Data access in rex is built on the ``Resource`` class. The class can be used to
open and explore NREL h5 files, extract and automatically unscale data, and
retrieve ``time_index`` and ``meta`` datasets in their native pandas datatypes.

.. code-block:: python

    from rex import Resource
    with Resource('/nrel/nsrdb/current/nsrdb_2020.h5') as res:
        ghi = res['ghi', :, 500]
        print(res.dsets)
        print(res.attrs['ghi'])
        print(res.time_index)
        print(res.meta)
        print(ghi)

Here, we are retrieving the ``ghi`` dataset for all time indices (axis=0) for
``gid`` 500 and also printing other useful meta data.

For a full description the ``Resource`` class API see the docs `here
<https://nrel.github.io/rex/_autosummary/rex.resource.Resource.html>`_.

There are also special ``Resource`` subclasses for many of the renewable energy
resource types. For a list of these classes and their corresponding
documentation, see the docs page `here
<https://nrel.github.io/rex/_autosummary/rex.renewable_resource.html>`_. For
example, the ``WindResource`` class can be used to open files in the WIND
Toolkit bucket (including datasets like NOW-23 and Sup3rWind) and will
interpolate windspeeds to the desired hub height, even if the requested
windspeed is not available as a ``dataset``:

.. code-block:: python

    from rex import WindResource
    with WindResource('/nrel/wtk/conus/wtk_conus_2007.h5') as res:
        ws88 = res['windspeed_88m', :, 1000]
        print(res.dsets)
        print(ws88)

Here, notice that ``windspeed_88m`` is not a ``dataset`` available in the WIND
Toolkit file, but it can be requested by the ``WindResource`` class, which
interpolates the windspeeds between the available 80 and 100 meter hub heights.

The rex Resource Extraction Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are also classes that implement additional quality-of-life features. For
example, you can use the ``ResourceX`` class to retrieve a timeseries DataFrame
for a requested coordinate:

.. code-block:: python

    from rex import ResourceX
    with ResourceX('/nrel/wtk/conus/wtk_conus_2007.h5') as res:
        df = res.get_lat_lon_df('temperature_2m', (39.7407, -105.1686))
        print(df)

Note that in this example, the ``ResourceX`` object first has to download the
full ``meta`` data, build a ``KDTree``, then query the tree. This takes a lot
of time for a single coordinate query. If you are querying multiple
coordinates, take a look at other methods like `ResourceX.lat_lon_gid
<https://nrel.github.io/rex/_autosummary/rex.resource_extraction.resource_extraction.ResourceX.html#rex.resource_extraction.resource_extraction.ResourceX.lat_lon_gid>`_
that get the ``gid`` for multiple coordinates at once. Also consider saving the
``gid`` indices you are interested in and reusing them instead of querying
these methods repeatedly.

You can also use a ``ResourceX`` class specific to a given resource type (e.g.,
wind or solar) to retrieve a DataFrame with all variables you will need to run
the System Advisor Model (SAM). For example, try:

.. code-block:: python

    from rex import SolarX
    with SolarX('/nrel/nsrdb/current/nsrdb_2020.h5') as res:
        df = res.get_SAM_lat_lon((39.7407, -105.1686))
        print(df)

For a full list of ``ResourceX`` classes with additional features specific to
various renewable energy resource types, see the docs `here
<https://nrel.github.io/rex/_autosummary/rex.resource_extraction.resource_extraction.html>`_.

Using rex with xarray
~~~~~~~~~~~~~~~~~~~~~

You can now use ``rex`` with ``xarray`` to open NREL datasets on the NREL HPC
and remotely outside of NREL! See the guide `here
<https://nrel.github.io/rex/misc/examples.xarray.html>`_ for details.
