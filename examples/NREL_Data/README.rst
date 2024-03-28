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
 - ``scale_factor`` - We frequently scale data by a multiplicative factor, round the data to integer precision, and store the data in integer arrays. The ``scale_factor`` is an attribute associated with the relevant h5 ``dataset`` that defines the multiplicative factor required to unscale the data from integer storage to the original physical units.
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
NREL HPC file paths in ``/datasets/`` and set ``hsds=False``.

Data Location - External Users
------------------------------

If you are not at NREL, the easiest way to access this data is via HSDS. These
files are massive and downloading the full files would crash your computer.
HSDS provides a solution to stream small chunks of the data to your laptop or
server for just the time or space domain you're interested in.

See `this docs page <https://nrel.github.io/rex/misc/examples.hsds.html>`_ for
instructions on how to set up HSDS and then continue on to the Data Access
Examples section below.

To find relevant HSDS files, you can use HSDS and h5pyd to explore the NREL
public data directory listings. For example, if you are running an HSDS local
server, you can use the CLI utility ``hsls``, for example, run: ``$ hsls
/nrel/`` or ``$ hsls /nrel/nsrdb/v3/``. You can also use h5pyd to do the same
thing. In a python kernel, ``import h5pyd`` and then run:
``print(list(h5pyd.Folder('/nrel/')))`` to list the ``/nrel/`` directory.

The `Open Energy Data Initiative (OEDI) <https://openei.org/wiki/Main_Page>`_
is also invaluable in finding energy-relevant public datasets that are not
necessarily spatiotemporal h5 data.

We have also experimented with external data access using `fsspec <https://nrel.github.io/rex/misc/examples.fsspec.html>`_ and `zarr <https://nrel.github.io/rex/misc/examples.zarr.html>`_, but the examples below may not work with these utilities.

Data Access Examples
--------------------

If you are on the NREL HPC, update the file paths in the examples below and set
``hsds=False``.

If you are not at NREL, see the "Data Location - External Users" section above
for how to setup HSDS and how to find the files that you're interested in. Then
update the file paths to the files you want and keep ``hsds=True``.

The rex Resource Class
++++++++++++++++++++++

Data access in rex is built on the ``Resource`` class. The class can be used to
open and explore NREL h5 files, extract and automatically unscale data, and
retrieve ``time_index`` and ``meta`` datasets in their native pandas datatypes.

.. code-block:: python

    from rex import Resource
    with Resource('/nrel/nsrdb/current/nsrdb_2020.h5', hsds=True) as res:
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
    with WindResource('/nrel/wtk/conus/wtk_conus_2007.h5', hsds=True) as res:
        ws88 = res['windspeed_88m', :, 1000]
        print(res.dsets)
        print(ws88)

Here, notice that ``windspeed_88m`` is not a ``dataset`` available in the WIND
Toolkit file, but it can be requested by the ``WindResource`` class, which
interpolates the windspeeds between the available 80 and 100 meter hub heights.

The rex Resource Extraction Class
+++++++++++++++++++++++++++++++++

There are also classes that implement additional quality-of-life features. For
example, you can use the ``ResourceX`` class to retrieve a timeseries DataFrame
for a requested coordinate:

.. code-block:: python

    from rex import ResourceX
    with ResourceX('/nrel/wtk/conus/wtk_conus_2007.h5', hsds=True) as res:
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
    with SolarX('/nrel/nsrdb/current/nsrdb_2020.h5', hsds=True) as res:
        df = res.get_SAM_lat_lon((39.7407, -105.1686))
        print(df)

For a full list of ``ResourceX`` classes with additional features specific to
various renewable energy resource types, see the docs `here
<https://nrel.github.io/rex/_autosummary/rex.resource_extraction.resource_extraction.html>`_.
