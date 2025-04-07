Using ``xarray``
================

As of rex ``v0.2.99``, you can read `NREL data files <https://nrel.github.io/rex/misc/examples.nrel_data.html>`_
using the popular open-source library `xarray <https://docs.xarray.dev/en/stable/index.html>`_. You can learn
more about the benefits of using ``xarray`` `here <https://docs.xarray.dev/en/stable/getting-started-guide/why-xarray.html>`_.

Basic Usage
-----------

Opening a single file
^^^^^^^^^^^^^^^^^^^^^

To read in an NREL data file, simply supply ``engine="rex"`` to the
`open_dataset <https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html#xarray-open-dataset>`_
function:

.. code-block:: python

    import os
    import xarray as xr
    from rex import TESTDATADIR

    WTK_2012_FP = os.path.join(TESTDATADIR, 'wtk', 'ri_100_wtk_2012.h5')

    ds = xr.open_dataset(WTK_2012_FP, engine="rex")
    ds

.. code-block:: python-console

    <xarray.Dataset> Size: 32MB
    Dimensions:             (time: 8784, gid: 200)
    Coordinates:
        time_index          (time) datetime64[ns] 70kB ...
      * time                (time) datetime64[ns] 70kB 2012-01-01 ... 2012-12-31T...
      * gid                 (gid) int64 2kB 0 1 2 3 4 5 ... 194 195 196 197 198 199
        latitude            (gid) float32 800B ...
        longitude           (gid) float32 800B ...
        country             (gid) |S13 3kB ...
        state               (gid) |S4 800B ...
        county              (gid) |S22 4kB ...
        timezone            (gid) int16 400B ...
        elevation           (gid) int16 400B ...
        offshore            (gid) int16 400B ...
    Data variables:
        pressure_0m         (time, gid) uint16 4MB ...
        pressure_100m       (time, gid) uint16 4MB ...
        pressure_200m       (time, gid) uint16 4MB ...
        temperature_100m    (time, gid) int16 4MB ...
        temperature_80m     (time, gid) int16 4MB ...
        winddirection_100m  (time, gid) uint16 4MB ...
        winddirection_80m   (time, gid) uint16 4MB ...
        windspeed_100m      (time, gid) uint16 4MB ...
        windspeed_80m       (time, gid) uint16 4MB ...


Lazy Loading - ``rex``
^^^^^^^^^^^^^^^^^^^^^

Note that the operation above (reading a file using ``xr.open_dataset``)
did not immediately load arrays into memory. This is because the ``rex``
backend provides lazily-loaded arrays.

.. NOTE:: This is not ``dask``, since we did not specify the ``chunks=...`` parameter in ``xr.open_dataset``.

You can verify this for yourself by checking the private ``_data`` attribute:

.. code-block:: python

    print(ds["windspeed_100m"].variable._data)


.. code-block:: python-console

    MemoryCachedArray(array=CopyOnWriteArray(array=LazilyIndexedArray(array=<rex.external.rexarray.RexArrayWrapper object at 0x7fdea431dae0>, key=BasicIndexer((slice(None, None, None), slice(None, None, None))))))

You can see that we have a ``LazilyIndexedArray`` instance around the custom
``RexArrayWrapper`` object. Notably, there is no data shown.

You can get the data to load into memory like so:

.. code-block:: python

    print(ds["windspeed_100m"].data)


.. code-block:: python-console

    [[ 7.25  7.13  6.9  ...  8.7   8.66  8.45]
    [ 8.02  7.7   8.12 ...  6.02  5.98  6.51]
    [10.23  9.76  9.82 ...  7.15  7.51  7.69]
    ...
    [ 8.74  8.78  9.19 ... 11.97 12.17 12.43]
    [10.34 10.33 10.41 ... 12.87 12.9  13.  ]
    [10.34 10.43 10.74 ... 14.77 14.85 14.82]]

Now if we check the ``_data`` attribute again, we can see that the data values
have been loaded:

.. code-block:: python

    print(ds["windspeed_100m"].data)


.. code-block:: python-console

    MemoryCachedArray(array=NumpyIndexingAdapter(array=array([[ 7.25,  7.13,  6.9 , ...,  8.7 ,  8.66,  8.45],
        [ 8.02,  7.7 ,  8.12, ...,  6.02,  5.98,  6.51],
        [10.23,  9.76,  9.82, ...,  7.15,  7.51,  7.69],
        ...,
        [ 8.74,  8.78,  9.19, ..., 11.97, 12.17, 12.43],
        [10.34, 10.33, 10.41, ..., 12.87, 12.9 , 13.  ],
        [10.34, 10.43, 10.74, ..., 14.77, 14.85, 14.82]], shape=(8784, 200))))


Operations on these arrays are not lazy and *will* cause them to get loaded into memory:

.. code-block:: python

    print(ds["windspeed_80m"] * 2)


.. code-block:: python-console

    <xarray.DataArray 'windspeed_80m' (time: 8784, gid: 200)> Size: 14MB
    array([[12.96, 12.82, 12.46, ..., 17.2 , 17.16, 16.76],
        [14.2 , 13.58, 14.5 , ..., 11.82, 11.24, 11.78],
        [18.64, 17.32, 17.26, ..., 14.18, 14.96, 15.32],
        ...,
        [16.04, 16.2 , 17.02, ..., 23.8 , 24.22, 24.74],
        [18.92, 18.92, 18.9 , ..., 25.56, 25.64, 25.86],
        [18.9 , 19.18, 19.74, ..., 29.3 , 29.46, 29.42]], shape=(8784, 200))
    Coordinates:
        time_index  (time) datetime64[ns] 70kB ...
      * time        (time) datetime64[ns] 70kB 2012-01-01 ... 2012-12-31T23:00:00
      * gid         (gid) int64 2kB 0 1 2 3 4 5 6 7 ... 193 194 195 196 197 198 199
        latitude    (gid) float32 800B ...
        longitude   (gid) float32 800B ...
        country     (gid) |S13 3kB ...
        state       (gid) |S4 800B ...
        county      (gid) |S22 4kB ...
        timezone    (gid) int16 400B ...
        elevation   (gid) int16 400B ...
        offshore    (gid) int16 400B ...


Lazy Loading - ``dask``
^^^^^^^^^^^^^^^^^^^^^

We can also request that our data be read in lazily using `dask <https://www.dask.org/>`_.
The easiest way to do this is to provide a ``chunks=...`` parameter in ``xr.open_dataset``:


.. code-block:: python

    ds_dask = xr.open_dataset(WTK_2012_FP, engine="rex", chunks="auto")
    ds_dask

.. code-block:: python-console

    <xarray.Dataset> Size: 32MB
    Dimensions:             (time: 8784, gid: 200)
    Coordinates:
        time_index          (time) datetime64[ns] 70kB dask.array<chunksize=(8784,), meta=np.ndarray>
      * time                (time) datetime64[ns] 70kB 2012-01-01 ... 2012-12-31T...
      * gid                 (gid) int64 2kB 0 1 2 3 4 5 ... 194 195 196 197 198 199
        latitude            (gid) float32 800B dask.array<chunksize=(200,), meta=np.ndarray>
        longitude           (gid) float32 800B dask.array<chunksize=(200,), meta=np.ndarray>
        country             (gid) |S13 3kB dask.array<chunksize=(200,), meta=np.ndarray>
        state               (gid) |S4 800B dask.array<chunksize=(200,), meta=np.ndarray>
        county              (gid) |S22 4kB dask.array<chunksize=(200,), meta=np.ndarray>
        timezone            (gid) int16 400B dask.array<chunksize=(200,), meta=np.ndarray>
        elevation           (gid) int16 400B dask.array<chunksize=(200,), meta=np.ndarray>
        offshore            (gid) int16 400B dask.array<chunksize=(200,), meta=np.ndarray>
    Data variables:
        pressure_0m         (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        pressure_100m       (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        pressure_200m       (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        temperature_100m    (time, gid) int16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        temperature_80m     (time, gid) int16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        winddirection_100m  (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        winddirection_80m   (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        windspeed_100m      (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        windspeed_80m       (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>


We can immediately tell that dask is being used because the data are represented by dask arrays.
Operations on this dataset *are* lazy:

.. code-block:: python

    print(ds_dask["windspeed_100m"].mean())

.. code-block:: python-console

    <xarray.DataArray 'windspeed_100m' ()> Size: 8B
    dask.array<mean_agg-aggregate, shape=(), dtype=float64, chunksize=(), chunktype=numpy.ndarray>

We can see that no values has been given. To run the computation, we have to call the ``.compute()``
method:

.. code-block:: python

    print(ds_dask["windspeed_100m"].mean().compute())

.. code-block:: python-console

    <xarray.DataArray 'windspeed_100m' ()> Size: 8B
    array(7.65926428)

For more information on using dask with xarray, see `this <https://docs.xarray.dev/en/stable/user-guide/dask.html>`_ guide.


Opening Multiple Files
^^^^^^^^^^^^^^^^^^^^^

You can use ``xr.open_mfdataset`` to open multiple NREL data files at once:


.. code-block:: python

    import os
    import xarray as xr
    from rex import TESTDATADIR

    WTK_FPS = os.path.join(TESTDATADIR, 'wtk', 'ri_100_wtk_20*.h5')

    ds = xr.open_mfdataset(WTK_FPS, engine="rex")
    ds

.. code-block:: python-console

    <xarray.Dataset> Size: 63MB
    Dimensions:             (time: 17544, gid: 200)
    Coordinates:
        time_index          (time) datetime64[ns] 140kB dask.array<chunksize=(8784,), meta=np.ndarray>
      * time                (time) datetime64[ns] 140kB 2012-01-01 ... 2013-12-31...
      * gid                 (gid) int64 2kB 0 1 2 3 4 5 ... 194 195 196 197 198 199
        latitude            (gid) float32 800B dask.array<chunksize=(200,), meta=np.ndarray>
        longitude           (gid) float32 800B dask.array<chunksize=(200,), meta=np.ndarray>
        country             (gid) |S13 3kB dask.array<chunksize=(200,), meta=np.ndarray>
        state               (gid) |S4 800B dask.array<chunksize=(200,), meta=np.ndarray>
        county              (gid) |S22 4kB dask.array<chunksize=(200,), meta=np.ndarray>
        timezone            (gid) int16 400B dask.array<chunksize=(200,), meta=np.ndarray>
        elevation           (gid) int16 400B dask.array<chunksize=(200,), meta=np.ndarray>
        offshore            (gid) int16 400B dask.array<chunksize=(200,), meta=np.ndarray>
    Data variables:
        pressure_0m         (time, gid) uint16 7MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        pressure_100m       (time, gid) uint16 7MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        pressure_200m       (time, gid) uint16 7MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        temperature_100m    (time, gid) int16 7MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        temperature_80m     (time, gid) int16 7MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        winddirection_100m  (time, gid) uint16 7MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        winddirection_80m   (time, gid) uint16 7MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        windspeed_100m      (time, gid) uint16 7MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        windspeed_80m       (time, gid) uint16 7MB dask.array<chunksize=(8784, 200), meta=np.ndarray>


The shape of ``time`` indicates that two years of data have been loaded. You can also verify this directly:


.. code-block:: python

    print(ds.time[[0, -1]])

.. code-block:: python-console

    <xarray.DataArray 'time' (time: 2)> Size: 16B
    array(['2012-01-01T00:00:00.000000000', '2013-12-31T23:00:00.000000000'],
        dtype='datetime64[ns]')
    Coordinates:
        time_index  (time) datetime64[ns] 16B dask.array<chunksize=(2,), meta=np.ndarray>
      * time        (time) datetime64[ns] 16B 2012-01-01 2013-12-31T23:00:00
    Attributes:
        standard_name:  time
        long_name:      time
        calendar:       proleptic_gregorian
        time_zone:      UTC


Remote Files
------------

You can also use ``xarray`` to open remote files directly.

Files on S3
^^^^^^^^^^^

For files on S3, you do not need to provide any extra information:


.. code-block:: python

    import xarray as xr

    ds = xr.open_dataset("s3://nrel-pds-nsrdb/current/nsrdb_1998.h5", engine="rex")
    ds

.. code-block:: python-console

    <xarray.Dataset> Size: 2TB
    Dimensions:                   (time: 17520, gid: 2018267)
    Coordinates:
        time_index                (time) datetime64[ns] 140kB ...
      * time                      (time) datetime64[ns] 140kB 1998-01-01 ... 1998...
      * gid                       (gid) int64 16MB 0 1 2 ... 2018264 2018265 2018266
        latitude                  (gid) float32 8MB ...
        longitude                 (gid) float32 8MB ...
        elevation                 (gid) int16 4MB ...
        timezone                  (gid) int16 4MB ...
        country                   (gid) |S36 73MB ...
        state                     (gid) |S31 63MB ...
        county                    (gid) |S51 103MB ...
    Data variables: (12/26)
        air_temperature           (time, gid) int16 71GB ...
        alpha                     (time, gid) uint8 35GB ...
        aod                       (time, gid) uint16 71GB ...
        asymmetry                 (time, gid) int8 35GB ...
        cld_opd_dcomp             (time, gid) uint16 71GB ...
        cld_press_acha            (time, gid) uint16 71GB ...
        ...                        ...
        ssa                       (time, gid) uint8 35GB ...
        surface_albedo            (time, gid) uint8 35GB ...
        surface_pressure          (time, gid) uint16 71GB ...
        total_precipitable_water  (time, gid) uint8 35GB ...
        wind_direction            (time, gid) uint16 71GB ...
        wind_speed                (time, gid) uint16 71GB ...
    Attributes:
        version:  3.2.2


Just like before, the data is lazy-loaded, so reading in the file does not take too long.
However, once you start processing the data, it will need to be downloaded, which can be
time consuming.

Files on HSDS
^^^^^^^^^^^^^

A more performant option is to use HSDS (see
`this guide <https://nrel.github.io/rex/misc/examples.hsds.html#setting-up-a-local-hsds-server>`_
on setting up your own local hsds server):

.. code-block:: python

    import xarray as xr

    ds = xr.open_dataset("/nrel/nsrdb/v3/nsrdb_1998.h5", engine="rex", hsds=True)
    ds

.. code-block:: python-console

    <xarray.Dataset> Size: 2TB
    Dimensions:                   (time: 17520, gid: 2018392)
    Coordinates: (12/13)
        time_index                (time) datetime64[ns] 140kB ...
      * time                      (time) datetime64[ns] 140kB 1998-01-01 ... 1998...
      * gid                       (gid) int64 16MB 0 1 2 ... 2018389 2018390 2018391
        latitude                  (gid) float32 8MB ...
        longitude                 (gid) float32 8MB ...
        elevation                 (gid) float32 8MB ...
        ...                        ...
        country                   (gid) |S30 61MB ...
        state                     (gid) |S30 61MB ...
        county                    (gid) |S30 61MB ...
        urban                     (gid) |S30 61MB ...
        population                (gid) int32 8MB ...
        landcover                 (gid) int16 4MB ...
    Data variables: (12/25)
        air_temperature           (time, gid) int8 35GB ...
        alpha                     (time, gid) int16 71GB ...
        aod                       (time, gid) int16 71GB ...
        asymmetry                 (time, gid) int16 71GB ...
        cld_opd_dcomp             (time, gid) int16 71GB ...
        cld_reff_dcomp            (time, gid) int16 71GB ...
        ...                        ...
        ssa                       (time, gid) int16 71GB ...
        surface_albedo            (time, gid) int16 71GB ...
        surface_pressure          (time, gid) int16 71GB ...
        total_precipitable_water  (time, gid) int16 71GB ...
        wind_direction            (time, gid) int16 71GB ...
        wind_speed                (time, gid) int16 71GB ...
    Attributes:
        Version:  3.0.6


Opening Multiple Remote Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``xr.open_mfdataset`` does not support the wildcard (``*``) syntax for remote files,
so to open multiple files on S3, you have to list them out explicitly:


.. code-block:: python

    import xarray as xr

    files = [
        "s3://nrel-pds-nsrdb/current/nsrdb_1998.h5",
        "s3://nrel-pds-nsrdb/current/nsrdb_1999.h5",
    ]
    ds = xr.open_mfdataset(files, engine="rex")
    ds

.. code-block:: python-console

    <xarray.Dataset> Size: 3TB
    Dimensions:                   (time: 35040, gid: 2018267)
    Coordinates:
        time_index                (time) datetime64[ns] 280kB dask.array<chunksize=(17520,), meta=np.ndarray>
      * time                      (time) datetime64[ns] 280kB 1998-01-01 ... 1999...
      * gid                       (gid) int64 16MB 0 1 2 ... 2018264 2018265 2018266
        latitude                  (gid) float32 8MB dask.array<chunksize=(2018267,), meta=np.ndarray>
        longitude                 (gid) float32 8MB dask.array<chunksize=(2018267,), meta=np.ndarray>
        elevation                 (gid) int16 4MB dask.array<chunksize=(2018267,), meta=np.ndarray>
        timezone                  (gid) int16 4MB dask.array<chunksize=(2018267,), meta=np.ndarray>
        country                   (gid) |S36 73MB dask.array<chunksize=(2018267,), meta=np.ndarray>
        state                     (gid) |S31 63MB dask.array<chunksize=(2018267,), meta=np.ndarray>
        county                    (gid) |S51 103MB dask.array<chunksize=(2018267,), meta=np.ndarray>
    Data variables: (12/26)
        air_temperature           (time, gid) int16 141GB dask.array<chunksize=(2000, 500), meta=np.ndarray>
        alpha                     (time, gid) uint8 71GB dask.array<chunksize=(2000, 1000), meta=np.ndarray>
        aod                       (time, gid) uint16 141GB dask.array<chunksize=(2000, 500), meta=np.ndarray>
        asymmetry                 (time, gid) int8 71GB dask.array<chunksize=(2000, 1000), meta=np.ndarray>
        cld_opd_dcomp             (time, gid) uint16 141GB dask.array<chunksize=(2000, 500), meta=np.ndarray>
        cld_press_acha            (time, gid) uint16 141GB dask.array<chunksize=(2000, 500), meta=np.ndarray>
        ...                        ...
        ssa                       (time, gid) uint8 71GB dask.array<chunksize=(2000, 1000), meta=np.ndarray>
        surface_albedo            (time, gid) uint8 71GB dask.array<chunksize=(2000, 1000), meta=np.ndarray>
        surface_pressure          (time, gid) uint16 141GB dask.array<chunksize=(2000, 500), meta=np.ndarray>
        total_precipitable_water  (time, gid) uint8 71GB dask.array<chunksize=(2000, 1000), meta=np.ndarray>
        wind_direction            (time, gid) uint16 141GB dask.array<chunksize=(2000, 500), meta=np.ndarray>
        wind_speed                (time, gid) uint16 141GB dask.array<chunksize=(2000, 500), meta=np.ndarray>
    Attributes:
        version:  3.2.2


Due to technical limitations, you cannot use ``xr.open_mfdataset`` to open multiple files
via HSDS. Instead, you can use the ``rex.open_mfdataset_hsds`` function, which does
accept wildcard inputs:

.. code-block:: python

    import xarray as xr
    from rex import open_mfdataset_hsds

    ds = open_mfdataset_hsds("/nrel/nsrdb/v3/nsrdb_199*.h5")
    ds


Parallel Computing with ``dask``
-------------------------------

Although your computations are lazy if you load your data with ``dask``, they still only run on a single
process (or thread pool) by default (see `here <https://docs.dask.org/en/stable/scheduling.html#scheduling>`_
for more info on the ``dask`` schedulers). In order to quickly and easily parallelize your computations,
you can use `dask-distributed <https://distributed.dask.org/en/stable/>`_.

To start off, install the required library:

.. code-block:: bash

    $ pip install distributed --upgrade


Next, you should start a ``dask`` client that controls your parallelization scheme:

.. code-block:: python

    from dask.distributed import Client
    client = Client(n_workers=4, memory_limit='10GB')

In this example, we have told dask that we would like our computations to take up 4 cores and
a maximum of 10GB of memory. Once this client is running, you can write your data analysis code
as normal. Any ``dask`` computations you do will be performed in chunks using 4 processes:


.. code-block:: python

    import os
    import xarray as xr
    from rex import TESTDATADIR

    WTK_2012_FP = os.path.join(TESTDATADIR, 'wtk', 'ri_100_wtk_2012.h5')
    ds = xr.open_dataset(WTK_2012_FP, engine="rex", chunks="auto")
    ds["windspeed_100m"].mean(dim="time").compute()

.. code-block:: python-console

    <xarray.Dataset> Size: 32MB
    Dimensions:             (time: 8784, gid: 200)
    Coordinates:
        time_index          (time) datetime64[ns] 70kB dask.array<chunksize=(8784,), meta=np.ndarray>
      * time                (time) datetime64[ns] 70kB 2012-01-01 ... 2012-12-31T...
      * gid                 (gid) int64 2kB 0 1 2 3 4 5 ... 194 195 196 197 198 199
        latitude            (gid) float32 800B dask.array<chunksize=(200,), meta=np.ndarray>
        longitude           (gid) float32 800B dask.array<chunksize=(200,), meta=np.ndarray>
        country             (gid) |S13 3kB dask.array<chunksize=(200,), meta=np.ndarray>
        state               (gid) |S4 800B dask.array<chunksize=(200,), meta=np.ndarray>
        county              (gid) |S22 4kB dask.array<chunksize=(200,), meta=np.ndarray>
        timezone            (gid) int16 400B dask.array<chunksize=(200,), meta=np.ndarray>
        elevation           (gid) int16 400B dask.array<chunksize=(200,), meta=np.ndarray>
        offshore            (gid) int16 400B dask.array<chunksize=(200,), meta=np.ndarray>
    Data variables:
        pressure_0m         (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        pressure_100m       (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        pressure_200m       (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        temperature_100m    (time, gid) int16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        temperature_80m     (time, gid) int16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        winddirection_100m  (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        winddirection_80m   (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        windspeed_100m      (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>
        windspeed_80m       (time, gid) uint16 4MB dask.array<chunksize=(8784, 200), meta=np.ndarray>


Remember that in order for the computations to be distributed using ``dask``, you must
load your data into ``dask`` arrays. The easiest way to do this is to specify ``chunks=...``
when your read in the data (as we've done above).


Case Studies
------------
Once you have opened the file with ``xarray``, you cen leverage all the power of that library to
perform data analysis tasks. Check out some examples of this below:

