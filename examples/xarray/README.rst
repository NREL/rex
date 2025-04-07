Using Xarray
============

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
^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^

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


.. code-block:: python-console

    <xarray.Dataset> Size: 3TB
    Dimensions:                   (time: 35040, gid: 2018392)
    Coordinates: (12/13)
        time_index                (time) datetime64[ns] 280kB dask.array<chunksize=(17520,), meta=np.ndarray>
      * time                      (time) datetime64[ns] 280kB 1998-01-01 ... 1999...
      * gid                       (gid) int64 16MB 0 1 2 ... 2018389 2018390 2018391
        latitude                  (gid) float32 8MB dask.array<chunksize=(252299,), meta=np.ndarray>
        longitude                 (gid) float32 8MB dask.array<chunksize=(252299,), meta=np.ndarray>
        elevation                 (gid) float32 8MB dask.array<chunksize=(12856,), meta=np.ndarray>
        ...                        ...
        country                   (gid) |S30 61MB dask.array<chunksize=(12856,), meta=np.ndarray>
        state                     (gid) |S30 61MB dask.array<chunksize=(12856,), meta=np.ndarray>
        county                    (gid) |S30 61MB dask.array<chunksize=(12856,), meta=np.ndarray>
        urban                     (gid) |S30 61MB dask.array<chunksize=(12856,), meta=np.ndarray>
        population                (gid) int32 8MB dask.array<chunksize=(12856,), meta=np.ndarray>
        landcover                 (gid) int16 4MB dask.array<chunksize=(12856,), meta=np.ndarray>
    Data variables: (12/25)
        air_temperature           (time, gid) int8 71GB dask.array<chunksize=(2688, 744), meta=np.ndarray>
        alpha                     (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
        aod                       (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
        asymmetry                 (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
        cld_opd_dcomp             (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
        cld_reff_dcomp            (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
        ...                        ...
        ssa                       (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
        surface_albedo            (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
        surface_pressure          (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
        total_precipitable_water  (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
        wind_direction            (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
        wind_speed                (time, gid) int16 141GB dask.array<chunksize=(2688, 372), meta=np.ndarray>
    Attributes:
        Version:  3.0.6


The object returned by this function is a standard ``xarray.DataSet``, so you can plug it directly into
your analysis workflow.


Parallel Computing with ``dask``
--------------------------------

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

    <xarray.DataArray 'windspeed_100m' (gid: 200)> Size: 2kB
    array([6.87944558, 6.93480874, 6.99835383, 6.93864071, 6.94729167,
           7.10710155, 7.20108265, 7.16832536, 7.17782559, 7.32525729,
           7.31806466, 7.33072291, 7.28809312, 7.22210041, 7.26382286,
           7.22113616, 7.07780852, 7.08420082, 7.06300888, 7.29566712,
           7.39969945, 7.45076844, 7.43958447, 7.43899704, 7.46535405,
           7.42845401, 7.29795651, 7.19493852, 7.05814663, 6.88442168,
           6.85640938, 6.85859176, 7.05063525, 7.09214253, 7.14200478,
           7.23776412, 7.31980874, 7.27634107, 7.23299977, 7.30763434,
           7.36336179, 7.29896175, 7.14903347, 7.01208447, 6.89744877,
           6.75741234, 6.70825137, 6.7998816 , 6.95533584, 6.86520833,
           6.89120105, 6.97641507, 7.03485883, 7.18894353, 7.24894923,
           7.16365437, 6.94202641, 6.83164959, 7.01116462, 7.16896744,
           7.12047473, 6.9952163 , 6.87737705, 6.76693875, 6.66649021,
           6.64727345, 6.77149704, 6.85299522, 7.09365551, 6.93533356,
           6.93294627, 7.01532673, 7.03204235, 7.06432377, 7.13546903,
           7.18949567, 7.31379098, 7.25667464, 7.01673725, 6.76937158,
           6.77199567, 7.02523338, 7.18999203, 7.13933629, 7.02185109,
           6.8873224 , 6.74311362, 6.63313183, 6.56818306, 6.56203097,
           6.55499772, 7.82230419, 7.6846209 , 7.01713912, 7.38534608,
           7.27920993, 7.14663138, 7.10545879, 7.03550091, 6.89493739,
           6.97185109, 7.14635246, 7.99603256, 7.87955373, 8.10967896,
           8.04715392, 7.94161544, 7.81445811, 8.21366576, 8.16868056,
           8.10710155, 8.00840392, 7.90002732, 7.6702015 , 8.24766735,
           8.20477117, 8.14225638, 8.05135246, 7.95503985, 7.78786544,
           8.31860997, 8.29575364, 8.25664276, 8.19562955, 8.11789617,
           8.02574112, 7.91357013, 8.36183402, 8.35696949, 8.3436612 ,
           8.32018784, 8.28632969, 8.23200137, 8.17405738, 8.09175319,
           8.00542919, 7.86773224, 8.4832434 , 8.4748133 , 8.44929986,
           8.42680328, 8.40229508, 8.38972336, 8.37218807, 8.3485633 ,
           8.32098588, 8.27343579, 8.23153119, 8.16400159, 8.08327641,
           7.98567851, 7.80577186, 8.50659722, 8.50146403, 8.50739982,
           8.51652436, 8.51465733, 8.51314094, 8.49275273, 8.47009904,
           8.44045765, 8.42123406, 8.39814663, 8.37284381, 8.34804303,
           8.30322746, 8.27097791, 8.21462773, 8.13777778, 8.05820355,
           7.94375455, 7.79726207, 8.52080829, 8.52497837, 8.52421107,
           8.53359403, 8.53315574, 8.53251138, 8.53214367, 8.53220856,
           8.53655624, 8.52449681, 8.50425888, 8.4733857 , 8.45109517,
           8.42566826, 8.39944331, 8.37924408, 8.33734403, 8.31360087,
           8.26787568, 8.20503074, 8.13375228, 8.04971881, 7.95212659,
           8.56010474, 8.55197177, 8.54179417, 8.54154599, 8.53991234])
    Coordinates:
      * gid        (gid) int64 2kB 0 1 2 3 4 5 6 7 ... 193 194 195 196 197 198 199
        latitude   (gid) float32 800B 41.96 41.98 42.0 41.9 ... 40.91 40.93 40.95
        longitude  (gid) float32 800B -71.79 -71.79 -71.78 ... -71.79 -71.78 -71.78
        country    (gid) |S13 3kB b'United States' b'United States' ... b'None'
        state      (gid) |S4 800B b'RI' b'RI' b'RI' ... b'None' b'None' b'None'
        county     (gid) |S22 4kB b'Providence' b'Providence' ... b'None' b'None'
        timezone   (gid) int16 400B -5 -5 -5 -5 -5 -5 -5 -5 ... -5 -5 -5 -5 -5 -5 -5
        elevation  (gid) int16 400B 202 178 174 195 201 170 187 ... 0 0 0 0 0 0 0
        offshore   (gid) int16 400B 0 0 0 0 0 0 0 0 0 0 0 ... 1 1 1 1 1 1 1 1 1 1 1


Remember that in order for the computations to be distributed using ``dask``, you must
load your data into ``dask`` arrays. The easiest way to do this is to specify ``chunks=...``
when your read in the data (as we've done above).


Case Studies
------------
Once you have opened the file with ``xarray``, you can take full advantage of the library's
powerful features to perform data analysis tasks. Check out some examples of this below:

- `Daily Aggregations using Xarray <https://github.com/NREL/rex/blob/main/examples/xarray/daily_agg.ipynb>`_
