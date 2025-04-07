Using ``xarray``
================

As of rex ``v0.2.99``, you can read `NREL data files <https://nrel.github.io/rex/misc/examples.nrel_data.html>`_
using the popular open-source library `xarray <https://docs.xarray.dev/en/stable/index.html>`_. You can learn
more about the benefits of using ``xarray`` `here <https://docs.xarray.dev/en/stable/getting-started-guide/why-xarray.html>`_.

Basic Usage
-----------

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
----------------------

Note that the operation above (reading a file using ``xr.open_dataset``)
did not immediately load arrays into memory. This is because the ``rex``
backend provides lazily-loaded arrays.

.. NOTE:: This is not Dask, since we did not specify the ``chunks=...`` parameter in ``xr.open_dataset``.

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


Lazy Loading - ``Dask``
-----------------------

We can also request that our data be read in lazily using `Dask <https://www.dask.org/>`_.
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
