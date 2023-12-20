Zarr
====

You can use `Zarr <https://zarr.dev/>`_ to open NREL h5 resource files hosted on AWS S3 on your local computer. In our internal tests, this has comparable performance to reading data with an `HSDS <https://nrel.github.io/rex/misc/examples.hsds.html>`_ local server. The benefit of this approach is that you don't need to run an HSDS server, the drawback is that you need to handle a large meta data file for every .h5 file you access on S3.

We currently do not have this integrated into the ``rex`` resource handler classes and are evaluating whether or not this is worthwhile.

Extra Requirements
------------------

You may need some additional software beyond the rex requirements to run this example:

.. code-block:: bash

    pip install s3fs zarr fsspec kerchunk

Code Example
------------

To open an h5 file hosted on AWS S3, follow the code example below. Here are some caveats to this approach:

- Change ``s3_path`` and ``meta_path`` to your desired paths
- The meta data file ``meta_path`` may take a long time to generate, typically a few minutes but in rare cases up to an hour.
- The meta data file will be a few hundred MB and should be saved on your local hard drive.
- The meta data file is unique to every h5 file regardless of spatial meta data or temporal time index.
- Take care to apply dataset scale factors to convert from integer precision to physical units. In this case, the GHI scale factor is just 1, but it is often greater than 1. The rex resource handlers do this automatically but you need to do this manually when reading the data straight from disk.

.. code-block:: python

    import fsspec
    import ujson
    import zarr
    from pathlib import Path
    from kerchunk.hdf import SingleHdf5ToZarr

    s3_path = 's3://nrel-pds-nsrdb/current/nsrdb_2020.h5'
    meta_path = "./nsrdb_2020.json"

    storage_opts = dict(mode="rb", anon=True, default_fill_cache=False,
                        default_cache_type="none")

    h5chunks = SingleHdf5ToZarr(s3_path, storage_options=storage_opts,
                                inline_threshold=0)

    metadata_json_path = Path(meta_path)
    if metadata_json_path.exists() is False:
        with open(metadata_json_path, 'wb') as out:
            out.write(ujson.dumps(h5chunks.translate()).encode())

    with open(metadata_json_path, "rb") as f:
        mapper = fsspec.get_mapper("reference://", fo=ujson.load(f),
                                   remote_protocol="s3",
                                   remote_options={"anon": True})

    data = zarr.open(mapper)

    arr = data['ghi'][:, 0] / data['ghi'].attrs["scale_factor"]
    print(list(data.keys()))
    print(data['ghi'], data['ghi'].shape, data['ghi'].attrs)
    print(arr)
