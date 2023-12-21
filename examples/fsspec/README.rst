fsspec
======

You can use ``fsspec`` to open NREL h5 resource files hosted on AWS S3 on your local computer. In our internal tests, this is slower than the `HSDS <https://nrel.github.io/rex/misc/examples.hsds.html>`_ and `zarr <https://nrel.github.io/rex/misc/examples.zarr.html>`_ examples, but is much easier to set up. This may be a good option for people outside of NREL trying to access small to medium amounts of NREL .h5 data in applications that are not sensitive to IO performance.

For more info on ``fsspec``, read the docs `here <https://filesystem-spec.readthedocs.io/en/latest/>`_

Extra Requirements
------------------

You may need some additional software beyond the rex requirements to run this example:

.. code-block:: bash

    pip install fsspec

Code Example
------------

To open an .h5 file hosted on AWS S3, follow the code example below. Here are some caveats to this approach:

- Change ``fp`` to your desired AWS .h5 resource paths.
- Running this example on a laptop, it takes ~14 seconds to read the meta data, and another ~14 seconds to read the GHI timeseries. This may be faster when running on AWS services in the same region hosting the .h5 file. It is much slower when running on the NREL VPN.
- The ``s3f`` object works like a local .h5 filepath and can be passed to any of the ``rex`` resource handlers, which will handle all of the data scaling and formatting.

.. code-block:: python

    import time
    import fsspec
    from rex import Resource

    fp = "s3://nrel-pds-nsrdb/current/nsrdb_1998.h5"
    s3f = fsspec.open(fp, mode='rb', anon=True, default_fill_cache=False)
    res = Resource(s3f.open())

    t0 = time.time()
    meta = res.meta
    print(meta)
    print(time.time() - t0)

    t0 = time.time()
    arr = res['ghi', :, 0]
    print(arr)
    print(time.time() - t0)
