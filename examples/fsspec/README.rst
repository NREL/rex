fsspec
======

Filesystem utilities from ``fsspec`` enable users outside of NREL to open h5 resource files hosted on AWS S3 on your local computer. In our internal tests, this is slower than the `HSDS <https://nrel.github.io/rex/misc/examples.hsds.html>`_ and `zarr <https://nrel.github.io/rex/misc/examples.zarr.html>`_ examples, but as of ``rex`` version v0.2.92 it requires zero setup beyond installing ``rex`` and ``fsspec`` as described below. This may be a good option for people outside of NREL trying to access small to medium amounts of NREL .h5 data in applications that are not sensitive to IO performance.

For more info on ``fsspec``, read the docs `here <https://filesystem-spec.readthedocs.io/en/latest/>`_


Code Example
------------

To open an .h5 file hosted on AWS S3, simply use a path to an S3 resource with any of the ``rex`` file handlers:

- Change ``fp`` to your desired AWS .h5 resource paths (find the s3 paths on `OEDI <https://openei.org/wiki/Main_Page>`_ or with the `AWS CLI <https://aws.amazon.com/cli/>`_).
- Running this example on a laptop, it takes ~14 seconds to read the meta data, and another ~14 seconds to read the GHI timeseries. This may be faster when running on AWS services in the same region hosting the .h5 file. It is much slower when running on the NREL VPN.

.. code-block:: python

    import time
    from rex import Resource

    fp = "s3://nrel-pds-nsrdb/current/nsrdb_1998.h5"
    res = Resource(fp)

    t0 = time.time()
    meta = res.meta
    print(meta)
    print(time.time() - t0)

    t0 = time.time()
    arr = res['ghi', :, 0]
    print(arr)
    print(time.time() - t0)
