Highly Scalable Data Service (HSDS)
===================================

`The Highly Scalable Data Service (HSDS)
<https://www.hdfgroup.org/solutions/highly-scalable-data-service-hsds/>`_ is a
cloud-optimized solution for storing and accessing HDF5 files, e.g. the NREL
wind and solar datasets. You can access NREL data via HSDS in a few ways. Read
below to find out more.

Note that raw NREL .h5 data files are hosted on AWS S3. In contrast, the files
on HSDS are not real "files". They are just domains that you can access with
h5pyd or rex tools to stream small chunks of the files stored on S3. The
multi-terabyte .h5 files on S3 would be incredibly cumbersome to access
otherwise.

You can now use HSDS with ``xarray`` to open NREL datasets remotely! See the guide
`here <https://nrel.github.io/rex/misc/examples.xarray.html>`_ for details.

Extra Requirements
------------------

You may need some additional software beyond the basic ``rex`` install to run this example
(make sure you are running not running higher than Python 3.11):

.. code-block:: bash

    pip install NREL-rex[hsds]

NREL Developer API
------------------

The easiest way to get started with HSDS is to get a developer API key via the
`NREL Developer Network <https://developer.nrel.gov/signup/>`_. Once you have
your API key, create an HSDS config file at ``~/.hscfg`` with the following
entries (make sure you update the ``hs_api_key`` entry):

.. code-block:: bash

  # NREL dev api
  hs_endpoint = https://developer.nrel.gov/api/hsds
  hs_api_key = your_api_key_goes_here

You should then be able to access NREL hsds data using ``rex`` and ``h5pyd`` as
per the `usage examples below
<https://nrel.github.io/rex/misc/examples.hsds.html#hsds-and-rex-usage-examples>`_.
Note that this API is hosted on an NREL server and will have limits on the
amount of data you can access via HSDS. If you get a the ``OSError: Error
retrieving data: None`` errors, it's probably because you're hitting the public
IO limits. You can confirm this by trying to extract a very small amount of
data with ``h5pyd`` like this:

.. code-block:: python

  import h5pyd
  nsrdb_file = '/nrel/nsrdb/v3/nsrdb_2018.h5'
  with h5pyd.File(nsrdb_file) as f:
    data = f['ghi'][0, 0]
    print(data)

If this simple query succeeds while larger data slices fail, it is almost definitely a limitation of the public API. You'll need to stand up your own HSDS server to retrieve more data. Read the section on "Setting up a Local HSDS Server" below to find out how.

Setting up a Local HSDS Server
------------------------------

Setting up an HSDS server on an EC2 instance or your local machine isn't too hard. The instruction set here is intended to be comprehensive and followed *exactly*. Most of these instructions are adapted from the `HSDS Repository <https://github.com/HDFGroup/hsds>`_ and the `h5pyd repository <https://github.com/HDFGroup/h5pyd>`_, but this tutorial is intended to be comprehensive and regularly maintained for NREL use. Please note the minor differences in the Unix- and Windows-specific instructions below and be sure to follow these subtleties exactly!

Make sure you have python 3.x (we recommend 3.10), pip, and git installed. We find it easiest to manage your HSDS environment by installing `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ and creating a clean HSDS environment. Once you have that setup, follow these instructions:

#. In your shell, install nrel-rex >= v0.2.88 using pip, making sure to include the optional HSDS dependency:

    .. code-block:: bash

      pip install "nrel-rex[hsds]>=0.2.88"

#. Set your environment variables (if using windows, use ``set`` instead of ``export``) (this has to be done every time you login to a shell unless you set these in your ``.bashrc``):

    .. code-block:: bash

      export AWS_S3_GATEWAY=http://s3.us-west-2.amazonaws.com
      export AWS_S3_NO_SIGN_REQUEST=1

#. Create a HSDS configuration file at ``~/.hscfg`` (you can also use the ``hsconfigure`` CLI utility) with ONLY the following entries:

    .. code-block:: bash

      # Local HSDS server
      hs_endpoint = http://localhost:5101
      hs_bucket = nrel-pds-hsds

#. Start your HSDS local server in the active shell by running the command ``$ hsds``
#. If you are on windows and see a "Windows Security Alert" pop up, check the box for "Private networks" and click "Allow access"
#. After a few seconds, you should see the HSDS local server print the successful status ``READY! use endpoint: http://localhost:5101``
#. Open a new shell instance, activate the HSDS python environment you've been using, and run ``$ hsinfo``. You should see something similar to the following if your local HSDS server is running correctly:

    .. code-block:: bash

        server name: Highly Scalable Data Service (HSDS)
        server state: READY
        endpoint: http://localhost:5101
        username: anonymous
        password:
        server version: 0.8.4
        node count: 4
        up: 53 sec
        h5pyd version: 0.18.0

#. If you see this successful message, you can move on. If ``hsinfo`` fails, something went wrong in the previous steps.
#. Test that h5pyd is configured correctly by running the following python script. You can also use the HSDS CLI utility ``$ hsls /nrel/``

    .. code-block:: python

        import h5pyd
        with h5pyd.Folder('/nrel/') as f:
            print(list(f))

#. Assuming you see a list of NREL public dataset directories (e.g. ``['nsrdb', 'wtk', ...]``, congratulations! You have setup HSDS and h5pyd correctly.

HSDS and rex Usage Examples
---------------------------

Now that you have an HSDS server running locally and h5pyd set up, you can
access NREL data as if you were on the NREL supercomputer. First, start by
browsing the NREL HSDS data offerings by exploring the HSDS folder structure:

    .. code-block:: python

        import h5pyd
        with h5pyd.Folder('/nrel/') as f:
            print(list(f))

        with h5pyd.Folder('/nrel/nsrdb/') as f:
            print(list(f))

        with h5pyd.Folder('/nrel/wtk/') as f:
            print(list(f))

These commands can also be run by using the HSDS CLI utility: ``$ hsls /nrel/``.

Once you find a file you want to access, you can use the ``rex`` utilities to
read the data. See the docs page `here
<https://nrel.github.io/rex/misc/examples.nrel_data.html>`_ for more details.
