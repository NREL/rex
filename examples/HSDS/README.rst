Highly Scalable Data Service (HSDS)
===================================

The Highly Scalable Data Service (HSDS) is a cloud-optimized solution for
storing and accessing HDF5 files, e.g. the NREL wind and solar datasets. You
can access NREL data via HSDS in a few ways. Read below to find out more.

NREL Developer API
------------------

The easiest way to get started with HSDS is to get a developer API key via the
`NREL Developer Network <https://developer.nrel.gov/signup/>`_. Once you have
your API key, create an HSDS config file at ``~/.hscfg`` with the following
entries (make sure you update the ``hs_api_key`` entry):

.. code-block:: bash

  # NREL dev api
  hs_endpoint = https://developer.nrel.gov/api/hsds
  hs_username =
  hs_password =
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

If this simple query succeeds while larger data slices fail, it is almost
definitely a limitation of the public API. You'll need to stand up your own
HSDS server to retrieve more data. Read on below to find out how.

Setting up a Local HSDS Server
------------------------------

Setting up an HSDS server on an EC2 instance or your local laptop isn't too
hard. Most of the instructions are copied from the `HSDS Repository
<https://github.com/HDFGroup/hsds>`_ and the `h5pyd repository
<https://github.com/HDFGroup/h5pyd>`_.

Make sure you have python 3.x, pip, and git installed. We find it easiest to
manage your HSDS environment by installing `miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_ and creating a clean HSDS
environment. Once you have that setup, follow these instructions:

#. Clone the HSDS repo: ``$ git clone https://github.com/HDFGroup/hsds``
#. Go to the hsds directory: ``$ cd hsds``
#. Run install: ``$ python setup.py install`` (this does some extra magic over a plain ``pip`` install)
#. Setup password file: ``$ cp ./admin/config/passwd.default ./admin/config/passwd.txt``
#. Create a directory the server will use to store data: ``$ mkdir hsds_data``
#. Create an HSDS test bucket: ``$ mkdir hsds_data/hsdstest``
#. Set your environment variables (make sure to update the ``ROOT_DIR``, ``AWS_SECRET_ACCESS_KEY``, and ``AWS_ACCESS_KEY_ID`` entries):

    .. code-block:: bash

        export ROOT_DIR=/your_hsds_repo_directory/hsds_data/
        export HSDS_ENDPOINT=local
        export ADMIN_USERNAME=admin
        export ADMIN_PASSWORD=admin
        export BUCKET_NAME=nrel-pds-hsds
        export AWS_REGION=us-west-2
        export AWS_S3_GATEWAY=http://s3.us-west-2.amazonaws.com/
        export AWS_SECRET_ACCESS_KEY=your_secret_access_key_goes_here
        export AWS_ACCESS_KEY_ID=your_access_key_id_goes_here

#. Optional: copy the config override file: ``cp ./admin/config/config.yml ./admin/config/override.yml``, update any config lines in the ``override.yml`` file that you wish to change, and remove all other lines (you can increase ``max_task_count``, ``dn_ram``, and ``sn_ram`` to increase the number of parallel HSDS workers and their memory allocation).
#. Start the HSDS server: ``$ ./runall.sh --no-docker``
#. Open a new shell, activate the python environment you've been using, and run ``hsinfo``. You should see something similar to the following if your local HSDS server is running correctly:

    .. code-block:: bash

        server name: Highly Scalable Data Service (HSDS)
        server state: READY
        endpoint: local
        username: admin (admin)
        password: *****
        server version: 0.7.3
        node count: 5
        up: 3 sec
        h5pyd version: 0.13.1

#. If you see this successful message, you can now move on to install h5pyd: ``pip install h5pyd``
#. Create a config file at ``~/.hscfg`` with the following:

    .. code-block:: bash

      # Local HSDS server
      hs_endpoint = local
      hs_username = admin
      hs_password = admin
      hs_api_key = None
      hs_bucket = nrel-pds-hsds

#. You should be in a new shell, so you'll need to set the same environment variables as in the HSDS server setup instructions above (``export ...``)
#. Test that h5pyd is configured correctly by running the following python script:

    .. code-block:: python

        import h5pyd
        with h5pyd.Folder('/nrel/') as f:
            print(list(f))

Assuming you see a list of NREL public dataset directories (e.g. ``['nsrdb',
'wtk', ...]``, congratulations! You have setup HSDS and h5pyd correctly.

HSDS and rex Usage Examples
---------------------------

Now that you have an HSDS server running locally and h5pyd set up, you can
access NREL data as if you were on the NREL super computer. First, start by
browsing the NREL HSDS data offerings by exploring the HSDS folder structure:

    .. code-block:: python

        import h5pyd
        with h5pyd.Folder('/nrel/') as f:
            print(list(f))

        with h5pyd.Folder('/nrel/nsrdb/') as f:
            print(list(f))

        with h5pyd.Folder('/nrel/wtk/') as f:
            print(list(f))

Once you find a file you want to access, you can use the ``rex`` utilities to
read the data:

    .. code-block:: python

        from rex import NSRDBX

        nsrdb_file = '/nrel/nsrdb/v3/nsrdb_2018.h5'
        nrel_coord = (39.741931, -105.169891)
        with NSRDBX(nsrdb_file, hsds=True) as f:
            meta = f.meta
            time_index = f.time_index
            datasets = f.datasets
            gid = f.lat_lon_gid(nrel_coord)
            dni = f.get_lat_lon_df('dni', nrel_coord)
            ghi = f['ghi', :, gid]

More details on the handler classes like ``NSRDBX`` can be found in the `rex
API reference <https://nrel.github.io/rex/_autosummary/rex.html>`_.
