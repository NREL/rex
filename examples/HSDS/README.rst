Highly Scalable Data Service (HSDS)
===================================

`The Highly Scalable Data Service (HSDS) <https://www.hdfgroup.org/solutions/highly-scalable-data-service-hsds/>`_ is a cloud-optimized solution for storing and accessing HDF5 files, e.g. the NREL wind and solar datasets. You can access NREL data via HSDS in a few ways. Read below to find out more.

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

If this simple query succeeds while larger data slices fail, it is almost definitely a limitation of the public API. You'll need to stand up your own HSDS server to retrieve more data. Read on below to find out how.

Setting up a Local HSDS Server
------------------------------

Setting up an HSDS server on an EC2 instance or your local laptop isn't too hard. Most of the instructions are copied from the `HSDS Repository <https://github.com/HDFGroup/hsds>`_ and the `h5pyd repository <https://github.com/HDFGroup/h5pyd>`_. Note that these install instructions are for a unix machine. For Windows machines, you can likely follow these instructions and use Windows-specific modifications from the HSDS and h5pyd repo instructions.

Make sure you have python 3.x (we recommend 3.10), pip, and git installed. We find it easiest to manage your HSDS environment by installing `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ and creating a clean HSDS environment. Once you have that setup, follow these instructions:

#. Clone the HSDS repo: ``$ git clone https://github.com/HDFGroup/hsds``
#. Go to the HSDS directory: ``$ cd hsds``
#. Run install: ``$ python setup.py install`` (this does some extra magic over a plain ``pip`` install)
#. Install h5pyd (no need to clone the repo on this one): ``$ pip install h5pyd``
#. Create a directory the server will use to store data: ``$ mkdir hsds_data``
#. Copy the config override file: ``$ cp ./admin/config/config.yml ./admin/config/override.yml`` and update these lines in the ``override.yml`` file (make sure you update the ``root_dir`` with the path to your cloned HSDS repo):

    .. code-block:: bash

        aws_region: us-west-2
        aws_s3_gateway: http://s3.us-west-2.amazonaws.com/
        aws_s3_no_sign_request: True
        hsds_endpoint: local
        root_dir: /<your_hsds_repo_directory>/hsds_data/
        bucket_name: nrel-pds-hsds

#. Optional: update performance options in the ``override.yml file`` like ``max_task_count``, ``dn_ram``, and ``sn_ram`` to increase the number of parallel HSDS workers and their memory allocation.
#. Start the HSDS server: ``$ sh ./runall.sh --no-docker`` and take note of the endpoint that is printed out (e.g. ``http+unix://%2Ftmp%2Fhs%2Fsn_1.sock``)
#. Open a new shell, activate the HSDS python environment you've been using, and run ``$ hsinfo``. You should see something similar to the following if your local HSDS server is running correctly:

    .. code-block:: bash

      server name: Highly Scalable Data Service (HSDS)
      server state: READY
      endpoint: http+unix://%2Ftmp%2Fhs%2Fsn_1.sock
      username: anonymous
      password:
      server version: 0.7.3
      node count: 4
      up: 1 min 51 sec
      h5pyd version: 0.13.1

#. If you see this successful message, you can move on. If ``hsinfo`` fails, something went wrong in the previous steps. 
#. Create a config file at ``~/.hscfg`` with the following (make sure the ``hs_endpoint`` matches the endpoint that the HSDS server printed out):

    .. code-block:: bash

      # Local HSDS server
      hs_endpoint = http+unix://%2Ftmp%2Fhs%2Fsn_1.sock

#. Test that h5pyd is configured correctly by running the following python script:

    .. code-block:: python

        import h5pyd
        with h5pyd.Folder('/nrel/') as f:
            print(list(f))

#. Assuming you see a list of NREL public dataset directories (e.g. ``['nsrdb', 'wtk', ...]``, congratulations! You have setup HSDS and h5pyd correctly.

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
