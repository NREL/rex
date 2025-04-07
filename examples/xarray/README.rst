Using ``xarray``
================

As of rex ``v0.2.99``, you can read `NREL data files <https://nrel.github.io/rex/misc/examples.nrel_data.html>`_
using the popular open-source library `xarray <https://docs.xarray.dev/en/stable/index.html>`_. You can learn
more about the benefits of using ``xarray`` `here <https://docs.xarray.dev/en/stable/getting-started-guide/why-xarray.html>`_.

Basic Usage
-----------

To read in an NREL data file, simply supply ``engine="rex"`` to the
`open_dataset <https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html#xarray-open-dataset>`_
function.
