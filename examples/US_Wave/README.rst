High Resolution Ocean Surface Wave Hindcast
===========================================

Description
-----------

The development of this dataset was funded by the U.S. Department of Energy,
Office of Energy Efficiency & Renewable Energy, Water Power Technologies Office
to improve our understanding of the U.S. wave energy resource and to provide
critical information for wave energy project development and wave energy
converter (WEC) conceptual design. This will be the highest resolution publicly
available long-term wave hindcast dataset that – when complete – covers the
entire U.S. EEZ. As such, it could be of value to any company with marine
operations inside the U.S. EEZ. Specifically, the data can be used to
investigate the historical record of wave statistics at any U.S. site. This
level of detail could be of interest to the Oil and Gas industry for offshore
platform engineering, to the offshore wind industry for turbine and array
design, to offshore aquaculture production and blue economy development, to
coastal communities for extreme hazards mitigation,  to global shipping
companies and fisherman for a better understanding of weather windows and
seasonal wave climate patterns at a spatial resolution that does not exist
elsewhere. The NREL Offshore Wind group has expressed significant interest in
this dataset for device structural modeling, array design, and economic
modeling.

Model
-----

This is the highest resolution publicly available wave hindcast dataset. The
multi-scale, unstructured-grid modeling approach using WaveWatch III and SWAN
enabled long-term (decades) high-resolution hindcasts in a large regional
domain. The model was extensively validated not only for the most common wave
parameters, but also six IEC resource parameters and 2D spectra with high
quality spectral data derived from publicly available buoys. This creation of
this dataset was funded by the U.S. Department of Energy, Office of Energy
Efficiency & Renewable Energy, Water Power Technologies Office under Contract
DE-AC05-76RL01830 to Pacific Northwest National Laboratory (PNNL). Additional
details on detailed definitions of the variables found in the dataset, the
SWAN and WaveWatch III model configuration and model validation are available
in a peer-review publication
`Development and validation of a high-resolution regional wave hindcast model for U.S. West Coast wave resource characterization <https://www.osti.gov/biblio/1599105>`_
and a PNNL technical report:
`High-Resolution Regional Wave Hindcast for the U.S.West Coast <https://www.osti.gov/biblio/1573061/>`_.
This study was funded by the U.S. Department of Energy, Office of Energy
Efficiency & Renewable Energy, Water Power Technologies Office under Contract
DE-AC05-76RL01830 to Pacific Northwest National Laboratory (PNNL).

The following variables were extracted from the SWAM Model data and are
available in the .h5 files:
- Mean Wave Direction: Direction Normal to the Wave Crests
- Significant Wave Height: Calculated as the zeroth spectral moment (i.e., H_m0)
- Mean Absolute Period: Resolved Spectral Moment (m_0/m_1)
- Peak Period: The period associated with the maximum value of the wave energy spectrum
- Mean Zero Crossing Period: Total wave energy flux from all directions
- Energy Period: Spectral width characterizes the relative spreading of energy in the wave spectrum.
- Directionality Coefficient: Fraction of total wave energy travelling in the direction of maximum wave power
- Maximum Energy Direction: The direction from which the most wave energy is travelling
- Omni-Directional Wave Power: Total wave energy flux from all directions
- Spectral Width: Spectral width characterizes the relative spreading of energy in the wave spectrum.

The dataset currently covers the U.S. Exclusive Economic Zone ('EEZ', up to
200 nautical miles from shore) offshore of the West Coast, and includes shallow
nearshore regions not covered by previous model hindcasts. Future additions to
the dataset will extend the coverage to the entire U.S. EEZ, including Island
territories. The dataset has a 3-hour timestep spanning 32 years from 1979
through 2010. It includes the most common wave statistics (wave height, wave
direction, wave period), alongside several other wave statistics developed for
the wave energy sector. The dataset was generated from the unstructured-grid
SWAN model output that was driven by a WaveWatch III model with global-regional
nested grids. The SWAN model simulations were performed with a spatial
resolution as fine as 200 meters in shallow waters:

- West Coast United States: Dataset Available
- East Coast United States: Available soon
- Alaskan Coast: Available soon

Directory structure
-------------------

High Resolution Ocean Surface Wave Hindcast data is made available as a series
of hourly .h5 and can be found at:
``/nrel/US_wave/``

Currently 2 domains of data are available:
- West Coast: ``/nrel/US_wave/West_Coast/West_Coast_wave_${year}.h5``
- Hawaii: ``/nrel/US_wave/Hawaii/Hawaii_wave_${year}.h5``

Data Format
-----------

The data is provided in high density data file (.h5) separated by year. The
variables mentioned above are provided in 2 dimensional time-series arrays with
dimensions (time x location). The temporal axis is defined by the ``time_index``
dataset, while the positional axis is defined by the ``coordinate`` dataset. The
units for the variable data is also provided as an attribute (``units``). The
SWAN and IEC valiable names are also provide under the attributes
(``SWAWN_name``) and (``IEC_name``) respectively.

Data Access Examples
--------------------

Example scripts to extract wave resource data using the command line or python
are provided below:

The easiest way to access and extract data is by using the Resource eXtraction
tool `rex <https://nrel.github.io/rex/>`_

To use `rex` with `HSDS <https://github.com/NREL/hsds-examples>`_ you will need
to install `h5pyd`:

.. code-block:: bash

  pip install h5pyd

Next you'll need to configure HSDS:

.. code-block:: bash

  hsconfigure

and enter at the prompt:

.. code-block:: bash

  hs_endpoint = https://developer.nrel.gov/api/hsds
  hs_username =
  hs_password =
  hs_api_key = 3K3JQbjZmWctY0xmIfSYvYgtIcM3CN0cb1Y2w9bf


*The example API key here is for demonstation and is rate-limited per IP. To get your own API key, visit https://developer.nrel.gov/signup/*

You can also add the above contents to a configuration file at `~/.hscfg`

WaveX CLI
+++++++++

The `WaveX <https://nrel.github.io/rex/rex/rex.resource_extraction.wave_cli.html#wavex>`_
command line utility provides the following options and commands:

.. code-block:: bash

  WaveX --help

  Usage: WaveX [OPTIONS] COMMAND [ARGS]...

    WaveX Command Line Interface

  Options:
    -h5, --wave_h5 PATH  Path to Resource .h5 file  [required]
    -o, --out_dir PATH   Directory to dump output files  [required]
    -v, --verbose        Flag to turn on debug logging. Default is not verbose.
    --help               Show this message and exit.

  Commands:
    dataset     Extract a single dataset
    multi-site  Extract multiple sites given in '--sites' .csv or .json as...
    sam-file    Extract all datasets needed for SAM for the nearest pixel to...


WaveX python class
++++++++++++++++++

.. code-block:: python

    from rex import WaveX

    wave_file = '/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5'
    with WaveX(wave_file, hsds=True) as f:
        meta = f.meta
        time_index = f.time_index
        swh = f['significant_wave_height']


``rex`` also allows easy extraction of the nearest site to a desired (lat, lon)
location:

.. code-block:: python

    from rex import WaveX

    wave_file = '/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5'
    lat_lon = (34.399408, -119.841181)
    with WaveX(wave_file, hsds=True) as f:
        lat_lon_swh = f.get_lat_lon_df('significant_wave_height', nwtc)


or to extract all sites in a given region:

.. code-block:: python

    from rex import WaveX

    wave_file = '/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5'
    jurisdication='California'
    with WaveX(wave_file, hsds=True) as f:
        date = '2010-07-04 18:00:00'
        swh_map = f.get_timestep_map('significant_wave_height', date
                                     region=jurisdiction,
                                     region_col='jurisdiction')

If you would rather access the US Wave data directly using h5pyd:

.. code-block:: python

    # Extract the average wave height
    import h5pydd
    import pandas as pd

    # Open .h5 file
    with h5pyd.File('/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5', mode='r') as f:
        # Extract meta data and convert from records array to DataFrame
        meta = pd.DataFrame(f['meta'][...])
        # Significant Wave Height
        swh = f['significant_wave_height']
        # Extract scale factor
        scale_factor = swh.attrs['scale_factor']
        # Extract, average, and unscale wave height
        mean_swh = swh[...].mean(axis=0) / scale_factor

    # Add mean windspeed to meta data
    meta['Average Wave Height'] = mean_swh

.. code-block:: python

    # Extract time-series data for a single site
    import h5pyd
    import pandas as pd

    # Open .h5 file
    with h5pyd.File('/nrel/US_wave/West_Coast/West_Coast_wave_2010.h5', mode='r') as f:
        # Extract time_index and convert to datetime
        # NOTE: time_index is saved as byte-strings and must be decoded
        time_index = pd.to_datetime(f['time_index'][...].astype(str))
        # Initialize DataFrame to store time-series data
        time_series = pd.DataFrame(index=time_index)
        # Extract wave height, direction, and period
        for var in ['significant_wave_height', 'mean_wave_direction',
                    'mean_absolute_period']:
            # Get dataset
            ds = f[var]
            # Extract scale factor
            scale_factor = ds.attrs['scale_factor']
            # Extract site 100 and add to DataFrame
            time_series[var] = ds[:, 100] / scale_factor

References
----------

Please cite the most relevant publication below when referencing this dataset:

1) `Wu, Wei-Cheng, et al. "Development and validation of a high-resolution regional wave hindcast model for US West Coast wave resource characterization." Renewable Energy 152 (2020): 736-753. <https://www.osti.gov/biblio/1599105>`_
2) `Yang, Z., G. García-Medina, W. Wu, and T. Wang, 2020. Characteristics and variability of the Nearshore Wave Resource on the U.S. West Coast. Energy. <https://doi.org/10.1016/j.energy.2020.117818>`_
3) `Yang, Zhaoqing, et al. High-Resolution Regional Wave Hindcast for the US West Coast. No. PNNL-28107. Pacific Northwest National Lab.(PNNL), Richland, WA (United States), 2018. <https://doi.org/10.2172/1573061>`_

Disclaimer and Attribution
--------------------------

The National Renewable Energy Laboratory (“NREL”) is operated for the U.S.
Department of Energy (“DOE”) by the Alliance for Sustainable Energy, LLC
("Alliance"). Pacific Northwest National Laboratory (PNNL) is managed and
operated by Battelle Memorial Institute ("Battelle") for DOE. As such the
following rules apply:

This data arose from worked performed under funding provided by the United
States Government. Access to or use of this data ("Data") denotes consent with
the fact that this data is provided "AS IS," “WHEREIS” AND SPECIFICALLY FREE
FROM ANY EXPRESS OR IMPLIED WARRANTY OF ANY KIND, INCLUDING BUT NOT LIMITED TO
ANY IMPLIED WARRANTIES SUCH AS MERCHANTABILITY AND/OR FITNESS FOR ANY
PARTICULAR PURPOSE. Furthermore, NEITHER THE UNITED STATES GOVERNMENT NOR ANY
OF ITS ASSOCITED ENTITES OR CONTRACTORS INCLUDING BUT NOT LIMITED TO THE
DOE/PNNL/NREL/BATTELLE/ALLIANCE ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY
FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE DATA, OR REPRESENT THAT
ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS. NO ENDORSEMENT OF THE DATA
OR ANY REPRESENTATIONS MADE IN CONNECTION WITH THE DATA IS PROVIDED. IN NO
EVENT SHALL ANY PARTY BE LIABLE FOR ANY DAMAGES, INCLUDING BUT NOT LIMITED TO
SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES ARISING FROM THE PROVISION OF THIS
DATA; TO THE EXTENT PERMITTED BY LAW USER AGREES TO INDEMNIFY
DOE/PNNL/NREL/BATTELLE/ALLIANCE AND ITS SUBSIDIARIES, AFFILIATES, OFFICERS,
AGENTS, AND EMPLOYEES AGAINST ANY CLAIM OR DEMAND RELATED TO USER'S USE OF THE
DATA, INCLUDING ANY REASONABLE ATTORNEYS FEES INCURRED.

The user is granted the right, without any fee or cost, to use or copy the
Data, provided that this entire notice appears in all copies of the Data. In
the event that user engages in any scientific or technical publication
utilizing this data user agrees to credit DOE/PNNL/NREL/BATTELLE/ALLIANCE in
any such publication consistent with respective professional practice.
