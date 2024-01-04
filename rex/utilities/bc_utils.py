# -*- coding: utf-8 -*-
"""
rex bias correction utilities.
"""
import numpy as np
import logging
from warnings import warn
import rex.bias_correction


logger = logging.getLogger(__name__)


def _parse_bc_table(bc_df, gids):
    """Parse the bias correction table for required bc functions and kwargs

    Parameters
    ----------
    bc_df : pd.DataFrame
        DataFrame with wind or solar resource bias correction table. This
        must have columns "gid" and "method", where "gid" is the resource
        file indices, and "method" is a function name from the
        ``rex.bias_correction`` module. Only windspeed or GHI+DNI+DHI are
        corrected, depending on the technology. See the
        ``rex.bias_correction`` module for more details on available
        bias correction methods.
    gids : list | np.ndarray
        Array of integer gids (spatial indices) from the source h5 file.
        This is used to get the correct bias correction parameters from
        ``bias_correct`` table based on its ``gid`` column

    Returns
    -------
    bc_fun : function
        Function from ``rex.bias_correction`` to use.
    bc_fun_kwargs : dict
        Kwargs from ``bc_df`` to input to ``bc_fun``. This may include extra
        kwargs that are not required by ``bc_fun`` and should be cleaned before
        passing to the function.
    bool_bc : np.ndarray
        1D Boolean array with length equal to the ``gids`` input with ``True``
        where data has available bias correction inputs in ``bc_df`` and
        ``False`` where not
    """

    if 'method' not in bc_df:
        msg = ('Bias correction table provided, but "method" not '
               'found! Need to specify "method" as a function name '
               'from `rex.bias_correction`')
        logger.error(msg)
        raise KeyError(msg)

    if bc_df.index.name != 'gid':
        msg = ('Bias correction table must have "gid" column but only '
               'found: {}'.format(list(bc_df.columns)))
        assert 'gid' in bc_df, msg
        bc_df = bc_df.set_index('gid')

    gid_arr = np.array(gids)
    bool_bc = np.isin(gid_arr, bc_df.index.values)
    if not bool_bc.all():
        missing = gid_arr[~bool_bc]
        msg = ('{} sites were missing from the bias correction table, '
               'not bias correcting: {}'.format(len(missing), missing))
        logger.warning(msg)
        warn(msg)

    fun_name = bc_df['method'].unique()
    msg = ('rex bias correction currently only supports a single unique '
           'bias correction method per chunk of sites but received: {}'
           .format(fun_name))
    assert len(fun_name) == 1, msg
    bc_fun = getattr(rex.bias_correction, fun_name[0], None)
    if bc_fun is None:
        avail = [x for x in dir(rex.bias_correction) if not x.startswith('_')]
        msg = ('Could not find method name "{}" in ``rex.bias_correction`` '
               'which has the available objects: {}'
               .format(fun_name[0], avail))
        logger.error(msg)
        raise KeyError(msg)

    bc_fun_kwargs = {}
    for col in bc_df.columns:
        arr = bc_df.loc[gid_arr[bool_bc], col].values
        bc_fun_kwargs[col] = np.expand_dims(arr, 0)

    return bc_fun, bc_fun_kwargs, bool_bc
