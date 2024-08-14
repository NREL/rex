# -*- coding: utf-8 -*-
"""
Classes to handle resource data at multiple spatiotemporal resolutions
"""
import pandas as pd
import os
import copy
import logging
from inspect import signature
from scipy.spatial import KDTree

from rex.resource import Resource
from rex.utilities.parse_keys import parse_keys
from rex.utilities.exceptions import ResourceRuntimeError

logger = logging.getLogger(__name__)


class MultiResolutionResource:
    """Multi-resolution resource handler. Uses two resource handlers for files
    at two different spatiotemporal resolutions, and then interpolates the
    lower resolution data to the higher resolution data on the fly.
    """

    INTERPOLABLE_DSETS = ["temperature", "pressure", "windspeed",
                          "winddirection"]
    VARIABLE_NAME = "height"
    VARIABLE_UNIT = "m"

    HR_ATTRS = ('meta', 'time_index', 'coordinates', 'lat_lon', 'data_version',
                'global_attrs', 'get_meta_arr', 'shape')
    """Attributes that are always taken only from the high-res data handler"""

    def __init__(self, h5_hr, h5_lr, handler_class=Resource,
                 handle_kwargs=None, nn_map=None, nn_d=None):
        """
        Parameters
        ----------
        h5_hr : str
            Filepath to high-resolution h5 resource file.
        h5_lr : str
            Filepath to low-resolution h5 resource file.
        handler_class : str
            rex Resource handler class (not initialized) to open both the high
            and low resolution h5 files (both files must be of the same
            resource handler class).
        handle_kwargs : dict, optional
            Dictionary of optional keyword arguments to initialize the
            handler_class for the h5_hr and h5_lr
        nn_map : np.ndarray
            Optional 1D array of nearest neighbor mappings. This will be
            created if not provided. This is created by making a kdtree of the
            lr_res coords and then querying with the hr_res coords. As an
            example, nn_map[10] will return the lr_res index corresponding to
            gid 10 from the hr_res data
        nn_d : np.ndarray
            Optional 1D array of nearest neighbor distances. This will be
            created if not provided. This is created by making a kdtree of the
            lr_res coords and then querying with the hr_res coords. As an
            example, nn_map[10] will return the distance between hr_res gid=10
            and the corresponding lr_res site
        """

        if handle_kwargs is None:
            handle_kwargs = {}

        self._h5_hr = h5_hr
        self._h5_lr = h5_lr
        self._hr_res = handler_class(h5_hr, **handle_kwargs)
        self._lr_res = handler_class(h5_lr, **handle_kwargs)
        self._nn_map = nn_map
        self._nn_d = nn_d

        if self._nn_map is None:
            self._nn_d, self._nn_map = self.make_nn_map(self._hr_res,
                                                        self._lr_res)

        self._interpolation_variable = self._hr_res._parse_interp_var(
            self.datasets)
        self._interpolation_variable.update(self._hr_res._parse_interp_var(
            self.datasets))
        prop_name = "{}s".format(self.VARIABLE_NAME)
        setattr(self, prop_name, self._interpolation_variable)

    @staticmethod
    def make_nn_map(hr_res, lr_res):
        """Make the low-res-to-high-res resource nearest neighbor mapping

        Parameters
        ----------
        hr_res : Resource | MultiFileResource | MultiYearResource
            rex resource handler for the high-resolution data. All retrieval
            gid's are based on this dataset, and the lr_res data is mapped to
            this.
        lr_res : Resource | MultiFileResource | MultiYearResource
            rex resource handler for the low-resolution data. The data from
            this handler is mapped to the hr_res data.

        Returns
        -------
        nn_d : np.ndarray
            1D array of nearest neighbor distances. This is created by making a
            kdtree of the lr_res coords and then querying with the hr_res
            coords. As an example, nn_map[10] will return the distance between
            hr_res gid=10 and the corresponding lr_res site
        nn_map : np.ndarray
            1D array of nearest neighbor mappings. This is created by making a
            kdtree of the lr_res coords and then querying with the hr_res
            coords. As an example, nn_map[10] will return the lr_res index
            corresponding to gid 10 from the hr_res data
        """
        tree = KDTree(lr_res.coordinates)
        nn_d, nn_map = tree.query(hr_res.coordinates)
        return nn_d, nn_map

    def map_ds_slice(self, ds_slice):
        """Map the requested dataset slice from high-res spatial indices to
        low-res spatial indices

        Parameters
        ----------
        ds_slice : tuple
            Tuple where each entry is a slice or list index argument for the
            respective axis, e.g. (slice(None), [0, 2]) retrieves the full
            axis=0 and indices 0 and 2 from axis=1.

        Returns
        -------
        ds_slice : tuple
            Tuple where each entry is a slice or list index argument for the
            respective axis, e.g. (slice(None), [0, 2]) retrieves the full
            axis=0 and indices 0 and 2 from axis=1.
            The returned value is now low-res spatial indices using simple
            nearest neighbor.
        """

        if len(ds_slice) == 1:
            ds_slice = ds_slice + (slice(None), )

        elif len(ds_slice) > 2:
            msg = 'Cannot handle ds_slice > 2D'
            logger.error(msg)
            raise ResourceRuntimeError(msg)

        t_slice, s_slice = ds_slice
        s_slice = self._nn_map[s_slice]
        return (t_slice, s_slice)

    def time_interp(self, arr):
        """Perform temporal interpolation on the low-res data to match the
        high-res data.

        Parameters
        ----------
        arr : np.ndarray
            2D array with shape (time, sites) where time corresponds to the
            low-resolution resource.

        Returns
        -------
        arr : np.ndarray
            2D array with shape (time, sites) where the time axis has been
            linearly interpolated to the high-resolution time index.
        """
        ndim = len(arr.shape)
        arr = pd.DataFrame(arr, index=self._lr_res.time_index)
        arr = arr.reindex(self._hr_res.time_index)
        arr = arr.interpolate('linear').ffill().bfill().values
        if ndim == 1 and len(arr.shape) == 2:
            arr = arr.flatten()
        return arr

    def close(self):
        """Close active file handlers."""
        self._hr_res.close()
        self._lr_res.close()

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.h5_file)

        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def __len__(self):
        return len(self._hr_res)

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)
        _, ds_name = os.path.split(ds)
        base_name, _ = self._parse_name(ds_name)

        hr_heights = getattr(self._hr_res, '_interpolation_variable', {})
        lr_heights = getattr(self._lr_res, '_interpolation_variable', {})
        hr_heights = hr_heights.get(base_name, [])
        lr_heights = lr_heights.get(base_name, [])

        if ds_name.startswith('time_index'):
            out = self._hr_res._get_time_index(ds, ds_slice)

        elif ds_name.startswith('meta'):
            out = self._hr_res._get_meta(ds, ds_slice)

        elif ds_name.startswith('coordinates'):
            out = self._hr_res._get_coords(ds, ds_slice)

        elif 'SAM' in ds_name:
            msg = ('SAM dataframe retrieval not implemented for '
                   'MultiResolutionResource, use '
                   'MultiResolutionResource.preload_SAM() instead.')
            logger.error(msg)
            raise NotImplementedError(msg)

        elif ds_name in self._hr_res.dsets or len(hr_heights) > 0:
            out = self._hr_res._get_ds(ds, ds_slice)

        elif ds_name in self._lr_res.dsets or len(lr_heights) > 0:
            ds_slice = self.map_ds_slice(ds_slice)
            out = self._lr_res._get_ds(ds, ds_slice)
            out = self.time_interp(out)

        else:
            msg = ('Could not find data for ds_name = {}. Available dsets in '
                   'hr_file ({}): {}. Available dsets in lr_file ({}): {}.'
                   .format(ds_name, self._h5_hr, self._hr_res.dsets,
                           self._h5_lr, self._lr_res.dsets))
            logger.error(msg)
            raise RuntimeError(msg)

        return out

    def __iter__(self):
        return iter(self.datasets)

    def __contains__(self, dset):
        return dset in self.datasets

    def __getattr__(self, attr):
        if attr in dir(self):
            return getattr(self, attr)
        if attr in self.HR_ATTRS:
            return getattr(self._hr_res, attr)
        else:
            try:
                hr_attr = getattr(self._hr_res, attr)
                lr_attr = getattr(self._lr_res, attr)
                if hasattr(hr_attr, '__call__'):
                    return hr_attr
                elif isinstance(hr_attr, list) and isinstance(lr_attr, list):
                    return list(set(hr_attr + lr_attr))
                elif isinstance(hr_attr, tuple) and isinstance(lr_attr, tuple):
                    return tuple(set(hr_attr + lr_attr))
                elif isinstance(hr_attr, dict) and isinstance(lr_attr, dict):
                    out = copy.deepcopy(lr_attr)
                    out.update(hr_attr)
                    return out
            except Exception as e:
                msg = ('Could not retrieve attribute "{}" from '
                       'MultiResolutionResource handler, the hr and lr '
                       'handler attributes could not be combined: {} {}'
                       .format(attr, hr_attr, lr_attr))
                logger.error(msg)
                raise AttributeError(msg) from e

    @classmethod
    def preload_SAM(cls, h5_hr, h5_lr, sites, *args,
                    handler_class=Resource,
                    nn_map=None, nn_d=None,
                    **kwargs):
        """Pre-load resource data in a SAM resource handler for PySAM / reV run

        Parameters
        ----------
        h5_hr : str
            Filepath to high-resolution h5 resource file.
        h5_lr : str
            Filepath to low-resolution h5 resource file.
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
        *args : list
            Additional arguments required by the resource-specific data handler
            preload_SAM() method (e.g. "hub_heights" is required by
            WindResource handlers and can be provided here).
        handler_class : str
            rex Resource handler class (not initialized) to open both the high
            and low resolution h5 files (both files must be of the same
            resource handler class).
        nn_map : np.ndarray
            Optional 1D array of nearest neighbor mappings. This will be
            created if not provided. This is created by making a kdtree of the
            lr_res coords and then querying with the hr_res coords. As an
            example, nn_map[10] will return the lr_res index corresponding to
            gid 10 from the hr_res data
        nn_d : np.ndarray
            Optional 1D array of nearest neighbor distances. This will be
            created if not provided. This is created by making a kdtree of the
            lr_res coords and then querying with the hr_res coords. As an
            example, nn_map[10] will return the distance between hr_res gid=10
            and the corresponding lr_res site
        **kwargs : dict
            Additional arguments required to either initialize the
            resource-specific data handler or call the resource-specific
            preload_SAM() method (e.g. "hub_heights" is required by
            WindResource.preload_SAM() method and can be provided here in
            addition to optional args like "icing" or "precip_rate").

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with high-resolution resource
            for sites in project_points
        """

        sig = signature(handler_class)
        handle_kwargs = {k: v for k, v in kwargs.items()
                         if k in sig.parameters}

        cls_kwargs = dict(nn_map=nn_map, nn_d=nn_d,
                          handler_class=handler_class,
                          handle_kwargs=handle_kwargs)

        with cls(h5_hr, h5_lr, **cls_kwargs) as mrr:
            sig = signature(mrr._hr_res._preload_SAM)
            preload_kwargs = {k: v for k, v in kwargs.items()
                              if k in sig.parameters}
            SAM_res = mrr._hr_res._preload_SAM(mrr, sites, *args,
                                               **preload_kwargs)

        return SAM_res
