# -*- coding: utf-8 -*-
"""
Classes to handle resource data at multiple spatiotemporal resolutions
"""
import copy
import logging
from scipy.spatial import KDTree
import warnings


logger = logging.getLogger(__name__)


class MultiResolutionResource:
    """Multi-resolution resource handler. Uses two resource handlers for files
    at two different spatiotemporal resolutions, and then interpolates the
    lower resolution data to the higher resolution data on the fly.
    """

    HR_ATTRS = ('meta', 'time_index', 'coordinates', 'lat_lon', 'data_version',
                'global_attrs', 'get_meta_arr')
    """Attributes that are always taken only from the high-res data handler"""

    def __init__(self, hr_res, lr_res, nn_map=None, nn_d=None):
        """
        Parameters
        ----------
        hr_res : Resource | MultiFileResource | MultiYearResource
            rex resource handler for the high-resolution data. All retrieval
            gid's are based on this dataset, and the lr_res data is mapped to
            this.
        lr_res : Resource | MultiFileResource | MultiYearResource
            rex resource handler for the low-resolution data. The data from
            this handler is mapped to the hr_res data.
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

        msg = ('The hr_res and lr_res classes need to be the same but '
               'received: {} and {}'
               .format(hr_res.__class__, lr_res.__class__))
        assert hr_res.__class__ == lr_res.__class__, msg

        self._hr_res = hr_res
        self._lr_res = lr_res
        self._nn_map = nn_map
        self._nn_d = nn_d

        if self._nn_map is None:
            tree = KDTree(self._lr_res.coordinates)
            self._nn_d, self._nn_map = tree.query(self._hr_res.coordinates)

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.h5_file)

        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._hr_res.close()
        self._lr_res.close()
        if type is not None:
            raise

    def __len__(self):
        return len(self._hr_res)

    def __getitem__(self, keys):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self.datasets):
            self._i = 0
            raise StopIteration

        dset = self.datasets[self._i]
        self._i += 1

        return dset

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
                if isinstance(hr_attr, list) and isinstance(lr_attr, list):
                    return list(set(hr_attr + lr_attr))
                elif isinstance(hr_attr, tuple) and isinstance(lr_attr, tuple):
                    return hr_attr + lr_attr
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
                raise RuntimeError(msg) from e

    def _preload_SAM(self, *args, **kwargs):
        """
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hr_sam = self._hr_res._preload_SAM(*args, **kwargs)
            hr_sites = args[0]
            lr_sites = [self._nn_map[i] for i in hr_sites]
            args = (lr_sites,) + args[1:]
            lr_sam = self._lr_res._preload_SAM(*args, **kwargs)

        return hr_sam
