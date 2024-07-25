"""Code for regridding data from one list of coordinates to another"""

import logging
import pickle
import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime as dt
from typing import Optional

import dask.array as da
import numpy as np
import pandas as pd
import psutil
from sklearn.neighbors import BallTree


logger = logging.getLogger(__name__)


@dataclass
class Regridder:
    """Interpolate from one grid to another using inverse weighted distances.

    This class builds ball tree and runs all queries to create full
    arrays of indices and distances for neighbor points. It computes
    an array of weights used to interpolate from the old grid to the new
    grid.

    Parameters
    ----------
    source_meta : :class:`pandas.DataFrame`
        Set of coordinates for source grid. Must contain "latitude"
        and "longitude" columns representing the coordinates
        (in degrees).
    target_meta : :class:`pandas.DataFrame`
        Set of coordinates for target grid. Must contain "latitude"
        and "longitude" columns representing the coordinates
        (in degrees).
    k_neighbors : int, optional
        Number of nearest neighbors to use for interpolation.
        By default, ``4``.
    n_chunks : int
        Number of spatial chunks to use for tree queries. The total
        number of points in the `target_meta` will be split into
        `n_chunks`, and the points in each chunk will be queried at the
        same time. By default, ``100``.
    max_workers : int, optional
        Max number of workers to use for running all tree queries needed
        to build the full set of indices and distances for each
        `target_meta` coordinate. By default, ``None``, which uses all
        available CPU cores.
    min_distance : float, optional
        Min Haversine distance to new grid points from original points
        before filling with NaNs. By default, ``1e-12``.
    leaf_size : int, optional
        Leaf size for :class:`~sklearn.neighbors.BallTree` instance.
        By default, ``4``.
    """

    source_meta: pd.DataFrame
    target_meta: pd.DataFrame
    k_neighbors: Optional[int] = 4
    n_chunks: Optional[int] = 100
    max_workers: Optional[int] = None
    min_distance: Optional[float] = 1e-12
    leaf_size: Optional[int] = 4

    def __post_init__(self):
        self._tree = None
        self._distances = None
        self._indices = None
        self._weights = None

        fields = pprint.pformat(asdict(self), indent=2)
        logger.info("Initialized `Regridder` with:\n%s", fields)

    @property
    def distances(self):
        """Get distances for all tree queries."""
        if self._distances is None:
            self.init_queries()
        return self._distances

    @property
    def indices(self):
        """Get indices for all tree queries."""
        if self._indices is None:
            self.init_queries()
        return self._indices

    def init_queries(self):
        """Initialize arrays for tree queries and perform all queries"""
        self._indices = [None] * len(self.target_meta)
        self._distances = [None] * len(self.target_meta)
        self.get_all_queries(self.max_workers)

    @property
    def weights(self):
        """Get weights used for regridding"""
        if self._weights is None:
            self._weights = _compute_weights(self.distances, self.min_distance)
        return self._weights

    @property
    def tree(self):
        """Build ball tree from source_meta"""
        if self._tree is None:
            logger.info("Building ball tree for regridding.")
            ll2 = self.source_meta[["latitude", "longitude"]].values
            ll2 = np.radians(ll2)
            self._tree = BallTree(ll2, leaf_size=self.leaf_size,
                                  metric="haversine")
        return self._tree

    def get_all_queries(self, max_workers=None):
        """Query ball tree for all coordinates in the target_meta and store
        results"""

        if max_workers == 1:
            logger.info("Querying all coordinates in serial.")
            self.save_query(slice(None))

        else:
            logger.info("Querying all coordinates in parallel.")
            self._parallel_queries(max_workers=max_workers)
        logger.info("Finished querying all coordinates.")

    def _parallel_queries(self, max_workers=None):
        """Get indices and distances for all points in target_meta, in
        serial"""
        futures = {}
        now = dt.now()
        slices = np.arange(len(self.target_meta))
        slices = np.array_split(slices, min(self.n_chunks, len(slices)))
        slices = [slice(s[0], s[-1] + 1) for s in slices]
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for i, s_slice in enumerate(slices):
                future = exe.submit(self.save_query, s_slice=s_slice)
                futures[future] = i
                mem = psutil.virtual_memory()
                msg = ("Query futures submitted: {} out of {}. Current "
                       "memory usage is {:.3f} GB out of {:.3f} GB total."
                       .format(i + 1, len(slices), mem.used / 1e9,
                               mem.total / 1e9))
                logger.info(msg)

            logger.info(f"Submitted all query futures in {dt.now() - now}.")

            for i, future in enumerate(as_completed(futures)):
                idx = futures[future]
                mem = psutil.virtual_memory()
                msg = ("Query futures completed: {} out of {}. Current memory "
                       "usage is {:.3f} GB out of {:.3f} GB total."
                       .format(i + 1, len(futures), mem.used / 1e9,
                               mem.total / 1e9))
                logger.info(msg)
                try:
                    future.result()
                except Exception as e:
                    msg = ("Failed to query coordinate chunk with index={}"
                           .format(idx))
                    logger.exception(msg)
                    raise RuntimeError(msg) from e

    def save_query(self, s_slice):
        """Save tree query for coordinates specified by given spatial slice"""
        out = self.tree.query(self.get_spatial_chunk(s_slice),
                              k=self.k_neighbors)
        self.distances[s_slice] = out[0]
        self.indices[s_slice] = out[1]

    def get_spatial_chunk(self, s_slice):
        """Get list of coordinates in target_meta specified by the given
        spatial slice

        Parameters
        ----------
        s_slice : slice
            Slice specifying which spatial indices in the target grid
            should be selected. This selects `n_points` from the target
            grid.

        Returns
        -------
        ndarray
            Array of `n_points` in `target_meta` selected by `s_slice`.
        """
        out = self.target_meta.iloc[s_slice][["latitude", "longitude"]].values
        return np.radians(out)

    def __call__(self, data):
        """Regrid given spatiotemporal data over entire grid.

        Parameters
        ----------
        data : ndarray
            Spatiotemporal data to regrid to target_meta. Data can be
            flattened in the spatial dimension to match the
            `target_meta` or be in a 2D spatial grid, e.g.:
            (spatial, temporal) or (spatial_1, spatial_2, temporal)

        Returns
        -------
        out : ndarray
            Flattened regridded spatiotemporal data
            (spatial, temporal)
        """
        # if isinstance(data, da.core.Array):
        #     return self._regrid_dask(data)
        # return self._regrid_numpy(data)

        if len(data.shape) == 3:
            data = data.reshape((data.shape[0] * data.shape[1], -1))

        msg = "Input data must be 2D (spatial, temporal)"
        assert len(data.shape) == 2, msg
        new_shape = (len(self.indices), self.k_neighbors, -1)

        vals = data[da.concatenate(self.indices)].reshape(new_shape)
        vals = da.transpose(vals, axes=(2, 0, 1))
        return da.einsum("ijk,jk->ij", vals, self.weights).T

    @classmethod
    def run(cls, source_meta, target_meta, source_data, k_neighbors=4,
            n_chunks=100, max_workers=None, min_distance=1e-12,
            leaf_size=4):
        """Regrid data using inverse distance weighting.

        Parameters
        ----------
        source_meta : :class:`pandas.DataFrame`
            Set of coordinates for source grid. Must contain "latitude"
            and "longitude" columns representing the coordinates
            (in degrees).
        target_meta : :class:`pandas.DataFrame`
            Set of coordinates for target grid. Must contain "latitude"
            and "longitude" columns representing the coordinates
            (in degrees).
        source_data : ndarray
            Spatiotemporal data to regrid to `target_meta` coordinate
            grid. Data can be flattened in the spatial dimension to
            match the `target_meta` or be in a 2D spatial grid, e.g.:
            (spatial, temporal) or (spatial_1, spatial_2, temporal)
        leaf_size : int, optional
            Leaf size for :class:`~sklearn.neighbors.BallTree` instance.
            By default, ``4``.
        k_neighbors : int, optional
            Number of nearest neighbors to use for interpolation.
            By default, ``4``.
        n_chunks : int
            Number of spatial chunks to use for tree queries. The total
            number of points in the `target_meta` will be split into
            `n_chunks`, and the points in each chunk will be queried at
            the same time. By default, ``100``.
        max_workers : int, optional
            Max number of workers to use for running all tree queries
            needed to build the full set of indices and distances for
            each `target_meta` coordinate. By default, ``None``, which
            uses all available CPU cores.
        min_distance : float, optional
            Min Haversine distance to new grid points from original
            points before filling with NaNs. By default, ``1e-12``.
        """
        regridder = cls(source_meta=source_meta, target_meta=target_meta,
                        leaf_size=leaf_size, k_neighbors=k_neighbors,
                        n_chunks=n_chunks, max_workers=max_workers,
                        min_distance=min_distance)
        regridder.get_all_queries(max_workers)
        return regridder(source_data)


class CachedRegridder:

    MIN_DISTANCE = 1e-12

    def __init__(self, cache_pattern):
        self.indices, self.distances = self.load_cache(cache_pattern)
        self.weights = _compute_weights(self.distances, self.MIN_DISTANCE)

    def __call__(self, data):
        """Regrid given spatiotemporal data over entire grid

        Parameters
        ----------
        data : ndarray
            Spatiotemporal data to regrid to target_meta. Data can be flattened
            in the spatial dimension to match the target_meta or be in a 2D
            spatial grid, e.g.:
            (spatial, temporal) or (spatial_1, spatial_2, temporal)

        Returns
        -------
        out : ndarray
            Flattened regridded spatiotemporal data
            (spatial, temporal)
        """
        if len(data.shape) == 3:
            data = data.reshape((data.shape[0] * data.shape[1], -1))

        msg = 'Input data must be 2D (spatial, temporal)'
        assert len(data.shape) == 2, msg

        vals = data[np.array(self.indices)]
        vals = np.transpose(vals, (2, 0, 1))

        out = np.einsum('ijk,jk->ij', vals, self.weights).T
        return out

    @staticmethod
    def load_cache(cache_pattern):
        """Load cached indices and distances from ball tree query"""
        index_file = cache_pattern.format(array_name='indices')
        distance_file = cache_pattern.format(array_name='distances')

        with open(index_file, 'rb') as f:
            indices = pickle.load(f)
        with open(distance_file, 'rb') as f:
            distances = pickle.load(f)

        logger.info('Loaded cache files: %s, %s', index_file, distance_file)
        return indices, distances

    @classmethod
    def build_cache(cls, cache_pattern, *args, **kwargs):
        """Cache indices and distances from ball tree query"""
        index_file = cache_pattern.format(array_name='indices')
        distance_file = cache_pattern.format(array_name='distances')

        regridder = Regridder(*args, **kwargs)

        with open(index_file, 'wb') as f:
            pickle.dump(regridder.indices, f, protocol=4)
        with open(distance_file, 'wb') as f:
            pickle.dump(regridder.distances, f, protocol=4)
        logger.info('Saved cache files: %s, %s', index_file, distance_file)


def _compute_weights(distances, min_distance):
    """Compute inverse weights from distance values. """
    dists = np.array(distances, dtype=np.float32)
    mask = dists < min_distance
    dists[mask] = min_distance
    if mask.sum() > 0:
        logger.info("%d of %d neighbor distances are within %.5f.",
                    np.sum(mask), np.prod(mask.shape),
                    min_distance)
    weights = 1 / dists
    weights[mask.any(axis=1), :] = (np.eye(1, dists.shape[1])
                                    .flatten())
    return weights / np.sum(weights, axis=-1)[:, None]
