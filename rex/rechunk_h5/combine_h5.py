# -*- coding: utf-8 -*-
"""
Module to rechunk existing .h5 files
"""
import h5py
import logging
import numpy as np
import os
import time

from rex.resource import Resource
from rex.rechunk_h5.rechunk_h5 import get_chunk_slices

logger = logging.getLogger(__name__)


class CombineH5:
    """
    Class to combine multiple .h5 files
    """
    def __init__(self, combined_h5, *source_h5, axis=1, overwrite=True):
        """
        Parameters
        ----------
        combined_h5 : str
            Path to save combined .h5 file to
        source_h5 : str
            Path to source .h5 files
        axis : int, optional
            axis to combine datasets along, by default 1
        overwrite : bool, optional
            Flag to overwrite an existing h5_dst file, by default True
        """
        self._combined_h5 = combined_h5
        self._source_h5 = source_h5
        self._axis = axis
        self._datasets = None
        self._dset_attrs = self._check_datasets()
        self._dst_h5 = h5py.File(self.combined_h5,
                                 mode='w' if overwrite else 'w-')
        self._transfer_global_attrs()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def close(self):
        """
        Close h5 instance
        """
        self._dst_h5.close()

    @property
    def combined_h5(self):
        """
        Path to combined .h5 file

        Returns
        -------
        str
        """
        return self._combined_h5

    @property
    def source_h5(self):
        """
        Paths to source .h5 files

        Returns
        -------
        tuple
        """
        return self._source_h5

    @property
    def datasets(self):
        """
        Datasets to combine

        Returns
        -------
        list
        """
        if self._datasets is None:
            datasets = []
            for h5_path in self._source_h5:
                with Resource(h5_path) as f:
                    datasets.append(f.datasets)

            self._datasets = list(set(datasets[0]).intersection(*datasets[1:]))

        return self._datasets

    def _check_dset_properties(self, dset_name):
        """
        Check to ensure dataset is in both domains and extract
        dataset attributes

        Parameters
        ----------
        dset_name : str
            Dataset to check

        Returns
        -------
        attrs : dict
            Dataset attributes {k: v}
        shape : tuple
            Dataset shape
        dtype : str | np.dtype
            Dataset dtype
        chunks : tuple | None
            Dataset chunk size
        """
        attrs = {}
        shape = None
        dtype = None
        chunks = None
        for h5_path in self.source_h5:
            with Resource(h5_path) as f:
                if dset_name in f:
                    dset_attrs = f.get_attrs(dset=dset_name)
                    dset_shape, dset_dtype, dset_chunks = \
                        f.get_dset_properties(dset_name)
                else:
                    msg = '{} not in {}'.format(dset_name, h5_path)
                    logger.error(msg)
                    raise ValueError(msg)

            attrs.update(dset_attrs)

            if shape is None:
                shape = list(dset_shape)
                chunks = dset_chunks
                dtype = dset_dtype
            else:
                if dset_chunks != chunks:
                    msg = ("{} chunks ({} != {}) do not match between source "
                           "files!".format(dset_name, chunks, dset_chunks))
                    logger.error(msg)
                    raise RuntimeError(msg)

                if dset_dtype != dtype:
                    msg = ("{} dtypes ({} != {}) do not match between source "
                           "files!".format(dset_name, dtype, dset_dtype))
                    logger.error(msg)
                    raise RuntimeError(msg)

                for i, s in enumerate(dset_shape):
                    # pylint: disable=unsubscriptable-object
                    if i != self._axis and s != shape[i]:
                        msg = ("{} shape ({} != {}) does not match between "
                               "source files!"
                               .format(dset_name, dset_shape, shape))
                        logger.error(msg)
                        raise RuntimeError(msg)

                if self._axis <= len(shape):
                    # pylint: disable=unsupported-assignment-operation
                    shape[self._axis] += dset_shape[self._axis]

        return attrs, tuple(shape), dtype, chunks

    def _check_datasets(self):
        """
        Check datasets to ensure compatible dtype and shape

        Returns
        -------
        dset_attrs : dict
            Dictionary of combined dataset properties and attributes
        """
        dset_attrs = {}
        for dset in self.datasets:
            if dset not in ['meta', 'coordinates', 'time_index']:
                attrs, shape, dtype, chunks = self._check_dset_properties(dset)
                dset_attrs[dset] = {"attrs": attrs,
                                    "shape": shape,
                                    "dtype": dtype,
                                    "chunks": chunks}

        return dset_attrs

    def _transfer_global_attrs(self):
        """
        Transfer global attributes
        """
        global_attrs = {}
        for h5_path in self.source_h5:
            with Resource(h5_path) as f:
                global_attrs.update(f.get_attrs())

        if global_attrs:
            logger.info('Transfering global attributes')
            for k, v in global_attrs.items():
                logger.debug("- Transfering {}: {}".format(k, v))
                self._dst_h5.attrs[k] = v

    def _combine_time_index(self):
        """
        Combine time_index
        """
        logger.info('Combining time_index')
        if self._axis == 0:
            time_index = None
            chunks = None
            attrs = {}
            for h5_path in self.source_h5:
                with Resource(h5_path) as f:
                    ti = f.h5['time_index'][...]
                    attrs.update(f.get_attrs('time_index'))
                    if time_index is None:
                        time_index = ti
                        chunks = f.get_dset_properties('time_index')[-1]
                    else:
                        time_index = np.append(time_index, ti)
        else:
            with Resource(self.source_h5[0]) as f:
                time_index = f.h5['time_index'][...]
                attrs = f.get_attrs('time_index')
                chunks = f.get_dset_properties('time_index')[-1]

        logger.debug('Combined time_index has:\n'
                     'shape: {}\n'
                     'dtype: {}\n'
                     'chunks: {}'
                     .format(time_index.shape, time_index.dtype, chunks))
        ds = self._dst_h5.create_dataset('time_index',
                                         shape=time_index.shape,
                                         dtype=time_index.dtype,
                                         chunks=chunks,
                                         data=time_index)
        if attrs:
            for k, v in attrs.items():
                logger.debug("- Transfering attr {}: {}".format(k, v))
                ds.attrs[k] = v

    def _combine_meta(self):
        """
        Combine meta
        """
        logger.info('Combining meta')
        if self._axis == 1:
            meta = None
            chunks = None
            attrs = {}
            for h5_path in self.source_h5:
                with Resource(h5_path) as f:
                    m = f.h5['meta'][...]
                    attrs.update(f.get_attrs('meta'))
                    if meta is None:
                        meta = m
                        chunks = f.get_dset_properties('meta')[-1]
                    else:
                        meta = np.append(meta, m)
        else:
            with Resource(self.source_h5[0]) as f:
                meta = f.h5['meta'][...]
                attrs = f.get_attrs('meta')
                chunks = f.get_dset_properties('meta')[-1]

        logger.debug('Combined meta has:\n'
                     'shape: {}\n'
                     'dtype: {}\n'
                     'chunks: {}'
                     .format(meta.shape, meta.dtype, chunks))
        ds = self._dst_h5.create_dataset('meta',
                                         shape=meta.shape,
                                         dtype=meta.dtype,
                                         chunks=chunks,
                                         data=meta)
        if attrs:
            for k, v in attrs.items():
                logger.debug("- Transfering attr {}: {}".format(k, v))
                ds.attrs[k] = v

    def _combine_coordinates(self):
        """
        combine coordinates
        """
        logger.info('Combining coordinates')
        if 'coordinates' in self.datasets:
            with Resource(self.source_h5[0]) as f:
                chunks = f.get_dset_properties('coordinates')[-1]
                attrs = f.get_attrs('coordinates')
        else:
            chunks = None
            attrs = {}

        if self._axis == 1:
            coords = None
            for h5_path in self.source_h5:
                with Resource(h5_path) as f:
                    c = f.lat_lon
                    if coords is None:
                        coords = c
                    else:
                        coords = np.append(coords, c, axis=0)
        else:
            with Resource(self.source_h5[0]) as f:
                coords = f.lat_lon

        logger.debug('Combined coordinates have:\n'
                     'shape: {}\n'
                     'dtype: {}\n'
                     'chunks: {}'
                     .format(coords.shape, coords.dtype, chunks))
        ds = self._dst_h5.create_dataset('coordinates',
                                         shape=coords.shape,
                                         dtype=coords.dtype,
                                         chunks=chunks,
                                         data=coords)

        if attrs:
            for k, v in attrs.items():
                logger.debug("- Transfering attr {}: {}".format(k, v))
                ds.attrs[k] = v

    def _init_dataset(self, dset_name, dset_attrs):
        """
        Initialize dataset

        Parameters
        ----------
        dset_name : str
            Name of dataset to initialize
        dset_attrs : dict
            Dictionary of dataset and properties and attributes

        Returns
        -------
        ds : h5py.Dataset
            Initialized Dataset instance
        """
        logger.info('Initializing {}'.format(dset_name))
        logger.debug('- has:\n'
                     'shape: {}\n'
                     'dtype: {}\n'
                     'chunks: {}'
                     .format(dset_attrs['shape'],
                             dset_attrs['dtype'],
                             dset_attrs['chunks']))
        ds = self._dst_h5.create_dataset(dset_name,
                                         shape=dset_attrs['shape'],
                                         dtype=dset_attrs['dtype'],
                                         chunks=dset_attrs['chunks'])
        attrs = dset_attrs['attrs']
        if attrs:
            for k, v in attrs.items():
                logger.debug("- Transfering attr {}: {}".format(k, v))
                ds.attrs[k] = v

        return ds

    def _load_data(self, ds_in, ds_out, start, process_size=None):
        """
        Load data from ds_in to ds_out

        Parameters
        ----------
        ds_in : h5py.Dataset
            Open dataset instance for source data
        ds_out : h5py.Dataset
            Open dataset instance for rechunked data
        start : int
            Start position in combined dataset
        dset_attrs : dict
            Dictionary of dataset attributes (dtype, chunks, attrs)
        process_size : int, optional
            Ammount of data to be transfered at a time, by default None
        """
        ts = time.time()
        shape = ds_in.shape
        sites = shape[self._axis]

        if process_size is not None:
            slice_map = get_chunk_slices(sites, process_size)
            for i, (s, e) in enumerate(slice_map):
                dset_slice = []
                s += start
                e += start
                for i, _ in enumerate(shape):
                    if i == self._axis:
                        dset_slice.append(slice(s, e))
                    else:
                        dset_slice.append(slice(None))

                dset_slice = tuple(dset_slice)
                ds_out[dset_slice] = ds_in[dset_slice]

                logger.debug('\t- chunk # {} ({}:{}) transfered'
                             .format(i, s, e))
        else:
            dset_slice = []
            for i, s in enumerate(shape):
                if i == self._axis:
                    end = start + s
                    dset_slice.append(slice(start, end))
                else:
                    dset_slice.append(slice(None))

            ds_out[tuple(dset_slice)] = ds_in[:]

        tt = (time.time() - ts) / 60
        logger.debug('Data transfered in {:.4f} min'.format(tt))

    def _combine_dataset(self, dset_name, dset_attrs, process_size=None):
        """
        Load data from ds_in to ds_out

        Parameters
        ----------
        dset_name : str
            Name of dataset to initialize
        dset_attrs : dict
            Dictionary of dataset and properties and attributes
        process_size : int, optional
            Ammount of data to be transfered at a time, by default None
        """
        logger.info('Combining {}'.format(dset_name))
        ds_comb = self._init_dataset(dset_name, dset_attrs)
        start = 0
        for h5_path in self.source_h5:
            logger.debug('Transfering data from {}'
                         .format(os.path.basename(h5_path)))
            with Resource(h5_path) as f:
                ds_in = f.h5[dset_name]
                self._load_data(ds_in, ds_comb, start,
                                process_size=process_size)
                start += ds_in.shape[self._axis]

    def combine(self, process_size=None):
        """
        Combine source .h5 files

        Parameters
        ----------
        process_size : int, optional
            Ammount of data to be transfered at a time, by default None
        """
        self._combine_time_index()
        self._combine_meta()
        self._combine_coordinates()

        for dset_name, dset_attrs in self._dset_attrs.items():
            self._combine_dataset(dset_name, dset_attrs,
                                  process_size=process_size)

    @classmethod
    def run(cls, combined_h5, *source_h5, axis=1, overwrite=True,
            process_size=None):
        """
        Combine source .h5 files

        Parameters
        ----------
        combined_h5 : str
            Path to save combined .h5 file to
        source_h5 : str
            Path to source .h5 files
        axis : int, optional
            axis to combine datasets along, by default 1
        overwrite : bool, optional
            Flag to overwrite an existing h5_dst file, by default True
        process_size : int, optional
            Ammount of data to be transfered at a time, by default None
        """
        logger.info('Combining data from {} into {}'
                    .format(source_h5, combined_h5))
        with cls(combined_h5, *source_h5,
                 axis=axis, overwrite=overwrite) as comb:
            comb.combine(process_size=process_size)
