# -*- coding: utf-8 -*-
"""
rex backend for xarray implementation

Heavily based on the h5netcdf backend:
https://github.com/pydata/xarray/blob/main/xarray/backends/h5netcdf_.py
"""
import io
import os
import json
import logging
import warnings
from pathlib import Path
from collections import namedtuple

import h5py
import numpy as np
import dask.array as da
from xarray import coding
from xarray.backends.common import (AbstractDataStore, BackendArray,
                                    BackendEntrypoint, _normalize_path)
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.locks import (HDF5_LOCK, combine_locks, ensure_lock,
                                   get_write_lock)
from xarray import conventions
from xarray.core import indexing
from xarray.core.dataset import Dataset
from xarray.core.utils import (FrozenDict, is_remote_uri,
                               read_magic_number_from_file,
                               try_read_magic_number_from_file_or_path,
                               close_on_error)
from xarray.core.variable import Variable

from rex.utilities import (rex_unscale, import_fsspec_or_fail, is_hsds_file,
                           import_h5pyd_or_fail, assert_read_only_mode)


logger = logging.getLogger(__name__)
TI_DTYPE = np.dtype('datetime64[ns]')
_SA_FN = Path(__file__).parent / "standard_attrs.json"
with _SA_FN.open(encoding="utf-8") as fh:
    _SA = json.load(fh)
_EN_FN = Path(__file__).parent / "encondings.json"
with _EN_FN.open(encoding="utf-8") as fh:
    _EN = json.load(fh)


def _open_remote_file(file_path):
    fsspec = import_fsspec_or_fail(file_path)
    s3f = fsspec.open(file_path, mode="rb", anon=True,
                      default_fill_cache=False)
    return s3f.open()  # pylint: disable=no-member


def _get_h5_fn(handle):
    try:
        return handle.filename
    except AttributeError:
        return _get_h5_fn(handle.file)


def _get_lock(filename, mode):
    if mode == "r":
        return HDF5_LOCK
    return combine_locks([HDF5_LOCK, get_write_lock(filename)])


def _h5_root_or_group(handle, group, mode):
    if group in {None, "", "/"}:
        return handle

    if not isinstance(group, str):
        raise ValueError("group must be a string or None")

    path = group.strip("/").split("/")
    for key in path:
        try:
            handle = handle[key]
        except KeyError as e:
            if mode != "r":
                handle = handle.create_group(key)
            else:
                # wrap error to provide slightly more helpful message
                raise OSError(f"group not found: {key}", e) from e
    return handle


def _rex_var_dtype(variable_name, in_dtype):
    if _is_time_index(variable_name):
        return TI_DTYPE

    if in_dtype is str:
        # use object dtype (with additional vlen string metadata)
        # because that's the only way in numpy to represent variable
        # length strings and to check vlen string dtype in further steps
        # it also prevents automatic string concatenation via
        # conventions.decode_cf_variable
        in_dtype = coding.strings.create_vlen_dtype(str)

    return in_dtype

def _read_attributes(h5_var):
    # xarray GH451
    # to ensure conventions decoding works properly on Python 3,
    # decode all bytes attributes to strings
    attrs = {}
    for k, v in h5_var.attrs.items():
        if k not in ["_FillValue", "missing_value"]:
            if isinstance(v, bytes):
                try:
                    v = v.decode("utf-8")
                except UnicodeDecodeError:
                    msg = (f"'utf-8' codec can't decode bytes for attribute "
                           f"{k!r} of h5 object {h5_var.name!r}, "
                           f"returning bytes undecoded.")
                    logger.warning(msg)
                    warnings.warn(msg, UnicodeWarning)
        attrs[k] = v
    return attrs


def _compile_attrs(name, var, meta_index):
    attrs = {}
    for stand_name, stand_attrs in _SA.items():
        if name.startswith(stand_name):
            attrs.update(stand_attrs)

    attrs.update(_read_attributes(var))
    if _is_from_meta(meta_index):
        attrs.setdefault("description",
                         "Extracted from H5 file 'meta' variable")

    return attrs


def _compile_encoding(name, var, fn, dimensions, orig_shape):
    # netCDF4 specific encoding
    encoding = {
        "chunksizes": var.chunks,
        "fletcher32": var.fletcher32,
        "shuffle": var.shuffle,
    }
    for encoding_name, default_encoding in _EN.items():
        if name.startswith(encoding_name):
            encoding.update(default_encoding)

    if var.chunks:
        encoding["preferred_chunks"] = dict(zip(dimensions, var.chunks))
    # Convert h5py-style compression options to NetCDF4-Python
    # style, if possible
    if var.compression == "gzip":
        encoding["zlib"] = True
        encoding["complevel"] = var.compression_opts
    elif var.compression is not None:
        encoding["compression"] = var.compression
        encoding["compression_opts"] = var.compression_opts

    # save source so __repr__ can detect if it's local or not
    encoding["source"] = fn
    encoding["original_shape"] = orig_shape
    encoding.setdefault("dtype", var.dtype)
    return encoding


def _is_time_index(variable_name):
    return variable_name.casefold() in {"time_index", "time"}


def _is_from_meta(idx):
    return idx > -1


def _is_from_coords(idx):
    return idx > -1


def _fix_keys(keys):
    """h5pyd fancy indexing only works if iterables are lists """
    for key in keys:
        try:
            yield list(key)
        except TypeError:
            yield key


def _is_h5_dset(val):
    try:
        val.keys()
        return False
    except AttributeError:
        return True


def _iter_h5_groups(root, parent="/"):
    parent = str(parent)
    ds = root[parent]
    if _is_h5_dset(ds):
        return

    yield parent
    for subgroup in ds:
        gpath = f"/{subgroup}" if parent == "/" else f"{parent}/{subgroup}"
        yield from _iter_h5_groups(root[parent], parent=gpath)


VarInfo = namedtuple("VarInfo", ["name", "var", "meta_index", "coord_index"],
                     defaults=[-1, -1])


class RexMetaVar:
    __slots__ = ("chunks", "fletcher32", "shuffle", "dtype", "shape",
                 "compression", "compression_opts", "attrs")

    def __init__(self, meta_var, dtype):
        self.chunks = meta_var.chunks
        self.fletcher32 = meta_var.fletcher32
        self.shuffle = meta_var.shuffle
        self.dtype = dtype
        self.shape = meta_var.shape
        self.compression = None
        self.compression_opts = None
        self.attrs = {}


class RexCoordVar:
    __slots__ = ("chunks", "fletcher32", "shuffle", "dtype", "shape",
                 "compression", "compression_opts", "attrs")

    def __init__(self, coord_var):
        self.chunks = coord_var.chunks
        self.fletcher32 = coord_var.fletcher32
        self.shuffle = coord_var.shuffle
        self.dtype = coord_var.dtype
        self.shape = coord_var.shape[:-1]
        self.compression = None
        self.compression_opts = None
        self.attrs = {}


class RexArrayWrapper(BackendArray):
    __slots__ = ("datastore", "dtype", "shape", "variable_name", "meta_index",
                 "coord_index", "scale_factor", "adder")

    def __init__(self, variable_name, datastore, dtype, shape, meta_index=-1,
                 coord_index=-1, scale_factor=1, adder=0):
        self.datastore = datastore
        self.variable_name = variable_name
        self.meta_index = meta_index
        self.coord_index = coord_index
        self.shape = shape
        self.dtype = _rex_var_dtype(variable_name, dtype)
        self.scale_factor = scale_factor
        self.adder = adder

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR,
            self._getitem)

    def _getitem(self, key):
        if self.datastore.hsds:
            key = tuple(_fix_keys(key))
        with self.datastore.lock:
            array = self.get_array(needs_lock=False)
            if _is_from_coords(self.coord_index):
                return array[(*key, self.coord_index)]
            if _is_time_index(self.variable_name):
                values_as_str = array[key].astype("U")
                if len(values_as_str.shape) < 1: # scalar ti
                    values_as_str = values_as_str.split("+")[0]
                    return np.array(values_as_str, dtype=TI_DTYPE)
                values_no_tz = np.char.partition(values_as_str, "+")[:, 0]
                return values_no_tz.astype(TI_DTYPE)

            if _is_from_meta(self.meta_index):
                if self.variable_name == "gid":
                    return da.arange(self.shape[0])[key].compute()

                meta_info = array[key]
                if len(meta_info.shape) < 1:  # scalar index
                    return np.array(meta_info[self.meta_index],
                                    dtype=self.dtype)
                return np.array([col[self.meta_index] for col in meta_info],
                                dtype=self.dtype)

            return rex_unscale(array[key], self.scale_factor, self.adder)

    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        if _is_from_coords(self.coord_index):
            return ds["coordinates"]
        if _is_from_meta(self.meta_index):
            return ds["meta"]
        if _is_time_index(self.variable_name):
            return ds["time_index"]
        return ds[self.variable_name]


class RexStore(AbstractDataStore):
    """Store for reading NREL-rex style data via h5py"""

    __slots__ = ("_filename", "_group", "manager", "mode", "is_remote",
                 "lock", "_ds_shape", "hsds")

    def __init__(self, manager, group=None, mode=None, hsds=False,
                 lock=HDF5_LOCK):
        self.manager = manager
        self._group = group
        self.mode = mode
        self._filename = _get_h5_fn(self.ds)
        self._ds_shape = None
        self.hsds = hsds
        self.lock = ensure_lock(lock)

    @classmethod
    def open(cls, filename, mode="r", group=None, lock=None, h5_driver=None,
             h5_driver_kwds=None, hsds=False, hsds_kwargs=None):
        """Open a RexStore instance

        Parameters
        ----------
        filename : _type_
            _description_
        mode : str, default="r"
            _description_. By default, ``"r"``.
        group : _type_, optional
            _description_. By default, ``None``.
        lock : _type_, optional
            _description_. By default, ``None``.
        h5_driver : _type_, optional
            _description_. By default, ``None``.
        hsds : _type_, optional
            _description_. By default, ``None``.
        hsds_kwargs : _type_, optional
            _description_. By default, ``None``.

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            If `filename` is a bytes object or if the file does not
            start with valid HDF5 magic number.
        """
        remote_file = (isinstance(filename, str)
                       and is_remote_uri(filename)
                       and h5_driver is None)
        if remote_file:
            assert_read_only_mode(mode, service="s3/fsspec")
            filename = _open_remote_file(filename)

        if isinstance(filename, bytes):
            raise ValueError("can't open rex HDF5 as bytes; "
                             "try passing a path or file-like object")
        if isinstance(filename, io.IOBase):
            magic_number = read_magic_number_from_file(filename)
            if not magic_number.startswith(b"\211HDF\r\n\032\n"):
                raise ValueError(f"{magic_number!r} is not the signature "
                                 "of a valid rex HDF5 file")

        if lock is None:
            lock = _get_lock(filename, mode)

        hsds = hsds or is_hsds_file(filename)

        if hsds:
            h5pyd = import_h5pyd_or_fail(filename)

            hsds_kwargs = hsds_kwargs or {}
            hsds_kwargs["use_cache"] = False
            assert_read_only_mode(mode)
            manager = CachingFileManager(h5pyd.File, filename, mode=mode,
                                         kwargs=hsds_kwargs)
        else:
            h5_kwargs = {"driver": h5_driver}
            h5_kwargs.update(h5_driver_kwds or {})
            manager = CachingFileManager(h5py.File, filename, mode=mode,
                                         kwargs=h5_kwargs)
        return cls(manager, group=group, mode=mode, lock=lock, hsds=hsds)

    def _acquire(self, needs_lock=True):
        """Acquire a handle to the file object that can access data"""
        with self.manager.acquire_context(needs_lock) as root:
            ds = _h5_root_or_group(root, self._group, self.mode)
        return ds

    @property
    def ds(self):
        """obj: File object that can be used to access the data"""
        return self._acquire()

    @property
    def ds_shape(self):
        """tuple: Shape of the dataset, i.e. (time_index, meta)"""
        if self._ds_shape is None:
            self._ds_shape = (self.ds.get("time_index", np.array([])).shape[0],
                              self.ds.get("meta", np.array([])).shape[0])
        return self._ds_shape

    def open_store_variable(self, name, var, meta_index=-1, coord_index=-1):
        """_summary_

        Parameters
        ----------
        name : _type_
            _description_
        var : _type_
            _description_
        meta_index : int, optional
            _description_. By default, ``-1``.
        coord_index : int, optional
            _description_. By default, ``-1``.

        Returns
        -------
        _type_
            _description_
        """
        dimensions = self._detect_dimensions(name, var, meta_index)
        attrs = _compile_attrs(name, var, meta_index)
        sf = attrs.pop("scale_factor", 1)
        ao = attrs.pop("add_offset", 0)

        data = indexing.LazilyIndexedArray(RexArrayWrapper(name, self,
                                                           var.dtype,
                                                           var.shape,
                                                           meta_index,
                                                           coord_index,
                                                           scale_factor=sf,
                                                           adder=ao))

        encoding = _compile_encoding(name, var, self._filename, dimensions,
                                     orig_shape=data.shape)
        return Variable(dimensions, data, attrs, encoding)

    def _detect_dimensions(self, name, var, meta_index):
        """Guess dimension based on var name or shape"""
        if _is_from_meta(meta_index):
            return ["gid"]

        if _is_time_index(name):
            return ["time_index"]

        if name in {"latitude", "longitude"}:
            return ["gid"]

        if var.shape == self.ds_shape:
            return ["time_index", "gid"]

        if var.shape == (self.ds_shape[0],):
            return ["time_index"]

        if var.shape == (self.ds_shape[1],):
            return ["gid"]

        return ["gid"]  # default to gid dimension

    def get_variables(self):
        """Mapping of variables in the store

        Returns
        -------
        FrozenDict
            Dictionary mapping variable name to :obj:`xr.Variable`
            instance.
        """
        return FrozenDict((name, self.open_store_variable(name, *var_info))
                          for name, *var_info in self._iter_vars())

    def _iter_vars(self):
        """Iterate of variables in the store

        Order matters, so we iterate over non-coordinate vars first.
        Everything in the meta dataframe is assumed to be a coordinate.
        """
        iter_meta = False
        iter_coords = False
        for k, v in self.ds.items():
            if not _is_h5_dset(v):  # groups are not vars
                continue
            if k.casefold() == "coordinates":
                iter_coords = True  # handle below
                continue
            if k == "meta":
                iter_meta = True  # handle below
                continue
            if k == "time_index":
                yield VarInfo("time_index", v)
                yield VarInfo("time", v)  # for users who expect "time"
                continue
            yield VarInfo(k, v)

        # Get "gid" first, before lat/lon
        if iter_meta:
            meta_var = self.ds["meta"]
            yield VarInfo("gid", RexMetaVar(meta_var, np.dtype("int64")),
                          meta_index=0)

        # Get "lat/lon" next, before rest of meta
        already_got_from_coords = set()
        if iter_coords:
            coord_var = self.ds["coordinates"]
            yield VarInfo("latitude", RexCoordVar(coord_var), coord_index=0)
            yield VarInfo("longitude", RexCoordVar(coord_var), coord_index=1)
            already_got_from_coords = {"latitude", "longitude"}

        # Get rest of meta
        if iter_meta:
            meta_var = self.ds["meta"]
            for ind, (name, dtype) in enumerate(meta_var.dtype.fields.items()):
                if name in already_got_from_coords:
                    continue
                yield VarInfo(name, RexMetaVar(meta_var, dtype[0]),
                              meta_index=ind)

    def get_coord_names(self):
        """Set of variable names that represent coordinate datasets

        Most of these come from the meta, but some are based on datasets
        like `time_index` or `coordinates`.

        Returns
        -------
        set
            Set of variable names that should be treated as coordinates.
        """
        coords = set()
        if "time_index" in self.ds:
            coords.add("time_index")
            coords.add("time")

        if "coordinates" in self.ds:
            coords.add("latitude")
            coords.add("longitude")

        if "meta" in self.ds:
            for name in self.ds["meta"].dtype.fields:
                coords.add(name)
            coords.add("gid")

        return coords

    def get_attrs(self):
        """Get Dataset attribute dictionary

        Returns
        -------
        dict
            Immutable dictionary of attributes for the dataset.
        """
        return FrozenDict(_read_attributes(self.ds))

    def get_dimensions(self):
        """Get Dataset dimensions

        Returns
        -------
        dict
            Immutable mapping of dataset dimension names to their shape.
        """
        return FrozenDict((k, len(v.shape)) for k, v in self.ds.items())

    def close(self, **kwargs):
        """Close the store"""
        self.manager.close(**kwargs)


class RexBackendEntrypoint(BackendEntrypoint):
    """Backend for NREL rex-style files

    See Also
    --------
    backends.RexStore
    """

    description = "Open NREL-rex style HDF5 files in Xarray"
    url = ("https://nrel.github.io/rex/_autosummary/"
           "rex.external.rexarray.RexBackendEntrypoint.html")
    open_dataset_parameters = ["filename_or_obj", "drop_variables", "group",
                               "lock", "h5_driver", "h5_driver_kwds", "hsds",
                               "hsds_kwargs"]

    def guess_can_open(self, filename_or_obj):
        """Guess if this backend can read a file

        Parameters
        ----------
        filename_or_obj : path-like
            Filename used to guess wether this backend can open.

        Returns
        -------
        bool
            Flag indicating wether this backend can open the file or
            not.
        """
        magic_number = try_read_magic_number_from_file_or_path(filename_or_obj)
        if magic_number is not None:
            return magic_number.startswith(b"\211HDF\r\n\032\n")

        if isinstance(filename_or_obj, (os.PathLike, str)):
            fn = os.path.basename(filename_or_obj).casefold()
            if any(kw in fn for kw in ["nsrdb", "wtk", "sup3r"]):
                return True

        return False

    def open_dataset(self, filename_or_obj, *, drop_variables=None, group=None,
                     lock=None, h5_driver=None, h5_driver_kwds=None,
                     hsds=None, hsds_kwargs=None):
        """_summary_

        Parameters
        ----------
        filename_or_obj : str | path-like | ReadBuffer | AbstractDataStore
            _description_
        drop_variables : str | Iterable[str] | None, optional
            _description_. By default, ``None``.
        group : _type_, optional
            _description_. By default, ``None``.
        lock : _type_, optional
            _description_. By default, ``None``.
        driver : _type_, optional
            _description_. By default, ``None``.
        driver_kwds : _type_, optional
            _description_. By default, ``None``.

        Returns
        -------
        _type_
            _description_
        """
        filename_or_obj = _normalize_path(filename_or_obj)
        store = RexStore.open(filename_or_obj, group=group, lock=lock,
                              h5_driver=h5_driver,
                              h5_driver_kwds=h5_driver_kwds,
                              hsds=hsds, hsds_kwargs=hsds_kwargs)

        with close_on_error(store):
            ds = self._load_rex_dataset(store, drop_variables)

        return ds

    def open_datatree(self, filename_or_obj, *, drop_variables=None,
                      group=None, lock=None, h5_driver=None,
                      h5_driver_kwds=None, hsds=None, hsds_kwargs=None):
        # Delayed import for Python 3.9 compat
        # pylint: disable=import-outside-toplevel
        from xarray.backends.common import datatree_from_dict_with_io_cleanup

        groups_dict = self.open_groups_as_dict(filename_or_obj,
                                               drop_variables=drop_variables,
                                               group=group, lock=lock,
                                               h5_driver=h5_driver,
                                               h5_driver_kwds=h5_driver_kwds,
                                               hsds=hsds,
                                               hsds_kwargs=hsds_kwargs)

        return datatree_from_dict_with_io_cleanup(groups_dict)

    def open_groups_as_dict(self, filename_or_obj, *, drop_variables=None,
                            group=None, lock=None, h5_driver=None,
                            h5_driver_kwds=None, hsds=None, hsds_kwargs=None):
        # Delayed import for Python 3.9 compat
        # pylint: disable=import-outside-toplevel
        from xarray.core.treenode import NodePath

        filename_or_obj = _normalize_path(filename_or_obj)
        store = RexStore.open(filename_or_obj, group=group, lock=lock,
                              h5_driver=h5_driver,
                              h5_driver_kwds=h5_driver_kwds,
                              hsds=hsds, hsds_kwargs=hsds_kwargs)

        # Check for a group and make it a parent if it exists
        if group:
            parent = NodePath("/") / NodePath(group)
        else:
            parent = NodePath("/")

        groups_dict = {}
        for path_group in _iter_h5_groups(store.ds, parent=parent):
            group_store = RexStore(store.manager, group=path_group,
                                   mode=store.mode, hsds=store.hsds, lock=lock)
            with close_on_error(group_store):
                group_ds = self._load_rex_dataset(group_store, drop_variables)

            if group:
                group_name = str(NodePath(path_group).relative_to(parent))
            else:
                group_name = str(NodePath(path_group))
            groups_dict[group_name] = group_ds

        return groups_dict

    @staticmethod
    def _load_rex_dataset(store, drop_variables):
        """Create a dataset from an open store"""
        variables, attrs = store.load()

        variables, attrs, coord_names = conventions.decode_cf_variables(
            variables, attrs, mask_and_scale=False, decode_times=False,
            concat_characters=True, decode_coords=False,
            drop_variables=drop_variables, use_cftime=False,
            decode_timedelta=False,
        )

        ds = Dataset(variables, attrs=attrs)
        coord_names = (store.get_coord_names().intersection(variables))
        ds = ds.set_coords(coord_names)
        dimension_coords = {name: name for name in ["time_index", "gid"]
                            if name in coord_names}
        if dimension_coords:
            ds = ds.set_index(dimension_coords)
        ds.set_close(store.close)

        return ds
