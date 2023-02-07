# -*- coding: utf-8 -*-
"""
pytests for multi-resolution resource handlers
"""
import h5py
import numpy as np
import os
import shutil
import tempfile
from scipy.spatial import KDTree

from rex import TESTDATADIR
from rex.multi_res_resource import MultiResolutionResource
from rex.outputs import Outputs
from rex.renewable_resource import WindResource


def make_multi_res_files(td):
    """Make multi-resolution files and handlers in a temporary directory from
    the full high-resolution files in the TESTDATADIR"""
    source_fp = os.path.join(TESTDATADIR, 'wtk/wtk_2010_100m.h5')
    fp_hr = os.path.join(td, 'wtk_2010_hr.h5')
    fp_lr = os.path.join(td, 'wtk_2010_lr.h5')
    shutil.copy(source_fp, fp_hr)

    lr_dsets = ['temperature_100m', 'pressure_100m']
    with WindResource(fp_hr) as hr_res:
        all_dsets = hr_res.dsets
        ti = hr_res.time_index
        meta = hr_res.meta
        lr_data = [hr_res[dset] for dset in lr_dsets]
        lr_attrs = hr_res.attrs
        lr_chunks = hr_res.chunks
        lr_dtypes = hr_res.dtypes

    t_slice = slice(None, None, 12)
    s_slice = slice(None, None, 10)
    lr_ti = ti[t_slice]
    lr_meta = meta.iloc[s_slice]
    lr_data = [d[t_slice, s_slice] for d in lr_data]
    lr_shapes = {d: (len(lr_ti), len(lr_meta)) for d in lr_dsets}
    Outputs.init_h5(fp_lr, lr_dsets, lr_shapes, lr_attrs, lr_chunks,
                    lr_dtypes, lr_meta, lr_ti)
    for name, arr in zip(lr_dsets, lr_data):
        Outputs.add_dataset(fp_lr, name, arr, lr_dtypes[name],
                            attrs=lr_attrs[name], chunks=lr_chunks[name])

    with h5py.File(fp_hr, 'a') as f:
        for dset in lr_dsets:
            del f[dset]

    lr_res = WindResource(fp_lr)
    hr_res = WindResource(fp_hr)
    mrr = MultiResolutionResource(hr_res, lr_res)

    assert len(mrr._nn_map) == len(hr_res.meta)
    assert all(np.isin(mrr._nn_map, np.arange(len(lr_res.meta))))

    assert all(d in mrr.dsets for d in all_dsets)
    assert all(d in mrr.shapes for d in all_dsets)
    assert all(d in mrr.scale_factors for d in all_dsets)
    assert all(d in mrr.attrs for d in all_dsets)
    assert all(d in mrr.attrs for d in all_dsets)

    return mrr, hr_res, lr_res


def test_mrr_indexing():
    """Test data indexing with the multi resolution resource handler."""
    with tempfile.TemporaryDirectory() as td:
        mrr, hr_res, lr_res = make_multi_res_files(td)
        tree = KDTree(lr_res.coordinates)

        dsets = ('pressure_100m', 'temperature_100m')
        gids_hr = ([0, 3, 4], slice(None), slice(3, 15, 3))
        for dset in dsets:
            lr_gids = tree.query(hr_res.coordinates)[1]
            lr_data = lr_res[dset, :, lr_gids]
            hr_data = mrr[dset]
            assert np.allclose(lr_data, hr_data[::12])

            for gid in gids_hr:
                lr_gids = tree.query(hr_res.coordinates[gid])[1]
                lr_data = lr_res[dset, :, lr_gids]
                hr_data = mrr[dset, :, gid]
                assert np.allclose(lr_data, hr_data[::12])
