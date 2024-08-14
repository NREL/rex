# -*- coding: utf-8 -*-
"""
pytests for multi-resolution resource handlers
"""
import pytest
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


# temporal step size (goes from 5min to hourly)
T_STEP = 12


def make_multi_res_files(td, interp_hh=False):
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

    t_slice = slice(None, None, T_STEP)
    s_slice = slice(None, None, 10)
    lr_ti = ti[t_slice]
    lr_meta = meta.iloc[s_slice]
    lr_data = [d[t_slice, s_slice] for d in lr_data]
    lr_shapes = {d: (len(lr_ti), len(lr_meta)) for d in lr_dsets}

    if interp_hh:
        new_dsets = []
        for i, dset in enumerate(lr_dsets):
            new = dset.replace('100m', '20m')
            new_dsets.append(new)
            lr_dtypes[new] = lr_dtypes[dset]
            lr_shapes[new] = lr_shapes[dset]
            lr_attrs[new] = lr_attrs[dset]
            lr_chunks[new] = lr_chunks[dset]
            lr_data.append(lr_data[i].copy() * 1.1)
        lr_dsets += new_dsets

    Outputs.init_h5(fp_lr, lr_dsets, lr_shapes, lr_attrs, lr_chunks,
                    lr_dtypes, lr_meta, lr_ti)
    for name, arr in zip(lr_dsets, lr_data):
        Outputs.add_dataset(fp_lr, name, arr, lr_dtypes[name],
                            attrs=lr_attrs[name], chunks=lr_chunks[name])

    with h5py.File(fp_hr, 'a') as f:
        for dset in lr_dsets:
            if dset in f:
                del f[dset]

    lr_res = WindResource(fp_lr)
    hr_res = WindResource(fp_hr)
    mrr = MultiResolutionResource(fp_hr, fp_lr, handler_class=WindResource)
    assert len(mrr._nn_map) == len(hr_res.meta)
    assert all(np.isin(mrr._nn_map, np.arange(len(lr_res.meta))))

    assert all(d in mrr.dsets for d in all_dsets)
    assert all(d in mrr.shapes for d in all_dsets)
    assert all(d in mrr.scale_factors for d in all_dsets)
    assert all(d in mrr.attrs for d in all_dsets)
    assert all(d in mrr.attrs for d in all_dsets)

    lr_res.close()
    hr_res.close()
    mrr.close()

    return fp_hr, fp_lr


def test_mrr_indexing():
    """Test data indexing with the multi resolution resource handler."""
    with tempfile.TemporaryDirectory() as td:
        fp_hr, fp_lr = make_multi_res_files(td)

        lr_res = WindResource(fp_lr)
        hr_res = WindResource(fp_hr)
        mrr = MultiResolutionResource(fp_hr, fp_lr, handler_class=WindResource)
        tree = KDTree(lr_res.coordinates)

        dsets = ('pressure_100m', 'temperature_100m')
        gids_hr = (0, 9, [1], [0, 3, 4], slice(None), slice(3, 15, 3))
        for dset in dsets:
            lr_gids = tree.query(hr_res.coordinates)[1]
            lr_data = lr_res[dset, :, lr_gids]
            hr_data = mrr[dset]
            assert np.allclose(lr_data, hr_data[::T_STEP])

            for gid in gids_hr:
                lr_gids = tree.query(hr_res.coordinates[gid])[1]
                lr_data = lr_res[dset, :, lr_gids]
                hr_data = mrr[dset, :, gid]
                assert len(lr_data.shape) == len(hr_data.shape)
                assert np.allclose(lr_data, hr_data[::T_STEP])

        lr_res.close()
        hr_res.close()
        mrr.close()


@pytest.mark.parametrize(['hh', 'interp_hh'], [[90, False], [50, True]])
def test_mrr_interp(hh, interp_hh):
    """Test hub height interpolation with the multi resolution resource handler
    """
    with tempfile.TemporaryDirectory() as td:
        fp_hr, fp_lr = make_multi_res_files(td, interp_hh=interp_hh)

        lr_res = WindResource(fp_lr)
        hr_res = WindResource(fp_hr)
        mrr = MultiResolutionResource(fp_hr, fp_lr, handler_class=WindResource)
        tree = KDTree(lr_res.coordinates)

        dsets = (f'pressure_{hh}m', f'temperature_{hh}m', f'windspeed_{hh}m')
        for dset in dsets:
            lr_gids = tree.query(hr_res.coordinates)[1]

            if dset.startswith(('press', 'temp')):
                truth = lr_res[dset, :, lr_gids]
                test = mrr[dset][::T_STEP]
            elif dset.startswith('wind'):
                truth = hr_res[dset]
                test = mrr[dset]

            assert np.allclose(truth, test)

        lr_res.close()
        hr_res.close()
        mrr.close()


def test_preload_sam():
    """Test preload of the SAM data object using the multi resolution resource
    handler."""
    sites = [0, 3, 5, 9]
    hh = 100
    with tempfile.TemporaryDirectory() as td:
        fp_hr, fp_lr = make_multi_res_files(td)
        mrr = MultiResolutionResource(fp_hr, fp_lr, handler_class=WindResource)
        sam = MultiResolutionResource.preload_SAM(fp_hr, fp_lr, sites,
                                                  hub_heights=hh,
                                                  handler_class=WindResource)

        for gid in sites:
            sam_df = sam[gid]
            for k in ('windspeed', 'temperature', 'pressure'):
                dset = f'{k}_{hh}m'
                true = mrr[dset, :, gid]
                test = sam_df[k].values
                if k == 'pressure':
                    true *= 9.86923e-6
                assert np.allclose(true, test)

        mrr.close()


@pytest.mark.timeout(10)
def test_multi_res_resource_iterator():
    """
    test MultiResolutionResource iterator. Incorrect implementation can
    cause an infinite loop
    """
    with tempfile.TemporaryDirectory() as td:
        fp_hr, fp_lr = make_multi_res_files(td)
        mrr = MultiResolutionResource(fp_hr, fp_lr, handler_class=WindResource)
        dsets_permutation = {(a, b) for a in mrr for b in mrr}
        num_dsets = len(mrr.datasets)

        mrr.close()

    assert len(dsets_permutation) == num_dsets ** 2
