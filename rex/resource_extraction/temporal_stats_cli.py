# -*- coding: utf-8 -*-
"""
Resource Statistics
"""
import click
import logging
import os

from rex.multi_file_resource import MultiFileNSRDB, MultiFileWTK
from rex.renewable_resource import (NSRDB, WaveResource, WindResource)
from rex.resource import Resource
from rex.resource_extraction.temporal_stats import TemporalStats
from rex.utilities.loggers import init_mult

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True, chain=True)
@click.option('--resource_path', '-h5', required=True,
              type=click.Path(),
              help=('Path to Resource .h5 files'))
@click.option('--dataset', '-dset', required=True,
              help='Dataset to extract stats for')
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--statistics', '-stats',
              type=click.Choice(['mean', 'median', 'stdev', 'std'],
                                case_sensitive=False), default='mean',
              multiple=True,
              help=("Statistics to extract, must be 'mean', 'median', 'std', "
                    "and / or 'stdev'")
              )
@click.option('--max_workers', '-mw', type=int, default=None,
              help=('Number of workers to use, if 1 run in serial, if None use'
                    ' all available cores'))
@click.option('--res_cls', '-res',
              type=click.Choice(['Resource', 'NSRDB', 'Wind', 'Wave',
                                 'MultiFileNSRDB', 'MultiFileWTK'],
                                case_sensitive=False),
              default='Resource',
              help='Resource type')
@click.option('--hsds', '-hsds', is_flag=True,
              help=("Boolean flag to use h5pyd to handle .h5 'files' hosted "
                    "on AWS behind HSDS"))
@click.option('--chunks_per_worker', '-cpw', default=5, type=int,
              help='Number of chunks to extract on each worker')
@click.option('--lat_lon_only', '-ll', is_flag=True,
              help='Only append lat, lon coordinates to stats')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, resource_path, dataset, out_dir, statistics, max_workers,
         res_cls, hsds, chunks_per_worker, lat_lon_only, verbose):
    """
    TemporalStats Command Line Interface
    """
    ctx.ensure_object(dict)

    if res_cls == 'Resource':
        res_cls = Resource
    elif res_cls == 'NSRDB':
        res_cls = NSRDB
    elif res_cls == 'Wind':
        res_cls = WindResource
    elif res_cls == 'Wave':
        res_cls = WaveResource
    if res_cls == 'MultiFileNSRDB':
        res_cls = MultiFileNSRDB
    if res_cls == 'MultiFileWTK':
        res_cls = MultiFileWTK

    res_stats = TemporalStats(resource_path, statistics=statistics,
                              max_workers=max_workers, res_cls=res_cls,
                              hsds=hsds)

    name = os.path.splitext(os.path.basename(resource_path))[0]
    out_fpath = '{}_{}.csv'.format(name, dataset)
    out_fpath = os.path.join(out_dir, out_fpath)
    init_mult(name, out_dir, verbose=verbose, node=True,
              modules=[__name__, 'rex.resource_extraction.resource_stats'])

    logger.info('Extracting Resource data from {}'.format(resource_path))
    logger.info('Outputs to be stored in: {}'.format(out_dir))

    if ctx.invoked_subcommand is None:
        all_stats = res_stats.all_stats(
            dataset,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        res_stats.save_stats(all_stats, out_fpath)
    else:
        ctx.obj['STATS'] = res_stats
        ctx.obj['DSET'] = dataset
        ctx.obj['CPW'] = chunks_per_worker
        ctx.obj['LL'] = lat_lon_only
        ctx.obj['OUT_PATH'] = out_fpath


@main.command()
@click.pass_context
def annual(ctx):
    """
    Compute Annual Stats
    """
    res_stats = ctx.obj['STATS']
    annual_stats = res_stats.annual_stats(
        ctx.obj['DSET'],
        chunks_per_worker=ctx.obj['CPW'],
        lat_lon_only=ctx.obj['LL'])

    out_fpath = ctx.obj['OUT_PATH'].replace('.csv', '_annual.csv')
    res_stats.save_stats(annual_stats, out_fpath)


@main.command()
@click.pass_context
def monthly(ctx):
    """
    Compute Monthly Stats
    """
    res_stats = ctx.obj['STATS']
    monthly_stats = res_stats.monthly_stats(
        ctx.obj['DSET'],
        chunks_per_worker=ctx.obj['CPW'],
        lat_lon_only=ctx.obj['LL'])

    out_fpath = ctx.obj['OUT_PATH'].replace('.csv', '_monthly.csv')
    res_stats.save_stats(monthly_stats, out_fpath)


@main.command()
@click.pass_context
def diurnal(ctx):
    """
    Compute Diurnal Stats
    """
    res_stats = ctx.obj['STATS']
    diurnal_stats = res_stats.diurnal_stats(
        ctx.obj['DSET'],
        chunks_per_worker=ctx.obj['CPW'],
        lat_lon_only=ctx.obj['LL'])

    out_fpath = ctx.obj['OUT_PATH'].replace('.csv', '_diurnal.csv')
    res_stats.save_stats(diurnal_stats, out_fpath)


@main.command()
@click.pass_context
def monthly_diurnal(ctx):
    """
    Compute Monthly-Diurnal Stats
    """
    res_stats = ctx.obj['STATS']
    monthly_diurnal_stats = res_stats.monthly_diurnal_stats(
        ctx.obj['DSET'],
        chunks_per_worker=ctx.obj['CPW'],
        lat_lon_only=ctx.obj['LL'])

    out_fpath = ctx.obj['OUT_PATH'].replace('.csv', '_monthly_diurnal.csv')
    res_stats.save_stats(monthly_diurnal_stats, out_fpath)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running TemporalStats CLI')
        raise
