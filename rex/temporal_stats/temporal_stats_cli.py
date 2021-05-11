# -*- coding: utf-8 -*-
"""
Resource Statistics Command Line Interface (CLI)
"""
import click
import logging
import os

from rex.multi_file_resource import MultiFileNSRDB, MultiFileWTK
from rex.renewable_resource import (NSRDB, WaveResource, WindResource)
from rex.resource import Resource
from rex.temporal_stats.temporal_stats import TemporalStats
from rex.utilities.cli_dtypes import INT
from rex.utilities.loggers import init_mult
from rex import __version__

logger = logging.getLogger(__name__)

RES_CLS = {'Resource': Resource,
           'NSRDB': NSRDB,
           'Wind': WindResource,
           'Wave': WaveResource,
           'MultiFileNSRDB': MultiFileNSRDB,
           'MultiFileWTK': MultiFileWTK}


@click.group(invoke_without_command=True, chain=True)
@click.version_option(version=__version__)
@click.option('--resource_path', '-h5', required=True,
              type=click.Path(),
              help=('Path to Resource .h5 files'))
@click.option('--dataset', '-dset', required=True,
              help='Dataset to extract stats for')
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--statistics', '-stats',
              type=click.Choice(['mean', 'median', 'stdev', 'std'],
                                case_sensitive=False), default=['mean'],
              multiple=True, show_default=True,
              help=("Statistics to extract, must be 'mean', 'median', 'std', "
                    "and / or 'stdev'")
              )
@click.option('--max_workers', '-mw', type=INT, default=None,
              show_default=True,
              help=('Number of workers to use, if 1 run in serial, if None use'
                    ' all available cores'))
@click.option('--res_cls', '-res',
              type=click.Choice(['Resource', 'NSRDB', 'Wind', 'Wave',
                                 'MultiFileNSRDB', 'MultiFileWTK'],
                                case_sensitive=False),
              default='Resource', show_default=True,
              help='Resource type')
@click.option('--hsds', '-hsds', is_flag=True,
              help=("Boolean flag to use h5pyd to handle .h5 'files' hosted "
                    "on AWS behind HSDS"))
@click.option('--chunks_per_worker', '-cpw', default=5, type=int,
              show_default=True,
              help='Number of chunks to extract on each worker')
@click.option('--lat_lon_only', '-ll', is_flag=True,
              help='Only append lat, lon coordinates to stats')
@click.option('--log_file', '-log', default=None, type=click.Path(),
              show_default=True,
              help='Path to .log file, if None only log to stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, resource_path, dataset, out_dir, statistics, max_workers,
         res_cls, hsds, chunks_per_worker, lat_lon_only, log_file, verbose):
    """
    TemporalStats Command Line Interface
    """
    ctx.ensure_object(dict)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    name = os.path.splitext(os.path.basename(resource_path))[0]
    out_fpath = '{}_{}.csv'.format(name, dataset)
    out_fpath = os.path.join(out_dir, out_fpath)

    if log_file is not None:
        name = os.path.basename(log_file).split('.')[0]
        log_dir = os.path.dirname(log_file)
    else:
        name = None
        log_dir = None

    init_mult(name, log_dir, [__name__, 'rex'], verbose=verbose)
    logger.info('Computing stats data from {}'.format(resource_path))
    logger.info('Outputs to be stored in: {}'.format(out_dir))

    res_stats = TemporalStats(resource_path, statistics=statistics,
                              res_cls=RES_CLS[res_cls], hsds=hsds)

    if ctx.invoked_subcommand is None:
        all_stats = res_stats.all_stats(
            dataset, max_workers=max_workers,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        res_stats.save_stats(all_stats, out_fpath)
    else:
        ctx.obj['STATS'] = res_stats
        ctx.obj['DSET'] = dataset
        ctx.obj['MAX_WORKERS'] = max_workers
        ctx.obj['CPW'] = chunks_per_worker
        ctx.obj['LL'] = lat_lon_only
        ctx.obj['OUT_PATH'] = out_fpath


@main.command()
@click.pass_context
def full(ctx):
    """
    Compute Stats for full file time-series
    """
    res_stats = ctx.obj['STATS']
    annual_stats = res_stats.full_stats(
        ctx.obj['DSET'], max_workers=ctx.obj["MAX_WORKERS"],
        chunks_per_worker=ctx.obj['CPW'],
        lat_lon_only=ctx.obj['LL'])

    out_fpath = ctx.obj['OUT_PATH'].replace('.csv', '_stats.csv')
    res_stats.save_stats(annual_stats, out_fpath)


@main.command()
@click.pass_context
def monthly(ctx):
    """
    Compute Monthly Stats
    """
    res_stats = ctx.obj['STATS']
    monthly_stats = res_stats.monthly_stats(
        ctx.obj['DSET'], max_workers=ctx.obj["MAX_WORKERS"],
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
        ctx.obj['DSET'], max_workers=ctx.obj["MAX_WORKERS"],
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
        ctx.obj['DSET'], max_workers=ctx.obj["MAX_WORKERS"],
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
