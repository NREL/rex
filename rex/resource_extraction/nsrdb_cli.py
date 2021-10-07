# -*- coding: utf-8 -*-
# pylint: disable=all
"""
NSRDBX Command Line Interface
"""
import click
import logging
import os

from rex.resource_extraction.resource_extraction import (NSRDBX, SolarX,
                                                         MultiFileNSRDBX)
from rex.resource_extraction.resource_cli import box as box_cmd
from rex.resource_extraction.resource_cli import dataset as dataset_grp
from rex.resource_extraction.resource_cli import multi_site as multi_site_cmd
from rex.resource_extraction.resource_cli import region as region_cmd
from rex.resource_extraction.resource_cli import sam_datasets as sam_cmd
from rex.resource_extraction.resource_cli import site as site_cmd
from rex.utilities.loggers import init_logger
from rex.utilities.utilities import check_res_file
from rex import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--solar_h5', '-h5', required=True,
              type=click.Path(),
              help=('Path to Resource .h5 file'))
@click.option('--out_dir', '-o', required=True, type=click.Path(exists=True),
              help='Directory to dump output files')
@click.option('--log_file', '-log', default=None, type=click.Path(),
              show_default=True,
              help='Path to .log file, if None only log to stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, solar_h5, out_dir, log_file, verbose):
    """
    NSRDBX Command Line Interface
    """
    ctx.ensure_object(dict)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ctx.obj['H5'] = solar_h5
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['CLS_KWARGS'] = {}

    multi_h5_res, hsds = check_res_file(solar_h5)
    name = os.path.splitext(os.path.basename(solar_h5))[0]
    if 'nsrdb' in name:
        ctx.obj['CLS'] = NSRDBX
    else:
        ctx.obj['CLS'] = SolarX

    if multi_h5_res and not hsds:
        assert os.path.exists(os.path.dirname(solar_h5))
        ctx.obj['CLS'] = MultiFileNSRDBX
    elif hsds:
        ctx.obj['CLS_KWARGS']['hsds'] = hsds
    else:
        assert os.path.exists(solar_h5)

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    init_logger('rex', log_file=log_file, log_level=log_level)

    logger.info('Extracting solar data from {}'.format(solar_h5))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


@main.command()
@click.option('--lat_lon', '-ll', nargs=2, type=float,
              default=None, show_default=True,
              help='(lat, lon) coordinates of interest')
@click.option('--gid', '-gid', type=int, default=None,
              show_default=True,
              help='Resource gid of interest')
@click.option('--sites', '-s', type=click.Path(exists=True), default=None,
              show_default=True,
              help=('.csv or .json file with columns "latitude", "longitude" '
                    'OR "gid"'))
@click.pass_context
def sam_datasets(ctx, lat_lon, gid, sites):
    """
    Extract all datasets needed for SAM for the nearest pixel(s) to the given
    (lat, lon) coordinates, the given resource gid, or the give sites
    """
    ctx.invoke(sam_cmd, lat_lon=lat_lon, gid=gid, sites=sites)


@main.group()
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract')
@click.pass_context
def dataset(ctx, dataset):
    """
    Extract a single dataset
    """
    ctx.invoke(dataset_grp, dataset=dataset)


@dataset.command()
@click.option('--lat_lon', '-ll', nargs=2, type=float,
              default=None, show_default=True,
              help='(lat, lon) coordinates of interest')
@click.option('--gid', '-gid', type=int, default=None, show_default=True,
              help='Resource gid of interest')
@click.pass_context
def site(ctx, lat_lon, gid):
    """
    Extract the nearest pixel to the given (lat, lon) coordinates OR the given
    resource gid
    """
    ctx.invoke(site_cmd, lat_lon=lat_lon, gid=gid)


@dataset.command()
@click.option('--region', '-r', type=str, required=True,
              help='Region to extract')
@click.option('--region_col', '-col', type=str, default='state',
              show_default=True,
              help='Meta column to search for region')
@click.option('--timestep', '-ts', type=str, default=None,
              show_default=True,
              help='Timestep to extract')
@click.pass_context
def region(ctx, region, region_col, timestep):
    """
    Extract a single dataset for all pixels in the given region
    """
    ctx.invoke(region_cmd, region=region, region_col=region_col,
               timestep=timestep)


@dataset.command()
@click.option('--lat_lon_1', '-ll1', nargs=2, type=float,
              required=True,
              help='One corner of the bounding box')
@click.option('--lat_lon_2', '-ll2', nargs=2, type=float,
              required=True,
              help='The other corner of the bounding box')
@click.option('--file_suffix', '-fs', default=None, show_default=True,
              help='Filename suffix')
@click.option('--timestep', '-ts', type=str, default=None, show_default=True,
              help='Timestep to extract')
@click.pass_context
def box(ctx, lat_lon_1, lat_lon_2, file_suffix, timestep):
    """
    Extract all gids in the given bounding box
    """
    ctx.invoke(box_cmd, lat_lon_1=lat_lon_1, lat_lon_2=lat_lon_2,
               file_suffix=file_suffix, timestep=timestep)


@dataset.command()
@click.option('--sites', '-s', type=click.Path(exists=True), required=True,
              help=('.csv or .json file with columns "latitude", "longitude" '
                    'OR "gid"'))
@click.pass_context
def multi_site(ctx, sites):
    """
    Extract multiple sites given in '--sites' .csv or .json as
    "latitude", "longitude" pairs OR "gid"s
    """
    ctx.invoke(multi_site_cmd, sites=sites)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running NSRDBX CLI')
        raise
