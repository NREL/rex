# -*- coding: utf-8 -*-
"""
WaveX Command Line Interface
"""
import click
import logging
import os

from rex.resource_extraction.resource_extraction import WaveX
from rex.resource_extraction.resource_cli import box as box_cmd
from rex.resource_extraction.resource_cli import dataset as dataset_grp
from rex.resource_extraction.resource_cli import multi_site as multi_site_cmd
from rex.resource_extraction.resource_cli import region as region_cmd
from rex.resource_extraction.resource_cli import sam_file as sam_file_cmd
from rex.resource_extraction.resource_cli import site as site_cmd
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import check_res_file

logger = logging.getLogger(__name__)


@click.group()
@click.option('--wave_h5', '-h5', required=True,
              type=click.Path(),
              help=('Path to Resource .h5 file'))
@click.option('--out_dir', '-o', required=True, type=click.Path(exists=True),
              help='Directory to dump output files')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, wave_h5, out_dir, verbose):
    """
    WaveX Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['H5'] = wave_h5
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['CLS_KWARGS'] = {}
    ctx.obj['CLS'] = WaveX

    _, hsds = check_res_file(wave_h5)
    name = os.path.splitext(os.path.basename(wave_h5))[0]
    if hsds:
        ctx.obj['CLS_KWARGS']['hsds'] = hsds
    else:
        assert os.path.exists(wave_h5)

    init_mult(name, out_dir, verbose=verbose, node=True,
              modules=[__name__, 'rex.resource_extraction',
                       'rex.renewable_resource'])

    logger.info('Extracting wave data from {}'.format(wave_h5))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


@main.command()
@click.option('--lat_lon', '-ll', nargs=2, type=click.Tuple([float, float]),
              default=None,
              help='(lat, lon) coordinates of interest')
@click.option('--gid', '-g', type=int, default=None,
              help='Resource gid of interest')
@click.pass_context
def sam_file(ctx, lat_lon, gid):
    """
    Extract all datasets needed for SAM for the nearest pixel to the given
    (lat, lon) coordinates OR the given resource gid
    """
    ctx.invoke(sam_file_cmd, lat_lon=lat_lon, gid=gid)


@main.command()
@click.option('--sites', '-s', type=click.Path(exists=True), required=True,
              help=('.csv or .json file with columns "latitude", "longitude" '
                    'OR "gid"'))
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract, if sam datasets us "SAM" or "sam"')
@click.pass_context
def multi_site(ctx, sites, dataset):
    """
    Extract multiple sites given in '--sites' .csv or .json as
    "latitude", "longitude" pairs OR "gid"s
    """
    ctx.invoke(multi_site_cmd, sites=sites, dataset=dataset)


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
@click.option('--gid', '-g', type=int, default=None,
              help='Resource gid of interest')
@click.option('--lat_lon', '-ll', nargs=2, type=click.Tuple([float, float]),
              default=None,
              help='(lat, lon) coordinates of interest')
@click.pass_context
def site(ctx, dataset, gid, lat_lon):
    """
    Extract the nearest pixel to the given (lat, lon) coordinates
    OR the given resource gid
    """
    ctx.invoke(site_cmd, dataset=dataset, lat_lon=lat_lon, gid=gid)


@dataset.command
@click.option('--region', '-r', type=str, required=True,
              help='Region to extract')
@click.option('--region_col', '-col', type=str, default='state',
              help='Meta column to search for region')
@click.option('--timestep', '-ts', type=str, default=None,
              help='Timestep to extract')
@click.pass_context
def region(ctx, dataset, region, region_col, timestep):
    """
    Extract a single dataset for all gids in the given region
    """

    ctx.invoke(region_cmd, dataset=dataset, region=region,
               region_col=region_col, timestep=timestep)


@dataset.command()
@click.option('--lat_lon_1', '-ll1', nargs=2, type=click.Tuple([float, float]),
              required=True,
              help='One corner of the bounding box')
@click.option('--lat_lon_2', '-ll2', nargs=2, type=click.Tuple([float, float]),
              required=True,
              help='The other corner of the bounding box')
@click.option('--timestep', '-ts', type=str, default=None,
              help='Timestep to extract')
@click.option('--file_suffix', '-fs', default=None,
              help='File name suffix')
@click.pass_context
def box(ctx, lat_lon_1, lat_lon_2, timestep, file_suffix):
    """
    Extract all pixels in the given bounding box
    """
    ctx.invoke(box_cmd, lat_lon_1=lat_lon_1, lat_lon_2=lat_lon_2,
               file_suffix=file_suffix, timestep=timestep)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running WaveX CLI')
        raise
