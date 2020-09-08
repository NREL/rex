# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Multi Year ResourceX Command Line Interface
"""
import click
import logging
import os

from rex.resource_extraction.resource_cli import box as box_cmd
from rex.resource_extraction.resource_cli import dataset as dataset_grp
from rex.resource_extraction.resource_cli import multi_site as multi_site_cmd
from rex.resource_extraction.resource_cli import region as region_cmd
from rex.resource_extraction.resource_cli import site as site_cmd
from rex.resource_extraction.resource_extraction import (MultiYearResourceX,
                                                         MultiYearNSRDBX,
                                                         MultiYearWindX,
                                                         MultiYearWaveX)
from rex.utilities.cli_dtypes import STRLIST, INTLIST
from rex.utilities.loggers import init_mult

logger = logging.getLogger(__name__)


@click.group()
@click.option('--resource_path', '-h5', required=True,
              type=click.Path(),
              help=('Path to Resource .h5 files'))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--years', '-yrs', type=INTLIST, default=None,
              help='List of years to access, by default None')
@click.option('--hsds', '-hsds', is_flag=True,
              help=("Boolean flag to use h5pyd to handle .h5 'files' hosted "
                    "on AWS behind HSDS"))
@click.option('--res_cls', '-res',
              type=click.Choice(['Resource', 'NSRDB', 'Wind', 'Wave'],
                                case_sensitive=False),
              default='Resource',
              help='Resource type')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, resource_path, out_dir, years, hsds, res_cls, verbose):
    """
    ResourceX Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['H5'] = resource_path
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['CLS_KWARGS'] = {'years': years, 'hsds': hsds}

    if res_cls == 'Resource':
        ctx.obj['CLS'] = MultiYearResourceX
    elif res_cls == 'NSRDB':
        ctx.obj['CLS'] = MultiYearNSRDBX
    elif res_cls == 'Wind':
        ctx.obj['CLS'] = MultiYearWindX
    elif res_cls == 'Wave':
        ctx.obj['CLS'] = MultiYearWaveX

    name = os.path.splitext(os.path.basename(resource_path))[0]
    init_mult(name, out_dir, verbose=verbose, node=True,
              modules=[__name__, 'rex.resource_extraction',
                       'rex.multi_year_resource'])

    logger.info('Extracting Resource data from {}'.format(resource_path))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


@main.command()
@click.option('--year', '-yr', type=str, required=True,
              help='Year to average')
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract')
@click.option('--region_col', '-col', type=str, default='state',
              help='Meta column to search for region')
@click.option('--region', '-r', type=str, default=None,
              help='Region to extract')
@click.pass_context
def year(ctx, year, dataset, region_col, region):
    """
    Average a single dataset for a given year
    Extract only pixels in region if given.
    """
    with ctx.obj['CLS'](ctx.obj['H5'], **ctx.obj['CLS_KWARGS']) as f:
        map_df = f.get_means_map(dataset, year, region=region,
                                 region_col=region_col)

    out_path = "{}-{}.csv".format(dataset, year)
    out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
    logger.info('Saving data to {}'.format(out_path))
    map_df.to_csv(out_path)


@main.command()
@click.option('--years', '-yrs', type=STRLIST, required=True,
              help='List of Years to average')
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract')
@click.option('--region_col', '-col', type=str, default='state',
              help='Meta column to search for region')
@click.option('--region', '-r', type=str, default=None,
              help='Region to extract')
@click.pass_context
def years(ctx, years, dataset, region_col, region):
    """
    Average a single dataset for a given years
    Extract only pixels in region if given.
    """
    with ctx.obj['CLS'](ctx.obj['H5'], **ctx.obj['CLS_KWARGS']) as f:
        map_df = f.get_means_map(dataset, years, region=region,
                                 region_col=region_col)

    out_path = "{}_{}-{}.csv".format(dataset, years[0], years[-1])
    out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
    logger.info('Saving data to {}'.format(out_path))
    map_df.to_csv(out_path)


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
def site(ctx, gid, lat_lon):
    """
    Extract the nearest pixel to the given (lat, lon) coordinates
    OR the given resource gid
    """
    ctx.invoke(site_cmd, lat_lon=lat_lon, gid=gid)


@dataset.command()
@click.option('--region', '-r', type=str, required=True,
              help='Region to extract')
@click.option('--region_col', '-col', type=str, default='state',
              help='Meta column to search for region')
@click.option('--timestep', '-ts', type=str, default=None,
              help='Timestep to extract')
@click.pass_context
def region(ctx, region, region_col, timestep):
    """
    Extract a single dataset for all gids in the given region
    """

    ctx.invoke(region_cmd, region=region, region_col=region_col,
               timestep=timestep)


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
        logger.exception('Error running MultiYearResourceX CLI')
        raise
