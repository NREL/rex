# -*- coding: utf-8 -*-
# pylint: disable=all
"""
US Wave Command Line Interface
"""
import click
import logging
import os

from rex.resource_extraction.resource_cli import box as box_cmd
from rex.resource_extraction.resource_cli import dataset as dataset_grp
from rex.resource_extraction.resource_cli import multi_site as multi_site_cmd
from rex.resource_extraction.resource_cli import region as region_cmd
from rex.resource_extraction.resource_cli import site as site_cmd
from rex.resource_extraction.multi_year_resource_cli \
    import map_means as means_grp
from rex.resource_extraction.multi_year_resource_cli \
    import year as yr_means_cmd
from rex.resource_extraction.multi_year_resource_cli \
    import multi_year as multi_yr_means_cmd
from rex.resource_extraction.resource_extraction import MultiYearWaveX
from rex.utilities.cli_dtypes import STRLIST, INTLIST
from rex.utilities.loggers import init_mult

logger = logging.getLogger(__name__)


@click.group()
@click.option('--domain', '-d', required=True,
              type=click.Choice(['West_Coast', 'Hawaii'],
                                case_sensitive=False),
              help=('Geospatial domain to extract data from'))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--years', '-yrs', type=INTLIST, default=None,
              help='List of years to access, by default None')
@click.option('--buoy', '-b', is_flag=True,
              help="Boolean flag to use access virtual buoy data")
@click.option('--eagle', '-hpc', is_flag=True,
              help="Boolean flag to use access data on NRELs HPC vs. via HSDS")
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, domain, out_dir, years, buoy, eagle, verbose):
    """
    US Wave Command Line Interface
    """
    ctx.ensure_object(dict)
    if eagle:
        path = '/datasets/US_wave/v1.0.0'
        hsds = False
    else:
        path = '/nrel/US_wave'
        hsds = True

    if buoy:
        path = os.path.join(path, 'virtual_buoy', domain)
    else:
        path = os.path.join(path, domain)

    ctx.obj['H5'] = path
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['CLS_KWARGS'] = {'years': years, 'hsds': hsds}
    ctx.obj['CLS'] = MultiYearWaveX

    name = os.path.splitext(os.path.basename(path))[0]
    init_mult(name, out_dir, verbose=verbose, node=True,
              modules=[__name__, 'rex.resource_extraction',
                       'rex.multi_year_resource'])

    logger.info('Extracting US wave data for {} from {}'.format(domain, path))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


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
@click.option('--lat_lon', '-ll', nargs=2, type=click.Tuple([float, float]),
              default=None,
              help='(lat, lon) coordinates of interest')
@click.option('--gid', '-g', type=int, default=None,
              help='Resource gid of interest')
@click.pass_context
def site(ctx, lat_lon, gid):
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


@dataset.group()
@click.option('--region', '-r', type=str, default=None,
              help='Region to extract')
@click.option('--region_col', '-col', type=str, default='state',
              help='Meta column to search for region')
def map_means(ctx, region, region_col):
    """
    Map means for given dataset in region if given.
    """
    ctx.invoke(means_grp, region=region, region_col=region_col)


@map_means.command()
@click.option('--year', '-yr', type=str, required=True,
              help='Year to average')
@click.pass_context
def year(ctx, year):
    """
    Map means for a given year
    """
    ctx.invoke(yr_means_cmd, year=year)


@map_means.command()
@click.option('--years', '-yrs', type=STRLIST, required=True,
              help='List of Years to average')
@click.pass_context
def multi_year(ctx, years):
    """
    Map means for a given year
    """
    ctx.invoke(multi_yr_means_cmd, years=years)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running US_wave CLI')
        raise
