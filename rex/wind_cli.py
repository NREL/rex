# -*- coding: utf-8 -*-
"""
WindX Command Line Interface
"""
import click
import logging
import os

from rex.resource_extraction import WindX, MultiFileWindX
from rex.resource_cli import dataset as dataset_cmd
from rex.resource_cli import multi_site as multi_site_grp
from rex.resource_cli import region as region_cmd
from rex.resource_cli import site as site_cmd
from rex.resource_cli import timestep as timestep_cmd
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import check_res_file

logger = logging.getLogger(__name__)


@click.group()
@click.option('--wind_h5', '-h5', required=True,
              type=click.Path(),
              help=('Path to Resource .h5 file'))
@click.option('--out_dir', '-o', required=True, type=click.Path(exists=True),
              help='Directory to dump output files')
@click.option('--compute_tree', '-t', is_flag=True,
              help='Flag to force the computation of the cKDTree')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, wind_h5, out_dir, compute_tree, verbose):
    """
    WindX Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['H5'] = wind_h5
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['CLS_KWARGS'] = {'compute_tree': compute_tree}

    multi_h5_res, hsds = check_res_file(wind_h5)
    if multi_h5_res:
        assert os.path.exists(os.path.dirname(wind_h5))
        ctx.obj['CLS'] = MultiFileWindX
    else:
        if hsds:
            ctx.obj['CLS_KWARGS']['hsds'] = hsds
        else:
            assert os.path.exists(wind_h5)

        ctx.obj['CLS'] = WindX

    name = os.path.splitext(os.path.basename(wind_h5))[0]
    init_mult(name, out_dir, verbose=verbose, node=True,
              modules=[__name__, 'rex.resource_extraction.resource_extraction',
                       'rex.renewable_resource'])

    logger.info('Extracting Wind data from {}'.format(wind_h5))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


@main.command()
@click.option('--hub_height', '-h', type=int, required=True,
              help='Hub height to extract SAM variables at')
@click.option('--lat_lon', '-ll', nargs=2, type=click.Tuple([float, float]),
              default=None,
              help='(lat, lon) coordinates of interest')
@click.option('--gid', '-g', type=int, default=None,
              help='Resource gid of interest')
@click.pass_context
def sam_file(ctx, hub_height, lat_lon, gid):
    """
    Extract all datasets at the given hub height needed for SAM for
    nearest pixel to the given (lat, lon) coordinates OR the given
    resource gid
    """
    if lat_lon is None and gid is None:
        click.echo("Must supply '--lat-lon' OR '--gid'!")
        raise click.Abort()
    elif lat_lon and gid:
        click.echo("You must only supply '--lat-lon' OR '--gid'!")
        raise click.Abort()

    with ctx.obj['CLS'](ctx.obj['H5'], compute_tree=ctx.obj['TREE']) as f:
        if lat_lon is not None:
            SAM_df = f.get_SAM_lat_lon(hub_height, lat_lon)
        elif gid is not None:
            SAM_df = f.get_SAM_gid(hub_height, lat_lon)

    SAM_df['winddirection_{}m'.format(hub_height)] = 0

    out_path = "{}.csv".format(SAM_df.name)
    out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
    logger.info('Saving data to {}'.format(out_path))
    SAM_df.to_csv(out_path)


@main.command()
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract')
@click.option('--gid', '-g', type=int, default=None,
              help='Resource gid of interest')
@click.option('--lat_lon', '-ll', nargs=2, type=click.Tuple([float, float]),
              default=None,
              help='(lat, lon) coordinates of interest')
@click.pass_context
def site(ctx, dataset, gid, lat_lon):
    """
    Extract a single dataset for the nearest pixel to the given (lat, lon)
    coordinates OR the given resource gid
    """
    ctx.invoke(site_cmd, dataset=dataset, lat_lon=lat_lon, gid=gid)


@main.command()
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract')
@click.option('--timestep', '-ts', type=str, required=True,
              help='Timestep to extract')
@click.option('--region', '-r', type=str, default=None,
              help='Region to extract')
@click.option('--region_col', '-col', type=str, default='state',
              help='Meta column to search for region')
@click.pass_context
def timestep(ctx, dataset, timestep, region, region_col):
    """
    Extract a single dataset for a single timestep
    Extract only pixels in region if given.
    """
    ctx.invoke(timestep_cmd, dataset=dataset, timestep=timestep, region=region,
               region_col=region_col)


@main.command()
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract')
@click.option('--region', '-r', type=str, required=True,
              help='Region to extract')
@click.option('--region_col', '-col', type=str, default='state',
              help='Meta column to search for region')
@click.pass_context
def region(ctx, dataset, region, region_col):
    """
    Extract a single dataset for all pixels in the given region
    """
    ctx.invoke(region_cmd, dataset=dataset, region=region,
               region_col=region_col)


@main.group()
@click.option('--sites', '-s', type=click.Path(exists=True), required=True,
              help=('.csv or .json file with columns "latitude", "longitude" '
                    'OR "gid"'))
@click.pass_context
def multi_site(ctx, sites):
    """
    Extract multiple sites given in '--sites' .csv or .json as
    "latitude", "longitude" pairs OR "gid"s
    """
    ctx.invoke(multi_site_grp, sites=sites)


@multi_site.command()
@click.option('--hub_height', '-h', type=int, required=True,
              help='Hub height to extract SAM variables at')
@click.pass_context
def sam(ctx, hub_height):
    """
    Extract SAM variables
    """
    gid = ctx.obj['GID']
    lat_lon = ctx.obj['LAT_LON']
    with ctx.obj['CLS'](ctx.obj['H5'], compute_tree=ctx.obj['TREE']) as f:
        meta = f['meta']
        if lat_lon is not None:
            SAM_df = f.get_SAM_lat_lon(hub_height, lat_lon)
        elif gid is not None:
            SAM_df = f.get_SAM_gid(hub_height, gid)

    name = ctx.obj['NAME']
    gids = []
    for df in SAM_df:
        out_path = "{}-{}.csv".format(df.name, name)
        out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
        df['winddirection_{}m'.format(hub_height)] = 0
        gids.append(int(df.name.split('-')[-1]))
        logger.info('Saving data to {}'.format(out_path))
        df.to_csv(out_path)

    out_path = "{}-meta.csv".format(name)
    out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
    meta = meta.loc[gids]
    meta.index.name = 'gid'
    logger.info('Saving meta data to {}'.format(out_path))
    meta.to_csv(out_path)


@multi_site.command()
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract')
@click.pass_context
def dataset(ctx, dataset):
    """
    Extract given dataset for all sites
    """
    ctx.invoke(dataset_cmd, dataset=dataset)


if __name__ == '__cli__':
    main(obj={})
