# -*- coding: utf-8 -*-
"""
WindX Command Line Interface
"""
import click
import logging
import os

from rex.resource_extraction.resource_extraction import WindX, MultiFileWindX
from rex.resource_extraction.resource_cli import box as box_cmd
from rex.resource_extraction.resource_cli import dataset as dataset_grp
from rex.resource_extraction.resource_cli import _parse_sites
from rex.resource_extraction.resource_cli import region as region_cmd
from rex.resource_extraction.resource_cli import site as site_cmd
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import check_res_file

logger = logging.getLogger(__name__)


@click.group()
@click.option('--wind_h5', '-h5', required=True,
              type=click.Path(),
              help=('Path to Resource .h5 file'))
@click.option('--out_dir', '-o', required=True, type=click.Path(exists=True),
              help='Directory to dump output files')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, wind_h5, out_dir, verbose):
    """
    WindX Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['H5'] = wind_h5
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['CLS_KWARGS'] = {}

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
              modules=[__name__, 'rex.resource_extraction',
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
def site(ctx, dataset, lat_lon, gid):
    """
    Extract the nearest pixel to the given (lat, lon) coordinates OR the
    given resource gid
    """

    ctx.invoke(site_cmd, dataset=dataset, lat_lon=lat_lon, gid=gid)


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
              help='Filename suffix')
@click.pass_context
def box(ctx, lat_lon_1, lat_lon_2, timestep, file_suffix):
    """
    Extract all pixels in the given bounding box
    """

    ctx.invoke(box_cmd, lat_lon_1=lat_lon_1, lat_lon_2=lat_lon_2,
               file_suffix=file_suffix, timestep=timestep)


@dataset.command
@click.option('--region', '-r', type=str, required=True,
              help='Region to extract')
@click.option('--region_col', '-col', type=str, default='state',
              help='Meta column to search for region')
@click.option('--timestep', '-ts', type=str, default=None,
              help='Time-step to extract')
@click.pass_context
def region(ctx, dataset, region, region_col, timestep):
    """
    Extract a single dataset for all gids in the given region
    """

    ctx.invoke(region_cmd, dataset=dataset, region=region,
               region_col=region_col, timestep=timestep)


@main.command()
@click.option('--sites', '-s', type=click.Path(exists=True), required=True,
              help=('.csv or .json file with columns "latitude", "longitude" '
                    'OR "gid"'))
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract, if sam datasets us "SAM" or "sam"')
@click.option('--hub_height', '-h', type=int, default=None,
              help='Hub height to extract SAM variables at')
@click.pass_context
def multi_site(ctx, sites, dataset, hub_height):
    """
    Extract multiple sites given in '--sites' .csv or .json as
    "latitude", "longitude" pairs OR "gid"s
    """
    name, gid, lat_lon = _parse_sites(sites)

    if dataset.lower() == 'sam':
        if hub_height is None:
            click.echo('hub_height must be supplied to extract SAM datasets!')
            click.Abort()

        with ctx.obj['CLS'](ctx.obj['H5'], **ctx.obj['CLS_KWARGS']) as f:
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
    else:
        with ctx.obj['CLS'](ctx.obj['H5'], **ctx.obj['CLS_KWARGS']) as f:
            meta = f['meta']
            if lat_lon is not None:
                site_df = f.get_lat_lon_df(dataset, lat_lon)
            elif gid is not None:
                site_df = f.get_gid_df(dataset, gid)

        out_path = "{}-meta.csv".format(name)
        out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
        meta = meta.loc[site_df.columns]
        logger.info('Saving meta data to {}'.format(out_path))
        meta.to_csv(out_path)

        out_path = "{}-{}.csv".format(dataset, name)
        out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
        logger.info('Saving data to {}'.format(out_path))
        site_df.to_csv(out_path)


if __name__ == '__cli__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running WindX CLI')
        raise
