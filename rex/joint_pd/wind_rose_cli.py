# -*- coding: utf-8 -*-
"""
Wind Rose Command Line Interface (CLI)
"""
import click
import logging
import os

from rex.joint_pd.joint_pd import JointPD
from rex.multi_file_resource import MultiFileWTK
from rex.multi_year_resource import MultiYearWindResource
from rex.renewable_resource import WindResource
from rex.utilities.cli_dtypes import INT
from rex.utilities.loggers import init_mult
from rex import __version__

logger = logging.getLogger(__name__)

RES_CLS = {'Wind': WindResource,
           'MultiFile': MultiFileWTK,
           'MultiYear': MultiYearWindResource}


@click.command()
@click.version_option(version=__version__)
@click.option('--wind_path', '-h5', required=True,
              type=click.Path(),
              help=('Path to wind resource .h5 files'))
@click.option('--hub_height', '-height', required=True,
              help='Hub-height at which to compute wind rose')
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--wspd_bins', '-wspd', nargs=3, type=int, default=(0, 30, 1),
              show_default=True,
              help='(start, stop, step) for wind speed bins')
@click.option('--wdir_bins', '-dir', nargs=3, type=int, default=(0, 360, 5),
              show_default=True,
              help='(start, stop, step) for wind direction bins')
@click.option('--max_workers', '-mw', type=INT, default=None,
              show_default=True,
              help=('Number of workers to use, if 1 run in serial, if None use'
                    ' all available cores'))
@click.option('--res_cls', '-res',
              type=click.Choice(['Wind', 'MultiFile', 'MultiYear'],
                                case_sensitive=False),
              default='Wind', show_default=True,
              help='Resource Handler')
@click.option('--hsds', '-hsds', is_flag=True,
              help=("Boolean flag to use h5pyd to handle .h5 'files' hosted "
                    "on AWS behind HSDS"))
@click.option('--chunks_per_worker', '-cpw', default=5, type=int,
              help='Number of chunks to extract on each worker')
@click.option('--log_file', '-log', default=None, type=click.Path(),
              show_default=True,
              help='Path to .log file, if None only log to stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
def main(wind_path, hub_height, out_dir, wspd_bins, wdir_bins, max_workers,
         res_cls, hsds, chunks_per_worker, log_file, verbose):
    """
    WindRose Command Line Interface
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    name = os.path.splitext(os.path.basename(wind_path))[0]
    name = name.replace('*', '')
    out_fpath = '{}_wind_rose-{}m.csv'.format(name, hub_height)
    out_fpath = os.path.join(out_dir, out_fpath)

    if log_file is not None:
        name = os.path.basename(log_file).split('.')[0]
        log_dir = os.path.dirname(log_file)
    else:
        name = None
        log_dir = None

    init_mult(name, log_dir, [__name__, 'rex'], verbose=verbose)
    logger.info('Computing wind rose from {}'.format(wind_path))
    logger.info('Outputs to be stored in: {}'.format(out_dir))

    JointPD.wind_rose(wind_path, hub_height, wspd_bins=wspd_bins,
                      wdir_bins=wdir_bins, sites=None,
                      res_cls=RES_CLS[res_cls], hsds=hsds,
                      max_workers=max_workers,
                      chunks_per_worker=chunks_per_worker,
                      out_fpath=out_fpath)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception('Error running WindRose CLI')
        raise
