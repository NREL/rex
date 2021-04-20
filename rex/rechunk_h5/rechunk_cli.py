# -*- coding: utf-8 -*-
"""
Rechunk h5 command line interface
"""
import click
import logging
import os

from rex.rechunk_h5.rechunk_h5 import RechunkH5
from rex.utilities.cli_dtypes import INT, STR
from rex.utilities.loggers import init_logger
from rex import __version__

logger = logging.getLogger(__name__)


@click.command()
@click.version_option(version=__version__)
@click.option('--src_h5', '-src', type=click.Path(exists=True), required=True,
              help="Source .h5 file path")
@click.option('--dst_h5', '-dst', type=click.Path(), required=True,
              help="Destination path for rechunked .h5 file")
@click.option('--var_attrs_path', '-vap', type=click.Path(exists=True),
              default=None, show_default=True,
              help=".json containing variable attributes")
@click.option('--hub_height', '-hgt', type=INT, default=None,
              show_default=True,
              help="Rechunk specific hub_height")
@click.option('--chunk_size', '-size', default=2, type=int, show_default=True,
              help="Chunk size in MB")
@click.option('--weeks_per_chunk', '-wpc', default=None, type=INT,
              show_default=True,
              help=("Number of weeks per time chunk, if None scale weeks "
                    "based on 8 weeks for hourly data"))
@click.option('--overwrite', '-rm', is_flag=True,
              help="Flag to overwrite an existing h5_dst file")
@click.option('--meta', '-m', default=None, type=click.Path(exists=True),
              show_default=True,
              help=("Path to .csv or .npy file containing meta to load into "
                    "rechunked .h5 file"))
@click.option('--process_size', '-s', default=None, type=INT,
              show_default=True,
              help="Size of each chunk to be processed")
@click.option('--check_dset_attrs', '-cda', is_flag=True,
              help='Flag to compare source and specified dataset attributes')
@click.option('--resolution', '-res', default=None, type=STR,
              show_default=True,
              help='New time resolution')
@click.option('--log_file', '-log', default=None, type=click.Path(),
              show_default=True,
              help='Path to .log file, if None only log to stdout')
@click.option('--verbose', '-v', is_flag=True,
              help='If used upgrade logging to DEBUG')
def main(src_h5, dst_h5, var_attrs_path, hub_height, chunk_size,
         weeks_per_chunk, overwrite, meta, process_size, check_dset_attrs,
         resolution, log_file, verbose):
    """
    RechunkH5 CLI entry point
    """
    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    init_logger('rex', log_file=log_file, log_level=log_level)

    dst_dir = os.path.dirname(dst_h5)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    RechunkH5.run(src_h5, dst_h5, var_attrs=var_attrs_path,
                  hub_height=hub_height, chunk_size=chunk_size,
                  weeks_per_chunk=weeks_per_chunk, overwrite=overwrite,
                  meta=meta, process_size=process_size,
                  check_dset_attrs=check_dset_attrs, resolution=resolution)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running Combined H5')
        raise
