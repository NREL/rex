# -*- coding: utf-8 -*-
"""
Rechunk h5 command line interface
"""
import click
import os

from rex.rechunk_h5.rechunk_h5 import RechunkH5
from rex.utilities.cli_dtypes import INT, STR
from rex.utilities.loggers import init_logger


@click.command()
@click.option('--src_h5', '-src', type=click.Path(exists=True), required=True,
              help="Source .h5 file path")
@click.option('--dst_h5', '-dst', type=click.Path(), required=True,
              help="Destination path for rechunked .h5 file")
@click.option('--var_attrs_path', '-vap', type=click.Path(exists=True),
              required=True,
              help=".json containing variable attributes")
@click.option('--hub_height', '-hgt', type=INT, default=None,
              help="Rechunk specific hub_height")
@click.option('--version', '-ver', default=None,
              help="File version number")
@click.option('--overwrite', '-rm', is_flag=True,
              help="Flag to overwrite an existing h5_dst file")
@click.option('--meta', '-m', default=None, type=click.Path(exists=True),
              help=("Path to .csv or .npy file containing meta to load into "
                    "rechunked .h5 file"))
@click.option('--process_size', '-s', default=None, type=INT,
              help="Size of each chunk to be processed")
@click.option('--check_dset_attrs', '-cda', is_flag=True,
              help='Flag to compare source and specified dataset attributes')
@click.option('--resolution', '-res', default=None, type=STR,
              help='New time resolution')
@click.option('--log_file', '-log', default=None, type=click.Path(),
              help='Path to .log file')
@click.option('--verbose', '-v', is_flag=True,
              help='If used upgrade logging to DEBUG')
def main(src_h5, dst_h5, var_attrs_path, hub_height, version, overwrite,
         meta, process_size, check_dset_attrs, resolution, log_file, verbose):
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

    init_logger('rex.rechunk_h5.rechunk_h5', log_file=log_file,
                log_level=log_level)

    dst_dir = os.path.dirname(dst_h5)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    RechunkH5.run(src_h5, dst_h5, var_attrs_path, hub_height=hub_height,
                  version=version, overwrite=overwrite, meta=meta,
                  process_size=process_size, check_dset_attrs=check_dset_attrs,
                  resolution=resolution)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        raise
