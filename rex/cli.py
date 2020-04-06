# -*- coding: utf-8 -*-
"""
rex command line interface (CLI).
"""
import click

from rex.utilities.cli_dtypes import STR


@click.group()
@click.option('--name', '-n', default='reV', type=STR,
              help='Job name. Default is "reV".')
@click.option('--config_file', '-c',
              required=True, type=click.Path(exists=True),
              help='reV configuration file json for a single module.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, config_file, verbose):
    """reV command line interface."""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['CONFIG_FILE'] = config_file
    ctx.obj['VERBOSE'] = verbose


if __name__ == '__main__':
    main(obj={})
