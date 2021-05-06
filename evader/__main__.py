# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Runs a demonstration of the evader package."""
import argparse
import ast
import importlib
import logging

import matplotlib.pyplot as plt

import evader.controller
import evader.demo
import evader.objects

module_logger = logging.getLogger(__name__)


def parse_axis_limits(s: str):
    """Parses axis limits string.

    The string must contain a tuple of two numbers as a Python literal.
    """
    return ast.literal_eval(s)


def parse_model_objects(s: str):
    """Parses model objects string.

    The string must contain the name of an importable module ''mod''.
    The module is imported and ''mod.get_model_objects()'' is called to
    get a list of model objects.
    """
    mod = importlib.import_module(s)
    return mod.get_model_objects


def parse_objects_args(s: str):
    """Parses arguments dictionary for model objects factory.

    The string must contain a Python dictionary literal.
    """
    res = ast.literal_eval(s)
    if not isinstance(res, dict):
        raise ValueError(
            'Model objects factory arguments must be a dictionary'
        )
    return res


def setup_argparse() -> argparse.ArgumentParser:
    """Sets up command line argument parser and returns it."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose',
        help='Verbosity level',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        default='INFO'
    )
    parser.add_argument(
        '--xlim',
        help='X axis limits tuple, like "(0, 100)"',
        type=parse_axis_limits,
        default=(0, 100)
    )
    parser.add_argument(
        '--ylim',
        help='Y axis limits tuple, like "(0, 100)"',
        type=parse_axis_limits,
        default=(0, 100)
    )
    parser.add_argument(
        '--objects',
        help=(
            'Name of a module containing get_model_objects() function, '
            'which returns a list of model objects'
        ),
        default='evader.examples.default'
    )
    parser.add_argument(
        '--objects-args',
        help=(
            'Dictionary of arguments passed to get_model_objects() function'
        ),
        type=parse_objects_args,
        default=dict()
    )
    parser.add_argument(
        '--tmax',
        help='Maximum model time',
        type=float,
        default=20.0
    )
    parser.add_argument(
        '--dt',
        help='Model time step',
        type=float,
        default=0.05
    )
    parser.add_argument(
        '--frame-time',
        help='Frame display time',
        type=float,
        default=0.05
    )
    parser.add_argument(
        '--matplotlib-style',
        help='Name of a Matplotlib style file',
    )
    return parser


def post_process_args(args: argparse.Namespace) -> argparse.Namespace:
    """Processes complex arguments.

    Processes command-line arguments that require further processing
    after argparse. Arguments needing complex processing and/or error
    reporting should be handled here.
    """
    if args.objects:
        args.objects = parse_model_objects(args.objects)
    return args


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = setup_argparse()
    args = parser.parse_args()
    return post_process_args(args)


def setup_logging(args: argparse.Namespace) -> None:
    """Sets up logging."""
    logging.basicConfig()
    module_logger.setLevel(args.verbose)
    logging.getLogger('evader').setLevel(args.verbose)


def run_demo(args: argparse.Namespace) -> None:
    """Runs a demonstration.

    Args:
        args: Command-line arguments.
    """
    objects = args.objects(args.objects_args)

    if args.matplotlib_style:
        plt.style.use(args.matplotlib_style)

    evader.demo.run(
        xlim=args.xlim,
        ylim=args.ylim,
        objects=objects,
        tmax=args.tmax,
        dt=args.dt,
        frame_time=args.frame_time
    )


def main() -> None:
    """The main function."""
    args = parse_args()

    setup_logging(args)

    run_demo(args)
    module_logger.info('Modeling finished')
    plt.show(block=True)


if __name__ == '__main__':
    main()
