# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Optimal evasion example module."""
from evader.math import float_array
import evader.controller
import evader.objects


def make_model_objects(air_contr):
    """Demonstrates an aircraft evading missiles.

    The aircraft tries to reach the target, while two missile systems try
    to shoot it down.

    Args:
        air_contr: Aircraft controller.
    """
    return [
        evader.objects.Aircraft(
            x=float_array([25, 90]),
            v=float_array([0, 0]),
            vmax=20,
            dvmax=15,
            controller=air_contr
        ),
        evader.objects.MissileSystem(
            x=float_array([75, 0]),
            v=float_array([0, 0]),
            missile_vmax=50,
            expl_range=5,
            rate_of_fire=2,
            firing_range=50,
            max_firing_angle=1.5,
            missile_factory=evader.objects.UnguidedMissile,
            name='MissileSystem1'
        ),
        evader.objects.MissileSystem(
            x=float_array([25, 0]),
            v=float_array([0, 0]),
            missile_vmax=50,
            expl_range=5,
            rate_of_fire=2,
            firing_range=50,
            max_firing_angle=1.5,
            missile_factory=evader.objects.UnguidedMissile,
            name='MissileSystem2'
        )
    ]


def get_model_objects(args=dict()):
    """Demonstrates an aircraft evading missiles.

    The aircraft tries to reach the target, while two missile systems try
    to shoot it down.

    Args:
        args: Optional; Arguments dictionary. Currently valid keys:

            * solver_name: Name of an evasion solver from the `evader.controller` module.
            * solver_args: Dictionary of arguments to be passed to the evasion solver.
    """
    solver_name = args.get('solver_name', 'EvasionSolverMaxNextDist')
    solver_factory = getattr(evader.controller, solver_name)
    solver_args = args.get('solver_args', dict())
    solver = solver_factory(**solver_args)
    air_contr = evader.controller.AircraftControllerOptimalEvasion(
        aircraft=None,
        target=float_array([50, 0]),
        evasion_solver=solver
    )
    return make_model_objects(air_contr)
