# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Fuel-economizing aircraft controller demonstration."""
from evader.math import float_array
import evader.controller
import evader.examples.optimal
import evader.objects


def get_model_objects(args=dict()):
    """Demonstrates an aircraft evading missiles.

    The aircraft tries to reach the target, while two missile systems
    try to shoot it down. The aircraft controller minimizes the amount
    of fuel spent to evade the missile.

    Args:
        args: Optional; Arguments dictionary. Currently ignored.
    """
    air_contr = evader.controller.AircraftControllerOptimalEvasion(
        aircraft=None,
        target=float_array([50, 0]),
        evasion_solver=evader.controller.EvasionSolverMinFuel(safe_dist=6)
    )
    return evader.examples.optimal.make_model_objects(air_contr)
