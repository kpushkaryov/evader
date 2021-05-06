# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Aircraft without evasion demonstration."""
from evader.math import float_array
import evader.controller
import evader.examples.optimal
import evader.objects


def get_model_objects(args=dict()):
    """Demonstrates an aircraft not evading missiles.

    The aircraft tries to reach a target, while two missile systems try
    to shoot it down.

    Args:
        args: Optional; Arguments dictionary. Currently ignored.
    """
    air_contr = evader.controller.AircraftControllerEvadingTargeted(
        aircraft=None,
        target=float_array([50, 0]),
    )
    return evader.examples.optimal.make_model_objects(air_contr)
