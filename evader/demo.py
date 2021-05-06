# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Missile evader demonstration examples runner."""
import matplotlib.pyplot as plt

from evader.math import float_array
import evader.model


def run(
        xlim,
        ylim,
        objects,
        tmax,
        dt,
        frame_time
):
    """Constructs `evader.model.Model` and runs it with the specified parameters.

    Args:
        xlim: X axis coordinate limits.
        ylim: Y axis coordinate limits.
        objects: Objects to add to the model.
        tmax: Modeling ending time.
        dt: Modeling time step.
        frame_time: Frame display time.
    """
    fig, ax = plt.subplots(
        subplot_kw={'xlim': xlim, 'ylim': ylim, 'aspect': 1}
    )
    xlb = float_array([xlim[0], ylim[0]])
    xub = float_array([xlim[1], ylim[1]])
    model = evader.model.Model(xlb, xub, fig, ax)

    for obj in objects:
        model.add_object(obj)

    model.run(tmax, dt, frame_time)

    return model
