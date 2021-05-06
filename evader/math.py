# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Mathematical utilities."""
import logging
import math
import numbers
import operator

import numpy as np
import numpy.linalg


def float_array(data, dtype=float, *args, **kwargs):
    r"""Makes an array of floats.

    Args:
        data: Array data.
        dtype: Array data type.
        \*args: Passed to :func:`numpy.array`.
        \**kwargs: Passed to :func:`numpy.array`.

    Returns:
        numpy.ndarray: New array.
    """
    return np.array(data, dtype, *args, **kwargs)


def dist(x, y):
    """Calculates Euclidean distance between two vectors."""
    return np.linalg.norm(x - y)


def cheb_dist(x, y):
    """Calculates Chebyshev distance between two vectors."""
    return np.max(np.abs(x - y))


def abs_dir_clip(x, xmax):
    """Clips absolute values of ``x`` components to ``xmax`` preserving direction.

    Args:
        x (numpy.ndarray): A vector.
        xmax (Union[numbers.Real, float]): A single number or a vector of
            maximum absolute values.

    Returns:
        numpy.ndarray: Vector ``x`` scaled so that absolute values of its
            components are less than or equal to xmax.
    """
    if isinstance(xmax, numbers.Real):
        xmax = [xmax] * len(x)
    k = 1
    for i in range(len(x)):
        if k * abs(x[i]) > xmax[i]:
            k = xmax[i] / abs(x[i])
    return k * x


def vec_angle(v1, v2):
    """Returns the angle between ``v1`` and ``v2`` in radians in [-pi; pi]."""
    return math.atan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))


def firing_direction(
        gun_x,
        gun_v,
        proj_speed,
        targ_x,
        targ_v,
        logger=logging.getLogger(__name__)
):
    """Calculates the direction to fire at a target.

    N. B.: gun_v must be None. Non-stationary guns aren't supported yet.

    Args:
        gun_x: Gun position vector.
        gun_v: Gun velocity vector.
        proj_speed (float): Projectile speed.
        targ_x: Target position vector.
        targ_v: Target velocity vector.
        logger: Optional; Logger object.

    Returns:
        np.ndarray: Fire direction vector.
    """
    assert gun_v is None, 'Non-stationary guns aren\'t supported yet.'
    if logger is None:
        logger = logger
    d1, d2 = targ_x - gun_x
    dsq = d1**2 + d2**2
    if dsq == 0:
        logger.debug('Won''t shoot myself!')
        return None
    vt1, vt2 = targ_v
    dis = proj_speed**2*dsq - (d1*vt2 - d2*vt1)**2
    if dis < 0:
        logger.debug('Unsolvable firing equation')
        return None
    sqrtdis = np.sqrt(dis)
    a = d2**2*vt1 - d1*d2*vt2
    b = d1*(d1*vt2 - d2*vt1)
    c = -d1*vt1 - d2*vt2
    d = (vt1**2 + vt2**2 - proj_speed**2)
    solt1 = (c + sqrtdis) / d
    solv1 = float_array([
        (a - d1*sqrtdis) / dsq,
        (b - d2*sqrtdis) / dsq
    ])
    solt2 = (c - sqrtdis) / d
    solv2 = float_array([
        (a + d1*sqrtdis) / dsq,
        (b + d2*sqrtdis) / dsq
    ])
    solutions = ((solt1, solv1), (solt2, solv2))
    logger.debug(f'Firing solutions: {solutions}]')
    sol = min(
        ((t, v) for t, v in solutions if t > 0),
        key=operator.itemgetter(0),
        default=None
    )
    return sol
