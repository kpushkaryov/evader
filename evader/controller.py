# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Model object controlling facilities."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import enum
import numpy as np
import scipy.optimize

from evader.math import abs_dir_clip, dist
import evader.common


class IModelObjectController:
    """Interface of a model object controller."""

    @abstractmethod
    def next(
            self,
            t: float,
            dt: float
    ) -> None:
        """Performs the next modeling step.

        Args:
            t: Current model time.
            dt: Model time step.
        """
        raise NotImplementedError

    @abstractmethod
    def draw(
            self,
            ax: Any
    ) -> None:
        """Draws the object.

        Args:
            ax: Axes object to draw in.
        """
        raise NotImplementedError

    @abstractmethod
    def erase(self) -> None:
        """Erases the object."""
        raise NotImplementedError


class EvadeResult(enum.Enum):
    """Results of :meth:`AircraftMissileEvadingTargetController.evade_missile`."""
    SUCCESS = enum.auto()
    FAIL = enum.auto()
    NO_THREAT = enum.auto()


class AircraftControllerEvadingTargeted(IModelObjectController):
    """Aircraft controller that can evade missiles and has a target.

    Tries to reach the target and evade missiles at the same
    time. Override :meth:`evade_missile` to perform an evasion maneuver.

    Args:
        aircraft: Controlled aircraft.
        target: Target position vector.

    """

    def __init__(
            self,
            aircraft,
            target
    ):
        """Initialize."""
        self.aircraft = aircraft
        self.target = target
        self._artists = []

    @property
    def logger(self):
        """Returns the aircraft logger."""
        return self.aircraft.logger

    def evade_missile(
            self,
            t: float,
            dt: float,
            missile
    ):
        """Evades a missile.

        This method is called when the aircraft detects a
        missile. Override it to perform an evasion maneuver. By default,
        does nothing and returns NO_THREAT.

        Args:
            t: Current model time.
            dt: Model time step.
            missile: Missile object.

        Returns:
            EvadeResult: Result of evasion.

        """
        return EvadeResult.NO_THREAT

    def next(
            self,
            t: float,
            dt: float
    ):
        """See :meth:`IModelObjectController.next`."""
        if not self.aircraft.destroyed:
            ev_res = None
            if missile := self.aircraft.find_missile():
                self.logger.debug('Missile detected')
                ev_res = self.evade_missile(t, dt, missile)
            if ev_res is None or ev_res is EvadeResult.NO_THREAT:
                if self.target is not None:
                    # This way the distance to the target decreases
                    # exponentially with time (x(t) = (x0 -
                    # x1)*exp(-t) + x1), guaranteeing soft landing
                    self.aircraft.v += abs_dir_clip(
                        self.target - self.aircraft.x - self.aircraft.v,
                        self.aircraft.dvmax
                    )

    def draw(
            self,
            ax
    ):
        """See :meth:`IModelObjectController.draw`."""
        # Draw the target
        self._artists = ax.plot(self.target[0], self.target[1], 'ro')

    def erase(self):
        """See :meth:`IModelObjectController.erase`."""
        evader.common.remove_artists(self._artists)


def tmindist(
        x1: np.ndarray,
        v1: np.ndarray,
        x2: np.ndarray,
        v2: np.ndarray
) -> float:
    """Returns the time of minimum distance between objects.

    Returns the time at which the distance between objects is minimum.

    Args:
        x1, x2: Positions of the objects.
        v1, v2: Velocities of the objects.
    """
    dv = v1 - v2
    dvsq = np.dot(dv, dv)
    if dvsq == 0.0:
        return 0.0
    dx = x1 - x2
    return -np.dot(dv, dx) / dvsq


def sqmindist(
        x1: np.ndarray,
        v1: np.ndarray,
        x2: np.ndarray,
        v2: np.ndarray
) -> float:
    """Returns the *squared* minimum distance between objects.

    Args:
        x1, x2: Positions of the objects.
        v1, v2: Velocities of the objects.
    """
    dv = v1 - v2
    dvsq = np.dot(dv, dv)
    dx = x1 - x2
    if dvsq == 0.0:
        return np.dot(dx, dx)
    return (dv[0]*dx[1] - dv[1]*dx[0])**2 / dvsq


def mindist(
        x1: np.ndarray,
        v1: np.ndarray,
        x2: np.ndarray,
        v2: np.ndarray
) -> float:
    """Returns the minimum distance between objects.

    Args:
        x1, x2: Positions of the objects.
        v1, v2: Velocities of the objects.
    """
    return np.sqrt(sqmindist(x1, v1, x2, v2))


@dataclass
class EvasionSolution:
    """Missile evasion solution."""
    v: np.ndarray
    """Solution velocity."""
    nit: int
    """Number of iterations performed."""
    success: bool
    """Whether the optimization process succeeded."""


class IEvasionSolver(Protocol):
    """Interface for missile evasion problem solvers."""

    @abstractmethod
    def __call__(
            self,
            aircraft,
            missile,
            start_v: np.ndarray,
            t: float,
            dt: float
    ) -> EvasionSolution:
        """Solves a missile evasion problem.

        Args:
            aircraft: Aircraft model object.
            missile: Missile model object.
            start_v: Starting velocity vector.
            t: Model time.
            dt: Model time step.

        Returns:
            EvasionSolution: Solution of the problem.
        """
        raise NotImplementedError


class AircraftControllerOptimalEvasion(AircraftControllerEvadingTargeted):
    """Aircraft controller trying to find an optimal evasion maneuver.

    This class does little itself: it checks if evasion is needed, logs
    some debug information and selects starting points. Then solution of the
    evasion problem is delegated to an evasion solver.

    Args:
        aircraft: Controlled aircraft.
        target: Aircraft target.
        evasion_solver: Callable object solving the evasion problem.
        evasion_vel_coeff: Coefficient determining offset of additional
            starting points (see :meth:`AircraftControllerOptimalEvasion.evade_missile`).

    """

    def __init__(
            self,
            aircraft,
            target,
            evasion_solver: IEvasionSolver,
            evasion_vel_coeff=0.9
    ):
        """Initialize."""
        super().__init__(aircraft, target)
        self.evasion_solver = evasion_solver
        self.evasion_vel_coeff = evasion_vel_coeff

    def evade_missile(
            self,
            t: float,
            dt: float,
            missile
    ):
        """Finds the optimal evasion solution."""
        def fnegsqmindist(v):
            return -sqmindist(
                self.aircraft.x,
                v,
                missile.x,
                missile.v
            )

        def fnegnextdist(v):
            return -dist(self.aircraft.x + dt*v, missile.x + dt*missile.v)

        def ftmin(v):
            return tmindist(self.aircraft.x, v, missile.x, missile.v)

        def fdmin(v):
            return mindist(self.aircraft.x, v, missile.x, missile.v)

        dvmax = self.aircraft.dvmax
        v0 = self.aircraft.v
        tmin = ftmin(v0)
        self.logger.debug(
            'Current minimum prognozed distance to the missile: '
            f'{fdmin(v0)} at t={tmin}'
        )
        if tmin < 0:
            self.logger.debug(
                'Minimum distance is in the past, nothing to do now'
            )
            return EvadeResult.NO_THREAT
        # v0, the current velocity, is the center of the search space
        # (v0 +/- dvmax) and the first starting point.
        #
        # Try multiple starting points, because gradient may be zero
        # at some of them and search will stall.
        #
        # A starting point shouldn't be at the bounds, because that may harm
        # the solution quality. So, use an offset less than dvmax multiplying
        # it by self.evasion_vel_coeff.
        for start in [
                v0,
                v0 - dvmax*self.evasion_vel_coeff,
                v0 + dvmax*self.evasion_vel_coeff
        ]:
            self.logger.debug(
                f'Starting evasion solution search from {start}. '
                'Current minimum prognozed distance to the missile: '
                f'{fdmin(start)} at t={ftmin(start)}'
            )
            res = self.evasion_solver(self.aircraft, missile, start, t, dt)
            self.logger.debug(
                f'Evasion solution: {res}'
            )
            # We also consider the search unsuccessful, if the number
            # of iterations was zero
            if res.success and res.nit > 0:
                self.logger.debug(
                    f'New aircraft velocity: {res.v}. '
                    'New minimum prognozed distance to the missile: '
                    f'{fdmin(res.v)} at t={ftmin(res.v)}'
                )
                self.aircraft.v = res.v
                return EvadeResult.SUCCESS
        self.logger.debug('No viable evasion solution')
        return EvadeResult.FAIL


class EvasionSolverMaxMinDist(IEvasionSolver):
    """Minimum distance maximizing solver.

    Maximizes the minimum distance to the missile.
    """

    def __call__(
            self,
            aircraft,
            missile,
            start_v: np.ndarray,
            t: float,
            dt: float
    ) -> EvasionSolution:
        """See :meth:`IEvasionSolver.__call__`."""
        def fnegsqmindist(v):
            return -sqmindist(
                aircraft.x,
                v,
                missile.x,
                missile.v
            )

        dvmax = aircraft.dvmax
        v0 = aircraft.v
        res = scipy.optimize.minimize(
            fnegsqmindist,
            start_v,
            bounds=[(x - dvmax, x + dvmax) for x in v0],
        )
        return EvasionSolution(v=res.x, nit=res.nit, success=res.success)


class EvasionSolverMinFuel(IEvasionSolver):
    """Fuel-economizing solver.

    Tries to keep safe distance from the missile and spend minimum
    fuel at the same time. It's assumed that amount of fuel spent is
    proportional to change of velocity.

    Args:
        safe_dist (float): Safe distance to the missile.
    """

    def __init__(
            self,
            safe_dist,
    ):
        """Initialize."""
        self.safe_dist = safe_dist

    def __call__(
            self,
            aircraft,
            missile,
            start_v: np.ndarray,
            t: float,
            dt: float
    ) -> EvasionSolution:
        """See :meth:`IEvasionSolver.__call__`."""
        def ffuel(v):
            return dist(v, aircraft.v)

        def fnegsqmindist(v):
            return -sqmindist(
                aircraft.x,
                v,
                missile.x,
                missile.v
            )

        dvmax = aircraft.dvmax
        v0 = aircraft.v
        sq_safe_dist = self.safe_dist**2
        res = scipy.optimize.minimize(
            ffuel,
            start_v,
            bounds=[(x - dvmax, x + dvmax) for x in v0],
            constraints={
                'type': 'ineq',
                'fun': lambda v: -fnegsqmindist(v) - sq_safe_dist
            }
        )
        return EvasionSolution(v=res.x, nit=res.nit, success=res.success)


class EvasionSolverMaxNextDist(IEvasionSolver):
    """Next distance maximizing solver.

    Maximizes the distance to the missile at the next modeling step.

    Args:
        metric: Function returning the distance between two vectors.
    """

    def __init__(
            self,
            metric=dist,
    ):
        """Initialize."""
        self.metric = metric

    def __call__(
            self,
            aircraft,
            missile,
            start_v: np.ndarray,
            t: float,
            dt: float
    ) -> EvasionSolution:
        """See :meth:`IEvasionSolver.__call__`."""
        def fnegnextdist(v):
            return -self.metric(
                aircraft.x + dt*v, missile.x + dt*missile.v
            )

        dvmax = aircraft.dvmax
        v0 = aircraft.v
        res = scipy.optimize.minimize(
                fnegnextdist,
                start_v,
                bounds=[(x - dvmax, x + dvmax) for x in v0],
            )
        return EvasionSolution(v=res.x, nit=res.nit, success=res.success)
