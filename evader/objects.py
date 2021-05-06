# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Aircraft model object."""
import logging

import matplotlib.patches as mpatches
import numpy as np

from evader.common import remove_artists
from evader.math import dist
import evader.baseobjects


module_logger = logging.getLogger(__name__)


class Aircraft(evader.baseobjects.ModelObjectBase):
    """Aircraft model object.

    Models a controllable aircraft in the model.

    Args:
        x: Position vector.
        v: Velocity vector.
        vmax: Maximum speed.
        dvmax: Maximum change of a velocity component per unit of time.
        controller: Controller object.
        name: Optional; Object name.
        logger: Optional; Logger object.
        model: Optional; Parent model.
    """

    def __init__(
            self,
            x,
            v,
            vmax,
            dvmax,
            controller,
            name='Aircraft',
            logger=None,
            model=None
    ):
        """Initialize."""
        super().__init__(model, name, logger, x, v)
        self.vmax = vmax
        self.dvmax = dvmax
        self.controller = controller
        self.destroyed = False
        self._artists = []
        self.controller.aircraft = self

    def _str_dict(self):
        return dict(super()._str_dict(), destroyed=self.destroyed)

    def find_missile(self):
        """Returns a dangerous missile or None.

        Returns a nearest missile in :attr:`self.model`, which is in valid
        state and not destroyed.
        """
        cand = []
        for obj in self.model.find_objects_by_type(evader.baseobjects.IMissile):
            if not obj.destroyed and obj.x is not None:
                cand.append(obj)
        if cand:
            return min(cand, key=lambda mis: dist(self.x, mis.x))
        return None

    def next(
            self,
            t,
            dt
    ):
        """See :meth:`IModelObject.next`."""
        self.controller.next(t, dt)
        self.v = evader.math.abs_dir_clip(self.v, self.vmax)
        self.x += self.v * dt
        self.logger.debug(f'Aircraft state: {self}')

    def draw(
            self,
            ax
    ):
        """See :meth:`IModelObject.draw`."""
        style = 'k^' if self.destroyed else 'b^'
        self._artists = ax.plot(self.x[0], self.x[1], style)
        self.controller.draw(ax)

    def erase(self):
        """See :meth:`IModelObject.erase`."""
        remove_artists(self._artists)
        self.controller.erase()

    def destroy(self):
        """Destroy the aircraft.

        Destruction sets the velocity to zero and :attr:`self.destroyed` to
        True.
        """
        self.logger.debug('Destroyed')
        self.v = np.zeros_like(self.v)
        self.destroyed = True


class UnguidedMissile(
        evader.baseobjects.ModelObjectBase,
        evader.baseobjects.IMissile
):
    """Unguided missile model object.

    Models an unguided missile with a proximity fuze in the model. Missile
    explodes when distance to the target is less than :attr:`self.expl_range`
    and destroys the target.

    Missile self-destructs, if :meth:`UnguidedMissile.self_destruct` is called.

    Args:
        x: Position vector.
        v: Velocity vector.
        vmax: Maximum missile speed.
        expl_range: explosion range.
        target: Target object.
        owner (MissileSystem): Who launched the missile.
        name: Optional; Object name.
        logger: Optional; Logger object.
        model: Optional; Parent model.
    """
    destroyed: bool = False

    def __init__(
            self,
            x,
            v,
            vmax,
            expl_range,
            target,
            owner,
            name='UnguidedMissile',
            logger=None,
            model=None
    ):
        """Initialize."""
        super().__init__(model, name, logger, x, v)
        self.vmax = vmax
        self.explosion_range = expl_range
        self.target = target
        self.owner = owner
        self.exploded = False
        self.destroyed = False
        self._artists = []

    def _str_dict(self):
        return dict(
            super()._str_dict(),
            destroyed=self.destroyed,
            exploded=self.exploded
        )

    def next(
            self,
            t,
            dt
    ):
        """See :meth:`IModelObject.next`."""
        targ_dist = dist(self.x, self.target.x)
        self.logger.debug(
            f'Missile state: {self}, distance to target: {targ_dist}'
        )
        if not self.destroyed and targ_dist <= self.explosion_range:
            self.logger.debug('Missile exploded')
            self.explode()
            self.target.destroy()
        self.x += self.v * dt
        if (
                not self.destroyed
                and (
                    (self.x < self.model.xlb).any()
                    or (self.x > self.model.xub).any()
                )
        ):
            self.logger.debug(
                'Missile has left the theater of war, self-destruct'
            )
            self.self_destruct()

    def draw(
            self,
            ax
    ):
        """See :meth:`IModelObject.draw`."""
        if self.exploded:
            self._artists = [
                ax.add_patch(mpatches.Circle(self.x, self.explosion_range))
            ]
        elif self.destroyed:
            self._artists = ax.plot(self.x[0], self.x[1], 'k*')
        else:
            self._artists = ax.plot(self.x[0], self.x[1], 'r*')

    def erase(self):
        """See :meth:`IModelObject.erase`."""
        remove_artists(self._artists)

    def explode(self):
        """Explodes the missile.

        Explosion destroys the missile and sets :attr:`self.exploded` to True.
        """
        self.logger.debug('Exploded')
        self.exploded = True
        self.destroy()

    def destroy(self):
        """Destroys the missile.

        Destruction sets the velocity to zero and :attr:`self.destroyed` to
        True.
        """
        self.logger.debug('Destroyed')
        self.v = np.zeros_like(self.v)
        self.destroyed = True

    def self_destruct(self):
        """Self-destructs the missile."""
        self.logger.debug('Self-destruct activated')
        self.destroy()


class MissileSystem(evader.baseobjects.ModelObjectBase):
    """Missile system model object.

    Models a missile system in the model.

    Args:
        x: Position vector.
        v: Velocity vector.
        missile_vmax: Maximum missile speed.
        expl_range: Missile explosion range.
        rate_of_fire: Cooldown time between launches.
        firing_range: Maximum distance to target.
        max_firing_angle: Maximum absolute launch angle from the vertical.
        missile_factory: Factory function making missiles.
        name: Optional; Object name.
        logger: Optional; Logger object.
        model: Optional; Parent model.
    """
    def __init__(
            self,
            x,
            v,
            missile_vmax,
            expl_range,
            rate_of_fire,
            firing_range,
            max_firing_angle,
            missile_factory,
            name='MissileSystem',
            logger=None,
            model=None
    ):
        """Initialize."""
        super().__init__(model, name, logger, x, v)
        self.missile_vmax = missile_vmax
        self.explosion_range = expl_range
        self.rate_of_fire = rate_of_fire
        self.firing_range = firing_range
        self.max_firing_angle = max_firing_angle
        self.missile_factory = missile_factory
        self.missile = None
        self._artists = []
        self.last_fire_time = None
        self.fired_missile_count = 0

    def _str_items(self):
        return dict(super()._str_dict(), last_fire_time=self.last_fire_time)

    def find_target(self):
        """Returns an aircraft to attack or None."""
        for obj in self.model.find_objects_by_type(Aircraft):
            if (
                    not obj.destroyed
                    and obj.x is not None
                    and dist(self.x, obj.x) <= self.firing_range
            ):
                return obj
        return None

    def firing_angle(self, direction):
        """Returns the firing angle for specified direction vector."""
        return evader.math.vec_angle(np.array([0, 1]), direction)

    def next(
            self,
            t,
            dt
    ):
        """See :meth:`IModelObject.next`."""
        if self.missile is not None:
            if dist(self.x, self.missile.x) > self.firing_range:
                self.logger.debug(
                    'Missile has left the firing range, self-destruct'
                )
                self.missile.self_destruct()
            if self.missile.destroyed:
                self.missile = None
        if self.missile is None:
            if (
                    (self.last_fire_time is None)
                    or (t >= self.last_fire_time + self.rate_of_fire)
            ):
                target = self.find_target()
                if target:
                    if self.fire(target):
                        self.last_fire_time = t
                else:
                    self.logger.debug('No target')
            else:
                self.logger.debug('Not firing due to cooldown')
        self.x += self.v * dt
        self.logger.debug(f'Missile system state: {self}')

    def fire(
            self,
            target
    ):
        """Fires a missile at a target.

        Fires a missile so that it can meet the target. Solves some equations
        to calculate the launch direction.

        Args:
            target: The target object.

        Returns:
            Missile object if fired or None.
        """
        self.logger.debug(
            f'Firing at the target [{target.x}, {target.v}] from {self.x}'
        )
        sol = evader.math.firing_direction(
            gun_x=self.x,
            gun_v=None,
            proj_speed=self.missile_vmax,
            targ_x=target.x,
            targ_v=target.v,
            logger=self.logger
        )
        if not sol:
            self.logger.debug('No viable firing solutions')
            return None
        self.logger.debug(f'Selected firing solution: {sol}')
        fang = self.firing_angle(sol[1])
        self.logger.debug(f'Current firing angle: {fang}')
        if fang > self.max_firing_angle:
            self.logger.debug(
                f'Firing angle is too big ({fang} > {self.max_firing_angle})'
            )
            return None
        self.missile = self.missile_factory(
            x=self.x,
            v=sol[1],
            vmax=self.missile_vmax,
            expl_range=self.explosion_range,
            target=target,
            owner=self,
            name=f'Missile{self.fired_missile_count}',
            logger=self.logger
        )
        self.fired_missile_count += 1
        self.model.add_object(self.missile)
        return self.missile

    def draw(
            self,
            ax
    ):
        """See :meth:`IModelObject.draw`."""
        self._artists = ax.plot(self.x[0], self.x[1], 'gs')

    def erase(self):
        """See :meth:`IModelObject.erase`."""
        remove_artists(self._artists)
