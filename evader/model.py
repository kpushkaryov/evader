# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Missile evader world model."""
import logging

import matplotlib.pyplot as plt


module_logger = logging.getLogger(__name__)


class Model:
    """Missile evader world model.

    The class contains the main modeling loop. Current world state is displayed
    graphically using :mod:`matplotlib` library.

    Args:
        xlb: Lower bounds of the world coordinates.
        xub: Upper bounds of the world coordinates.
        fig: Figure for graphics output.
        ax: Axes for graphics output.
        logger: Logger object.
    """
    def __init__(
            self,
            xlb,
            xub,
            fig,
            ax,
            logger=None
    ):
        """Initialize."""
        self.xlb = xlb
        self.xub = xub
        self.fig = fig
        self.ax = ax
        self.objects = []
        if logger is None:
            logger = module_logger
        self.logger = logger.getChild(__class__.__name__)
        self.paused = False
        self.exit = False
        self._key_press_conn_id = None
        self.pause_keys = [' ']
        self.exit_keys = ['escape']

    def add_object(self, obj):
        """Adds an object to the model."""
        obj.model = self
        self.objects.append(obj)

    def remove_object(self, obj):
        """Removes an object from the model."""
        self.objects.remove(obj)

    def find_objects_by_type(self, type_):
        """Finds objects of specified type."""
        return (obj for obj in self.objects if isinstance(obj, type_))

    def _on_key_press(self, event):
        """Key press event handler."""
        self.logger.debug(f'Key pressed: {event.key!r}')
        if event.key in self.pause_keys:
            self.logger.info('Pause key pressed')
            self.paused = not self.paused
        elif event.key in self.exit_keys:
            self.logger.info('Exit key pressed')
            self.exit = True

    def run(
            self,
            tmax,
            dt,
            frame_time
    ):
        """Run the model.

        Args:
            tmax: Modeling ending time.
            dt: Modeling time step.
            frame_time: Frame display time.

        Returns:
            Final model time.
        """
        self._key_press_conn_id = self.fig.canvas.mpl_connect(
            'key_press_event',
            self._on_key_press
        )
        t = 0
        while not self.exit and t <= tmax:
            if not self.paused:
                title = f't = {t:.2f}'
                self.logger.debug(title)
                self.ax.set_title(title)
                t += dt
                for obj in self.objects:
                    obj.erase()
                    obj.draw(self.ax)
                    obj.next(t, dt)
            plt.pause(frame_time)
        self.fig.canvas.mpl_disconnect(self._key_press_conn_id)
        return t
