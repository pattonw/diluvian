# -*- coding: utf-8 -*-


from __future__ import division

import itertools
import logging

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random
import six
from six.moves import queue
from tqdm import tqdm

from .config import CONFIG
from .octrees import OctreeVolume
from .postprocessing import Body
from .util import (
        get_color_shader,
        pad_dims,
        WrappedViewer,
        )


class Skeleton(object):
    """
    TODO: improve this.

    The skeleton object is primarily a collection of regions that can be rendered together.
    """

    def __init__(self, regions):
        self.regions = regions

    def render_skeleton(self):
        from mayavi import mlab

        fig = mlab.figure(size=(1280, 720))

        masks = []
        bound_list = []
        for region in self.regions:
            body = region.to_body()
            mask, bounds = body.get_seeded_component(CONFIG.postprocessing.closing_shape)
            masks.append(mask)
            bound_list.append(bounds)

        for mask in masks:
            grid = mlab.pipeline.scalar_field(mask)
            grid.spacing = CONFIG.volume.resolution

            mlab.pipeline.iso_surface(grid, color=(random.random(),
                                                   random.random(),
                                                   random.random()),
                                            contours=[0.5], 
                                            opacity=0.6)

        mlab.orientation_axes(figure=fig, xlabel='Z', zlabel='X')
        mlab.view(azimuth=45, elevation=30, focalpoint='auto', roll=90, figure=fig)
        mlab.show()

    def fill_render(self, model, save_movie=True, **kwargs):
        from mayavi import mlab

        body = self.to_body()
        mask = body.mask

        fig = mlab.figure(size=(1280, 720))

        if self.target is not None:
            target_grid = mlab.pipeline.scalar_field(np.transpose(self.target))
            target_grid.spacing = np.flipud(CONFIG.volume.resolution)

            target_grid = mlab.pipeline.iso_surface(target_grid, contours=[0.5], color=(1, 0, 0), opacity=0.1)

        grid = mlab.pipeline.scalar_field(np.transpose(mask.astype(np.int32)))
        grid.spacing = np.flipud(CONFIG.volume.resolution)

        contour = mlab.pipeline.iso_surface(grid, color=(0, 1, 0), contours=[0.5], opacity=0.6)
        contour.actor.property.backface_culling = True
        grid = contour.mlab_source

        mlab.orientation_axes(figure=fig)
        mlab.view(azimuth=45, elevation=60, focalpoint='auto', figure=fig)

        fill_generator = self.fill(model, generator=True, **kwargs)

        FRAMES_PER_MOVE = 2
        FPS = 60.0
        ORBIT_RATE = 0.125

        @mlab.animate(delay=int(1000.0/FPS), ui=True)
        def animate():
            try:
                for _, _ in fill_generator:
                    body = self.to_body()
                    mask = body.mask
                    grid.set(scalars=np.transpose(mask.astype(np.int32)))

                    for _ in range(FRAMES_PER_MOVE):
                        view = list(mlab.view(figure=fig))
                        view[0] = (view[0] + ORBIT_RATE * 360.0 / FPS) % 360.0
                        mlab.view(azimuth=view[0], elevation=view[1], focalpoint='auto')
                        fig.scene.render()
                        # fig.scene.movie_maker.animation_step()
                        yield
            except Region.EarlyFillTermination:
                pass
            fig.scene.movie_maker.record = False
            fig.scene.movie_maker.animation_stop()

        if save_movie:
            fig.scene.movie_maker.record = True
        a = animate()  # noqa

        mlab.show()


