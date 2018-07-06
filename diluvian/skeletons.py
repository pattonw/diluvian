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
        self.start, self.stop = self.get_skeleton_bounds()
        self.masks = list(self.get_masks())
        self.intersections = self.get_intersections()

    def get_skeleton_bounds(self):
        start = [float("inf"),float("inf"),float("inf")]
        stop = [0,0,0]
        for region in self.regions:
            start = [min(start[i], region.orig_bounds.start[i]) for i in range(3)]
            stop = [max(stop[i], region.orig_bounds.stop[i]) for i in range(3)]
        return start, stop

    def get_masks(self):
        for region in self.regions:
            final_mask = np.zeros(np.array(self.stop)-np.array(self.start))
            body = region.to_body()
            mask, _ = body.get_seeded_component(CONFIG.postprocessing.closing_shape)
            bounds = region.orig_bounds
            final_mask[list(map(slice,
                                np.array(bounds.start) - np.array(self.start),
                                np.array(bounds.stop) - np.array(self.start)))] = mask
            yield final_mask

    def get_intersections(self):
        past_mask = None
        for mask in self.masks:
            if past_mask is not None:
                intersection = np.floor_divide((mask + past_mask),2)
                yield intersection
            else:
                past_mask = mask

    def render_skeleton(self):
        from mayavi import mlab

        fig = mlab.figure(size=(1280, 720))

        for mask in self.masks:
            grid = mlab.pipeline.scalar_field(mask)
            grid.spacing = CONFIG.volume.resolution

            mlab.pipeline.iso_surface(grid, color=(random.random(),
                                                   random.random(),
                                                   random.random()),
                                            contours=[0.5], 
                                            opacity=0.2)

        mlab.orientation_axes(figure=fig, xlabel='Z', zlabel='X')
        mlab.view(azimuth=45, elevation=30, focalpoint='auto', roll=90, figure=fig)


        fig2 = mlab.figure(size=(1280, 720))

        for intersection in self.intersections:
            grid = mlab.pipeline.scalar_field(intersection)
            grid.spacing = CONFIG.volume.resolution

            mlab.pipeline.iso_surface(grid, color = (random.random(),
                                                     random.random(),
                                                     random.random()), contours = [0.5], opacity = 0.2)

        mlab.orientation_axes(figure=fig2, xlabel='Z', zlabel='X')
        mlab.view(azimuth=45, elevation=30, focalpoint='auto', roll=90, figure=fig2)
        mlab.show()


