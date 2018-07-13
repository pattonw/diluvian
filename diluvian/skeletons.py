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
    skeleton should have a tree structure in its regions so that it can efficiently find regions
    to compare for intersections, and wont highlight all branches as missing.
    """

    def __init__(self, regions = None):
        self.regions = regions
        self.tree = self.SkeletonTree()

    def gather_info(self):
        if self.build_tree(self.regions):
            print(self.tree.dump_tree())
        self.start, self.stop = self.get_skeleton_bounds()
        self.masks = list(self.get_masks())
        self.intersections = self.get_intersections()

    def build_tree(self, regions):
        missed_regions = []
        for region in regions:
            regionNode = self.RegionNode(region)
            if not self.tree.addRegion(regionNode):
                missed_regions.append(region)
        if len(missed_regions) == 0:
            return True
        elif len(missed_regions) == len(regions):
            raise Exception("Skeleton contains disconnected components")
        else:
            return build_tree(missed_regions)

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

            final_center = np.copy(final_mask)
            s = [x//2 for x in mask.shape]
            final_center[tuple(np.array(bounds.start)-np.array(self.start)+np.array(s))] = 1

            final_mask[list(map(slice,
                                np.array(bounds.start) - np.array(self.start),
                                np.array(bounds.stop) - np.array(self.start)))] = mask
            yield [final_mask, final_center]

    def get_intersections(self):
        past_mask = None
        for mask in self.masks:
            mask = mask[0]
            if past_mask is not None:
                intersection = np.floor_divide((mask + past_mask),2)
                yield intersection
            else:
                past_mask = mask

    def render_skeleton(self):
        from mayavi import mlab
        self.gather_info()

        fig = mlab.figure(size=(1280, 720))

        for mask in self.masks:
            grid = mlab.pipeline.scalar_field(mask[0])
            grid.spacing = CONFIG.volume.resolution
            center_grid = mlab.pipeline.scalar_field(mask[1])
            center_grid.spacing = CONFIG.volume.resolution

            colors = (random.random(), random.random(), random.random())
            mlab.pipeline.iso_surface(grid, color=colors,
                                            contours=[0.5], 
                                            opacity=0.1)
            mlab.pipeline.iso_surface(center_grid, color=colors,
                                            contours=[0.5], 
                                            opacity=1)

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

    class SkeletonTree():
        def __init__(self):
            self.root = None

        def addRegion(self, node):
            print("adding region")
            print(self.root)
            if self.root is None:
                print("root is none")
                self.root = node
                return True
            elif self.root.isChild(node):
                print("parent of root")
                node.appendChild(root)
                self.root = node
                return True
            else:
                print("child of root")
                return self.root.appendChild(node)

        def dump_tree(self):
            return str(self.root)
            

    class RegionNode():
        def __init__(self, region, children = None):
            self.region = region
            self.id = region.orig_bounds.node_id[0]
            self.pid = region.orig_bounds.node_id[1]
            if children:
                self.children = children
            else:
                self.children = []
        
        def appendChild(self, regionNode):
            print(self.id, self.pid)
            print(regionNode.id, regionNode.pid)
            if self.isParent(regionNode):
                self.children.append(regionNode)
                return True
            else:
                for child in self.children:
                    if child.appendChild(regionNode):
                        return True
                return False
        
        def isEqual(self, node):
            return node.id == self.id

        def isParent(self, node):
            return self.id == node.pid
        
        def isChild(self, node):
            return node.id == self.pid

        def hasId(self, id):
            return self.id == id

        def hasPid(self, pid):
            return self.pid == pid

        def searchId(self, id):
            if self.hasId(id):
                return self
            else:
                for child in self.children:
                    found = child.searchId(id)
                    if found is not None:
                        return found
                return None
        
        def __str__(self):
            if len(self.children) > 0:
                return str(self.id) + "[" + ','.join([str(x) for x in self.children]) + ']'
            else:
                return str(self.id)



