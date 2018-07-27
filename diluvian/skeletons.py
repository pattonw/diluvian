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
from .util import get_color_shader, pad_dims, WrappedViewer


class Skeleton(object):
    """
    TODO: improve this.

    The skeleton object is primarily a collection of regions that can be rendered together.
    skeleton should have a tree structure in its regions so that it can efficiently find regions
    to compare for intersections, and won't highlight all branches as missing.
    """

    def __init__(self):
        self.tree = self.SkeletonTree()
        self.start = [float("inf"), float("inf"), float("inf")]
        self.stop = [0, 0, 0]

    def get_bounds(self):
        return self.start, self.stop

    def is_filled(self, nid):
        for node in self.tree.traverse():
            if node.id == nid:
                return node.body != None
        raise Exception("node {0} not found".format(nid))

    def outline(self, nodes, shape):
        min_seed = [float("inf")] * 3
        max_seed = [0] * 3
        for node in nodes:
            seed = node[2:]
            min_seed = [min(seed[i], min_seed[i]) for i in range(3)]
            max_seed = [max(seed[i], max_seed[i]) for i in range(3)]
            region_node = self.RegionNode(node = node)
            if not self.tree.add_region(region_node):
                raise Exception("region not parent or child of previous regions")
        self.start = [min_seed[i] - shape[i] // 2 for i in range(3)]
        self.stop = [max_seed[i] + shape[i] // 2 + 1 for i in range(3)]

    def get_masks(self, show_seeds=True):
        for node in self.tree.traverse():
            mask = np.zeros(np.array(self.stop) - np.array(self.start))
            node_mask, _ = node.body.get_seeded_component(
                CONFIG.postprocessing.closing_shape
            )
            bounds = node.bounds

            print("node_mask shape: {0}".format(node_mask.shape))
            print("node bounds: {0}".format(bounds.stop - bounds.start))

            mask[
                list(
                    map(
                        slice,
                        np.array(bounds.start) - np.array(self.start),
                        np.array(bounds.stop) - np.array(self.start),
                    )
                )
            ] = node_mask

            if show_seeds:
                node_center = [x // 2 for x in node_mask.shape]
                seed_mask = np.zeros(np.array(self.stop) - np.array(self.start))
                seed_mask[
                    tuple(
                        np.array(bounds.start)
                        - np.array(self.start)
                        + np.array(node_center)
                    )
                ] = 1
            else:
                seed_mask = None

            yield mask, seed_mask

    def add_region(self, region, update_bounds=False):
        """
        Add a region to the tree. Currently this is done by checking if the 
        region being added is a child or a parent of any of the regions in 
        the tree, if not it will throw an error.
        """
        if not self.tree.add_region(self.RegionNode(region)):
            raise Exception("region not parent or child of previous regions")
        elif update_bounds:
            self.start = [
                min(self.start[i], region.orig_bounds.start[i]) for i in range(3)
            ]
            self.stop = [
                max(self.stop[i], region.orig_bounds.stop[i]) for i in range(3)
            ]

    def get_intersections(self):
        for parent in self.tree.traverse():
            for child in parent.children:
                parent_mask, _ = parent.body.get_seeded_component(
                    CONFIG.postprocessing.closing_shape
                )
                child_mask, _ = child.body.get_seeded_component(
                    CONFIG.postprocessing.closing_shape
                )

                int_start = [
                    min(
                        parent.bounds.start[i],
                        child.bounds.start[i],
                    )
                    for i in range(3)
                ]
                int_stop = [
                    max(
                        parent.bounds.stop[i],
                        child.bounds.stop[i],
                    )
                    for i in range(3)
                ]
                int_mask = np.zeros(np.array(int_stop) - np.array(int_start))

                int_mask[
                    list(
                        map(
                            slice,
                            np.array(parent.bounds.start)
                            - np.array(int_start),
                            np.array(parent.bounds.stop)
                            - np.array(int_start),
                        )
                    )
                ] += parent_mask
                int_mask[
                    list(
                        map(
                            slice,
                            np.array(child.bounds.start)
                            - np.array(int_start),
                            np.array(child.bounds.stop)
                            - np.array(int_start),
                        )
                    )
                ] += child_mask
                int_mask = int_mask // 2

                mask = np.zeros(np.array(self.stop) - np.array(self.start))
                mask[
                    list(
                        map(
                            slice,
                            np.array(int_start) - np.array(self.start),
                            np.array(int_stop) - np.array(self.start),
                        )
                    )
                ] = int_mask
                yield mask

    def save_skeleton_masks(self, output_file, show_seeds=True):
        fig = mlab.figure(size=(1280, 720))

        output = []
        for mask, seed_mask in self.get_masks(show_seeds):
            output.append((mask, seed_mask))
        np.save('output_file', output)


    def render_skeleton(self, show_seeds=True):
        from mayavi import mlab

        fig = mlab.figure(size=(1280, 720))

        for mask, seed_mask in self.get_masks(show_seeds):
            grid = mlab.pipeline.scalar_field(mask)
            grid.spacing = CONFIG.volume.resolution

            colors = (random.random(), random.random(), random.random())
            mlab.pipeline.iso_surface(grid, color=colors, contours=[0.5], opacity=0.1)

            if show_seeds:
                seed_grid = mlab.pipeline.scalar_field(seed_mask)
                seed_grid.spacing = CONFIG.volume.resolution

                mlab.pipeline.iso_surface(
                    seed_grid, color=colors, contours=[0.5], opacity=1
                )

        mlab.orientation_axes(figure=fig, xlabel="Z", zlabel="X")
        mlab.view(azimuth=45, elevation=30, focalpoint="auto", roll=90, figure=fig)

        fig2 = mlab.figure(size=(1280, 720))

        for intersection in self.get_intersections():
            grid = mlab.pipeline.scalar_field(intersection)
            grid.spacing = CONFIG.volume.resolution

            mlab.pipeline.iso_surface(
                grid,
                color=(random.random(), random.random(), random.random()),
                contours=[0.5],
                opacity=0.1,
            )

        mlab.orientation_axes(figure=fig2, xlabel="Z", zlabel="X")
        mlab.view(azimuth=45, elevation=30, focalpoint="auto", roll=90, figure=fig2)
        mlab.show()

    class SkeletonTree:
        def __init__(self):
            self.root = None

        def add_region(self, node):
            if self.root is None:
                self.root = node
                return True
            elif self.root.is_child(node):
                node.append_child(self.root)
                self.root = node
                return True
            else:
                return self.root.append_child(node)

        def fill(self, orig_bounds, body):
            for node in self.traverse():
                if node.id == orig_bounds.node_id[0]:
                    if node.body is not None:
                        logging.debug("past body: {0},  new body: {1}".format(node.body, body))
                        node.body = body
                        node.bounds = orig_bounds
                        return True
                    else:
                        logging.debug("new body: {0}".format(body))
                        node.body = body
                        node.bounds = orig_bounds
                        return True
                        raise Exception("resetting a region is not supported")
            raise Exception("node {0} not found".format(nid))

        def dump_tree(self):
            return str(self.root)

        def traverse(self):
            if self.root is None:
                print("NO ROOTS")
            else:
                return self.root.traverse()

    class RegionNode:
        def __init__(self, region=None, children=None, node=None):
            self.body = None
            self.bounds = None
            if node is not None:
                self.id = node[0]
                self.pid = node[1]
                self.center = node[2:]
            elif region is not None:
                self.body = region.to_body()
                self.bounds = region.orig_bounds
                self.id = region.orig_bounds.node_id[0]
                self.pid = region.orig_bounds.node_id[1]
            else:
                raise Exception("node or region must be provided")
            if children:
                self.children = children
            else:
                self.children = []

        def set_bounds(self, bounds):
            self.bounds = bounds


        def append_child(self, regionNode):
            if self.is_parent(regionNode):
                self.children.append(regionNode)
                return True
            else:
                for child in self.children:
                    if child.append_child(regionNode):
                        return True
                return False

        def is_equal(self, node):
            return node.id == self.id

        def is_parent(self, node):
            return self.id == node.pid

        def is_child(self, node):
            return node.id == self.pid

        def has_id(self, id):
            return self.id == id

        def has_pid(self, pid):
            return self.pid == pid

        def has_parent(self):
            return self.has_pid(None)

        def search_id(self, id):
            if self.has_id(id):
                return self
            else:
                for child in self.children:
                    found = child.search_id(id)
                    if found is not None:
                        return found
                return None

        def traverse(self):
            yield (self)
            for child in self.children:
                for x in child.traverse():
                    yield (x)

        def __str__(self):
            if len(self.children) > 0:
                return (
                    str(self.id) + "[" + ",".join([str(x) for x in self.children]) + "]"
                )
            else:
                return str(self.id)

