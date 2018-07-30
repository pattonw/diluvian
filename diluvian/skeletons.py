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
    A Skeleton is an object that handles storing information about
    the skeleton you wish to flood fill and contains methods for
    its analysis.
    """

    def __init__(self):
        """
        initialize a new Skeleton with no data.
        """
        self.tree = self.SkeletonTree()
        self.start = [float("inf"), float("inf"), float("inf")]
        self.stop = [0, 0, 0]

    def get_bounds(self):
        """
        get the absolute min and max coordinates of the
        volume covered by the skeleton.
        """
        return self.start, self.stop

    def is_filled(self, nid):
        """
        helper method for filling in a tree
        """
        for node in self.tree.traverse():
            if node.id == nid:
                return node.body != None
        raise Exception("node {0} not found".format(nid))

    def outline(self, nodes, shape):
        """
        This method takes a list of nodes with their coordinates and a shape vector.
        Each nodes coordinate is assumed to be the desired seed point of a volume of the 
        desired shape centered on the given node. This is used to build the tree structure
        before starting flood filling.

        This is to avoid spending time filling a skeleton with unhandled abnormalities.
        Currently unhandled abnormalities to be added:
        - disconnected segments in tree
        """
        min_seed = [float("inf")] * 3
        max_seed = [0] * 3
        for node in nodes:
            seed = [node[2 + i] // [1, 4, 4][i] for i in range(3)]
            min_seed = [min(seed[i], min_seed[i]) for i in range(3)]
            max_seed = [max(seed[i], max_seed[i]) for i in range(3)]
            region_node = self.RegionNode(node=node)
            if not self.tree.add_region(region_node):
                raise Exception("region not parent or child of previous regions")
        self.start = [int(min_seed[i] - shape[i] // 2) for i in range(3)]
        self.stop = [int(max_seed[i] + shape[i] // 2 + 1) for i in range(3)]

    def get_masks(self, show_seeds=True):
        """
        returns masks and seed masks for each individual section that has been filled.
        This is useful for visualizations but very slow on large skeletons.

        WIP: currently creates a numpy array of the same size as the whole skeleton
        for each mask so that masks can be rendered in the same image with correct
        relative positioning.
        Instead mask array should only be as large as necessary and save its starting
        coordinates to be hugely more efficient.
        """
        for node in self.tree.traverse():
            print("{0}, {1}".format(np.array(self.start), np.array(self.stop)))
            mask = np.zeros(np.array(self.stop) - np.array(self.start))
            try:
                node_mask, _ = node.body.get_seeded_component(
                    CONFIG.postprocessing.closing_shape
                )
                bounds = node.bounds
            except Exception as e:
                logging.debug(e)
                node_mask = np.zeros(node.body.mask.shape)
                bounds = node.bounds

            print("Bounds: start: {0}, stop: {1}".format(bounds.start, bounds.stop))
            logging.debug("node_mask shape: {0}".format(node_mask.shape))
            logging.debug(
                "Bounds: start: {0}, stop: {1}".format(bounds.start, bounds.stop)
            )

            print(
                "slice: {0}".format(
                    list(
                        map(
                            slice,
                            np.array(bounds.start) - np.array(self.start),
                            np.array(bounds.stop) - np.array(self.start),
                        )
                    )
                )
            )

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

    def get_skeleton_mask(self):
        """
        get one big mask for the entire skeleton, ignoring individual sections
        and seeds. Much more efficient and saves on a lot of memory and computation
        time.
        """
        self.skeleton_mask = np.zeros(self.stop - self.start)
        for node in self.tree.traverse():
            try:
                node_mask, _ = node.body.get_seeded_component(
                    CONFIG.postprocessing.closing_shape
                )
                bounds = node.bounds
            except Exception as e:
                logging.debug(e)
                continue

            self.skeleton_mask[
                list(
                    map(
                        slice,
                        np.array(bounds.start) - np.array(self.start),
                        np.array(bounds.stop) - np.array(self.start),
                    )
                )
            ] = np.maximum(
                self.skeleton_mask[
                    list(
                        map(
                            slice,
                            np.array(bounds.start) - np.array(self.start),
                            np.array(bounds.stop) - np.array(self.start),
                        )
                    )
                ],
                node_mask,
            )
        return self.skeleton_mask

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
        """
        get the intersections of neighboring reagions in the tree.
        Very useful for determining location of false merges.

        WIP:
        currently also returns masks of same size as skeleton for positioning.
        Should return masks only of necessary size along with starting coords.
        """
        for parent in self.tree.traverse():
            for child in parent.children:
                parent_mask, _ = parent.body.get_seeded_component(
                    CONFIG.postprocessing.closing_shape
                )
                child_mask, _ = child.body.get_seeded_component(
                    CONFIG.postprocessing.closing_shape
                )

                int_start = [
                    max(parent.bounds.start[i], child.bounds.start[i]) for i in range(3)
                ]
                int_stop = [
                    min(parent.bounds.stop[i], child.bounds.stop[i]) for i in range(3)
                ]
                int_mask = np.zeros(np.array(int_stop) - np.array(int_start))

                int_mask += parent_mask[list(
                        map(
                            slice,
                            np.array(int_start),
                            np.array(int_stop),
                        )
                    )] += child_mask[list(
                        map(
                            slice,
                            np.array(int_start),
                            np.array(int_stop),
                        )
                    )]
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
        """
        save skeleton masks to a file for rendering elsewhere
        """
        output = []
        for mask, seed_mask in self.get_masks(show_seeds):
            output.append((mask, seed_mask))
        np.save(output_file, output)

    def render_skeleton(self, show_seeds=True, with_intersections=False):
        """
        render each region of a skeleton individually with their seed points and color them
        accordingly. Optionally create a second image with the intersections.
        """
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

        if with_intersections:
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
    
    def render_large_skeleton(self):
        """
        Render the entire skeleton, ignoring individual sections and seed points.
        """
        from mayavi import mlab

        fig = mlab.figure(size=(1280, 720))

        mask = self.get_skeleton_mask()
        grid = mlab.pipeline.scalar_field(mask)
        grid.spacing = CONFIG.volume.resolution

        colors = (random.random(), random.random(), random.random())
        mlab.pipeline.iso_surface(grid, color=colors, contours=[0.5], opacity=0.1)

        mlab.orientation_axes(figure=fig, xlabel="Z", zlabel="X")
        mlab.view(azimuth=45, elevation=30, focalpoint="auto", roll=90, figure=fig)

        mlab.show()

    class SkeletonTree:
        """
        The skeleton tree class is a pretty standard tree data structure.
        The tree has a root node from which it can iterate of all of its nodes.
        """
        def __init__(self):
            """
            Initialize an empty tree
            """
            self.root = None

        def add_region(self, node):
            """
            add a region to the tree.

            WIP:
            Currently any node to be added must either be a parent or a child of
            nodes already in the tree. Does not yet support building the tree 
            with arbitrary node input order.
            """
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
            """
            Fill in an existing node with data from flood filling.
            """
            for node in self.traverse():
                if node.id == orig_bounds.node_id[0]:
                    if node.body is not None:
                        logging.debug(
                            "past body: {0},  new body: {1}".format(node.body, body)
                        )
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
            """
            Get a string representation of the tree
            """
            return str(self.root)

        def traverse(self):
            """
            Iterate over the elements of the tree
            """
            if self.root is None:
                print("NO ROOTS")
            else:
                return self.root.traverse()

    class RegionNode:
        """
        The RegionNode goes along with the SkeletonTree

        Each node contains information about the region it represents such as bounds
        and the flood filled mask.
        """
        def __init__(self, region=None, children=None, node=None):
            """
            initialize a node with either a region or a node
            """
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
            """
            set the bounds of a regionNode
            """
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

