# -*- coding: utf-8 -*-


from __future__ import division

import itertools
import logging
from collections import deque

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

    Note, the skeleton object only deals with coordinates in the
    output space. So if there was any downsampling, make sure all
    coordinates are properly scaled before being passed to the 
    skeleton.
    """

    def __init__(self, id_pairs):
        """
        initialize a new Skeleton with a list of id_pairs.
        each id_pair is a tuple (id, parent_id)
        if id == parent_id, that point is the root.
        """
        self.tree = self.SkeletonTree()
        self.start = [float("inf"), float("inf"), float("inf")]
        self.stop = [0, 0, 0]
        self.outline_from_pairs(id_pairs)

    def get_bounds(self):
        """
        get the absolute min and max coordinates of the
        volume covered by the skeleton.
        """
        if all([self.start[i] < self.stop[i] for i in range(len(self.start))]):
            return np.array(self.start), np.array(self.stop)
        else:
            raise Exception("Skeleton does not have a positive volume")

    def update_bounds(self, bounds):
        self.start = np.array([min(self.start[i], bounds.start[i]) for i in range(3)])
        self.stop = np.array([max(self.stop[i], bounds.stop[i]) for i in range(3)])

    def fill(self, nid, bounds, body):
        if self.tree.fill(nid, bounds, body):
            logging.info("region {0} successfully filled".format(nid))
            self.update_bounds(bounds)

    def is_filled(self, nid):
        """
        helper method for filling in a tree
        """
        node = self.tree.search(nid)
        if node:
            return node.is_filled()
        else:
            raise Exception("node {0} not found".format(nid))

    def outline_from_pairs(self, pairs):
        """
        This method takes a list of nodes with their coordinates and a shape vector.
        Each nodes coordinate is assumed to be the desired seed point of a volume of the 
        desired shape centered on the given node. This is used to build the tree structure
        before starting flood filling.
        """
        self.tree.outline(pairs)

    def get_masks(self):
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
            if node.has_volume():
                mask, _ = node.body.get_seeded_component(
                    CONFIG.postprocessing.closing_shape
                )
                bounds = node.get_bounds()
                id = node.key
            else:
                continue
            yield (mask, bounds)

    def get_skeleton_mask(self):
        """
        get one big mask for the entire skeleton, ignoring individual sections
        and seeds. Much more efficient and saves on a lot of memory and computation
        time.
        """
        start, stop = self.get_bounds()
        self.skeleton_mask = np.zeros(stop - start)
        for node in self.tree.traverse():
            try:
                node_mask, _ = node.get_body().get_seeded_component(
                    CONFIG.postprocessing.closing_shape
                )
                node_start, node_stop = node.get_bounds()
            except Exception as e:
                logging.debug(e)
                continue

            self.skeleton_mask[
                list(
                    map(
                        slice,
                        np.array(node_start) - np.array(start),
                        np.array(node_stop) - np.array(start),
                    )
                )
            ] = np.maximum(
                self.skeleton_mask[
                    list(
                        map(
                            slice,
                            np.array(node_start) - np.array(start),
                            np.array(node_stop) - np.array(start),
                        )
                    )
                ],
                node_mask,
            )
        return self.skeleton_mask

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

                int_mask += (
                    parent_mask[
                        list(map(slice, np.array(int_start), np.array(int_stop)))
                    ]
                    + child_mask[
                        list(map(slice, np.array(int_start), np.array(int_stop)))
                    ]
                )
                int_mask = int_mask // 2

                yield int_mask, int_start

    def save_skeleton_mask(self, output_file):
        """
        save skeleton masks to a file for rendering elsewhere
        """
        np.save(output_file, self.get_skeleton_mask())

    def save_skeleton_masks(self, output_file):
        output = []
        for x in self.get_masks():
            output.append(x)
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

        def outline(self, pairs):
            nodes = {}
            for id, pid in pairs:
                nodes[id] = self.RegionNode(id)
            for id, pid in pairs:
                if (pid is None or id == pid) and self.root is None:
                    self.root = nodes[id]
                elif pid is None or id == pid:
                    raise Exception("Multiple root nodes not supported")
                else:
                    child = nodes[id]
                    parent = nodes[pid]
                    parent.append_child(child)
            if self.root is None:
                raise Exception("Root node was not in the list")
            if len([x for x in self.traverse()]) != len(pairs):
                raise Exception(
                    "number of nodes in tree ({0}) does not match number of nodes given ({1})".format(
                        len([x for x in self.traverse()]), len(pairs)
                    )
                )

        def fill(self, key, bounds, body):
            """
            Fill in an existing node with data from flood filling.
            """
            x = self.search(key)
            if x:
                return x.set_data(bounds, body)
            else:
                raise Exception("node {0} not found".format(nid))
            

        def dump_tree(self):
            """
            Get a string representation of the tree
            """
            return str(self.root)

        def search(self, key):
            for node in self.traverse():
                if node.has_key(key):
                    return node
            raise Exception("node {0} does not exist".format(key))

        def traverse(self):
            """
            Iterate over the elements of the tree
            """
            if self.root is None:
                logging.debug("NO ROOTS")
            else:
                return self.breadth_first_traversal()

            
        def breadth_first_traversal(self):
            queue = deque([self.root])

            while len(queue) > 0:
                current = queue.popleft()
                yield current
                for child in current.get_children():
                    queue.append(child)

        class RegionNode:
            """
            The RegionNode goes along with the SkeletonTree

            Each node contains information about the region it represents such as bounds
            and the flood filled mask.
            """

            def __init__(self, id):
                """
                initialize a node with either a region or a node
                """
                self.value = {
                    "body": None,
                    "start": None,
                    "stop": None,
                    "children": [],
                    "is_filled": False,
                    "has_volume": False,
                }
                self.key = id

            def set_bounds(self, bounds):
                self.value["start"] = bounds.start
                self.value["stop"] = bounds.stop

            def get_bounds(self):
                return self.value["start"], self.value["stop"]

            def set_body(self, body):
                self.value["body"] = body

            def get_body(self):
                return self.value["body"]

            def set_data(self, bounds, body):
                self.set_bounds(bounds)
                self.set_body(body)
                self.value['is_filled'] = True
                try:
                    node_mask, _ = body.get_seeded_component(
                        CONFIG.postprocessing.closing_shape
                    )
                    self.value['has_volume'] = True
                    return True
                except Exception as e:
                    logging.debug(e)
                    self.value['has_volume'] = False
                    return False

            def append_child(self, child):
                self.value["children"].append(child)

            def get_children(self):
                return self.value["children"]

            def get_child(self, index):
                if len(self.get_children()) > index:
                    return self.get_children[index]
                else:
                    return None

            def is_filled(self):
                return self.value['is_filled']

            def has_volume(self):
                return self.value['has_volume']

            def is_equal(self, node):
                return node.key == self.key

            def has_key(self, key):
                return self.key == key

            def __str__(self):
                if len(self.children) > 0:
                    return (
                        str(self.key)
                        + "["
                        + ",".join([str(x) for x in self.get_children()])
                        + "]"
                    )
                else:
                    return str(self.key)

