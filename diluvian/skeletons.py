# -*- coding: utf-8 -*-


from __future__ import division

import logging
from collections import deque
from skimage import measure

import numpy as np
import random

from .config import CONFIG


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

    def __init__(self, id_pairs=None):
        """
        initialize a new Skeleton with a list of id_pairs.
        each id_pair is a tuple (id, parent_id)
        if id == parent_id, that point is the root.
        """
        self.tree = self.SkeletonTree()
        self.start = [float("inf"), float("inf"), float("inf")]
        self.stop = [0, 0, 0]
        self.scale = {"shift": np.array([0, 0, 6050]), "scale": np.array([1, 4, 4])}
        if id_pairs is not None:
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
        returns masks and bounds for each node
        """
        for node in self.tree.traverse():
            if node.has_volume():
                mask, _ = node.get_body().get_seeded_component(
                    CONFIG.postprocessing.closing_shape
                )
                bounds = node.get_bounds()
                # id = node.key
            else:
                continue
            yield (mask, bounds, node.key, node.pid)

    def get_disjoint_masks(self):
        """
        First combines intersecting masks before returning them.
        This is done by combining the binary arrays. Note that
        this approach will become intractable for large connected
        components. Thus it may be better to work with combining
        the meshes themselves.
        """

        def intersects(bound_a, bound_b):
            def f(a, A, b, B, i):
                k = 6 if i == 0 else 32
                a = a + k
                b = b + k
                A = A - k
                B = B - k
                return a <= b <= A or a <= B <= A or b <= a <= B or b <= A <= B

            intersections = [
                f(bound_a[0][i], bound_a[1][i], bound_b[0][i], bound_b[1][i], i)
                for i in range(3)
            ]
            if all(intersections):
                return True
            else:
                return False

        def combine(mask_A, bound_A, mask_B, bound_B):
            new_start = np.array([min(bound_A[0][i], bound_B[0][i]) for i in range(3)])
            new_stop = np.array([max(bound_A[1][i], bound_B[1][i]) for i in range(3)])
            combined = np.zeros(np.array(new_stop) - np.array(new_start))
            combined[
                list(map(slice, bound_B[0] - new_start, bound_B[1] - new_start))
            ] = mask_B
            combined[
                list(map(slice, bound_A[0] - new_start, bound_A[1] - new_start))
            ] = np.maximum(
                combined[
                    list(map(slice, bound_A[0] - new_start, bound_A[1] - new_start))
                ],
                mask_A,
            )
            return (combined, (new_start, new_stop))

        done = False
        masks, bounds = [], []
        for mask in self.get_masks():
            masks.append(mask[0])
            bounds.append(mask[1])

        while not done:
            temp_masks = []
            temp_bounds = []
            done = True
            for i in range(len(masks)):
                mask = masks[i]
                bound = bounds[i]
                combined = False
                for j in range(len(temp_masks)):
                    if not combined and intersects(bound, temp_bounds[j]):
                        done = False
                        mask, bound = combine(
                            mask, bound, temp_masks[j], temp_bounds[j]
                        )
                        temp_masks[j] = mask
                        temp_bounds[j] = bound
                        combined = True
                if not combined:
                    temp_masks.append(mask)
                    temp_bounds.append(bound)
            masks = temp_masks
            bounds = temp_bounds
        return zip(masks, bounds)

    def get_skeleton_mask(self):
        """
        get one big mask for the entire skeleton, very inefficient for large
        volumes
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
        Useful for determining location of false merges.
        """
        for parent in self.tree.traverse():
            for child in parent.children:
                parent_mask, parent_bound = parent.get_mask()
                child_mask, child_bound = child.get_mask()

                if parent_mask is None or child_mask is None:
                    continue

                int_start = [
                    max(parent_bound.start[i], child_bound.start[i]) for i in range(3)
                ]
                int_stop = [
                    min(parent_bound.stop[i], child_bound.stop[i]) for i in range(3)
                ]
                int_mask = np.zeros(np.array(int_stop) - np.array(int_start))

                int_mask += (
                    parent_mask[
                        list(map(slice, np.array(int_start), np.array(int_stop)))
                    ]
                    + child_mask[
                        list(map(slice, np.array(int_start), np.array(int_stop)))
                    ]
                ) // 2

                if np.sum(int_mask) == 0:
                    continue

                yield int_mask, int_start, (parent.key, child.key)

    def save_skeleton_mask(self, output_file):
        """
        save skeleton mask to a file for rendering elsewhere
        """
        np.save(output_file, self.get_skeleton_mask())

    def save_skeleton_masks(self, output_file):
        output = []
        for x in self.get_masks():
            output.append(x)
        np.save(output_file, output)

    def save_skeleton_mask_mesh(self, output_file):
        """
        Combine masks into one large mesh
        """
        all_verts = []
        all_faces = []
        for x in self.get_disjoint_masks():
            mask = np.pad(x[0], ((1,), (1,), (1,)), "constant", constant_values=(0,))
            verts, faces, normals, values = measure.marching_cubes_lewiner(mask, 0.5)
            verts = [[v[i] + x[1][0][i] - 1 for i in range(3)] for v in verts]
            faces = [[f[i] + len(all_verts) for i in range(3)] for f in faces]
            for v in verts:
                all_verts.append(v)
            for f in faces:
                all_faces.append(f)
            logging.info("Number of vertices: ({})".format(len(verts)))
            logging.info("Number of faces: ({})".format(len(faces)))
        logging.info("Total number of vertices: ({})".format(len(all_verts)))
        logging.info("Total number of faces: ({})".format(len(all_faces)))
        np.save(output_file, [all_verts, all_faces])

    def save_skeleton_mask_meshes(self, output_file):
        """
        create many small meshes around each sample point of the skeleton.
        """
        ids = []
        mesh_verts = []
        mesh_faces = []
        for x in self.get_masks():
            mask = np.pad(x[0], ((1,), (1,), (1,)), "constant", constant_values=(0,))
            verts, faces, normals, values = measure.marching_cubes_lewiner(mask, 0.5)
            verts = [[v[i] + x[1][0][i] - 1 for i in range(3)] for v in verts]
            mesh_verts.append(verts)
            mesh_faces.append(faces)
            ids.append(x[2])
        np.save(output_file, [mesh_verts, mesh_faces, ids])

    def save_stats(self, output_file):
        """
        Creates a csv of the form (id, pid, x, y, z, intersect) where:
        -id is the nodes id,
        -pid is the id of the nodes parent (own id if root),
        -x is the x coordinate of a node
        -y is the y coordinate of a node
        -z is the z coordinate of a node
        -intersect is whether a node's mask intersects with the mask of its parent.
        """
        with open(output_file + ".csv", "w") as f:
            parent = self.root
            node_queue = [self.root]
            while len(node_queue) > 0:
                current = node_queue.pop(0)
                if parent is None:
                    parent = current
                f.write(
                    "{},{},{},{},{},{}".format(
                        current,
                        parent,
                        current.x,
                        current.y,
                        current.z,
                        int(current.intersects(parent)),
                    )
                )
                for child in current.get_children():
                    node_queue.append(child)
                parent = current
                current = node_queue.pop(0)

    def render_skeleton(self, show_seeds=True, with_intersections=False):
        """
        TODO.
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
            for nid, pid in pairs:
                nodes[nid] = self.RegionNode(nid, pid)
            for nid, pid in pairs:
                if (pid is None or nid == pid) and self.root is None:
                    self.root = nodes[nid]
                elif pid is None or nid == pid:
                    raise Exception("Multiple root nodes not supported")
                else:
                    child = nodes[nid]
                    if pid in nodes:
                        parent = nodes[pid]
                        parent.append_child(child)
            if self.root is None:
                kid_counts = {}
                for nid in nodes:
                    current = nodes[nid]
                    if nid in kid_counts:
                        current.pid = None
                        kid_counts[nid] += 1
                    else:
                        while current.pid in nodes:
                            current = nodes[current.pid]
                        if current.key in kid_counts:
                            kid_counts[current.key] += 1
                        else:
                            kid_counts[current.key] = 1
                rootid = max(
                    [[nid, kid_counts[nid]] for nid in kid_counts], key=lambda x: x[1]
                )[0]
                self.root = nodes[rootid]

                # raise Exception("Root node was not in the list")

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
                raise Exception("node {0} not found".format(key))

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

            def __init__(self, nid, pid=None):
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
                self.pid = pid
                self.key = nid

            def set_bounds(self, bounds):
                self.value["start"] = bounds.start
                self.value["stop"] = bounds.stop

            def get_bounds(self):
                return self.value["start"], self.value["stop"]

            def set_body(self, body):
                self.value["body"] = body

            def get_body(self):
                return self.value["body"]

            def get_mask(self):
                return self.value["body"].body.get_seeded_component(
                    CONFIG.postprocessing.closing_shape
                )

            def set_data(self, bounds, body):
                self.set_bounds(bounds)
                self.set_body(body)
                self.value["is_filled"] = True
                try:
                    node_mask, _ = body.get_seeded_component(
                        CONFIG.postprocessing.closing_shape
                    )
                    self.value["has_volume"] = True
                    return True
                except Exception as e:
                    logging.debug(e)
                    self.value["has_volume"] = False
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
                return self.value["is_filled"]

            def has_volume(self):
                return self.value["has_volume"]

            def is_equal(self, node):
                return node.key == self.key

            def has_key(self, key):
                return self.key == key

            def intersects(self, other):
                """
                Check if the overlapping sections of two nodes share any points.
                They cannot overlap if either one failed to generate a mask.
                """
                if not self.is_filled() or not other.is_filled():
                    return False
                mask_A = self.get_mask()
                bound_A = self.get_bounds()
                mask_B = other.get_mask()
                bound_B = other.get_bounds()
                new_start = np.array(
                    [max(bound_A[0][i], bound_B[0][i]) for i in range(3)]
                )
                new_stop = np.array(
                    [min(bound_A[1][i], bound_B[1][i]) for i in range(3)]
                )
                small_mask_a = mask_A[
                    list(map(slice, new_start - bound_A[0], new_stop - bound_A[0]))
                ]
                small_mask_b = mask_B[
                    list(map(slice, new_start - bound_B[0], new_stop - bound_B[0]))
                ]
                combined = (small_mask_a + small_mask_b) // 2
                return np.sum(combined) > 0

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
