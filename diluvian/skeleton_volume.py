# -*- coding: utf-8 -*-

from .skeleton_analyser import Analyser
from .skeleton_visualizer import Visualizer
from .arbors import Arbor
from .octrees import OctreeVolume

import numpy as np


class Skeleton:
    def __init__(self):
        self.io = self.InputOutput()
        self.analyser = Analyser(self.io)
        self.visualizer = Visualizer(self.io, self.analyser)

    class InputOutput:
        """
        Class for interfacing with a skeleton volume. This class includes the
        data type for nodes to store data such as coordinates and bounds
        """

        def __init__(self):
            self.arbor = Arbor()
            self.segmentation = None
            self.segmentation_counts = None
            self.min_coord = None
            self.max_coord = None
            self.min_bound = None
            self.max_bound = None
            self.max_mass_v_mag = -1
            self.fov_shape = None
            self.filled = {}
            self.node_map = None

        def is_filled(self, nid):
            return self.filled.get(nid, False)

        def fill(self, nid, nid_bounds, body):
            if self.node_map is None:
                self.node_map = {}
                for node in self.get_nodes():
                    self.node_map[node.key] = node
            node = self.node_map[nid]

            node.value.set_bounds((nid_bounds.start, nid_bounds.stop))
            try:
                mask, _ = body.get_seeded_component()
                node.value.set_mask(mask)
            except ValueError:
                print("No seeded component this time mate!")
            self.filled[nid] = True

        def save_masks(self, filename=""):
            masks = []
            nids = []
            for node in self.get_nodes():
                try:
                    nids.append(node.key)
                    masks.append(node.value.mask)
                except AttributeError:
                    print("Node {} has no mask".format(node.key))
            np.save(filename + "_masks", masks)
            np.save(filename + "_nids", nids)

        def new_skeleton(self):
            return Skeleton()

        def scale(self, scaling_function):
            """
            scale the coordinates of all nodes 
            """
            assert (
                self.segmentation is None and self.segmentation_counts is None
            ), "scaling segmentation data is not yet supported"
            self.min_coord = (
                np.array(list(zip(*scaling_function(*zip(self.min_coord))))[0])
                if self.min_coord is not None
                else None
            )
            self.max_coord = (
                np.array(list(zip(*scaling_function(*zip(self.max_coord))))[0])
                if self.max_coord is not None
                else None
            )
            self.min_bound = (
                np.array(list(zip(*scaling_function(*zip(self.min_bound))))[0])
                if self.min_bound is not None
                else None
            )
            self.max_bound = (
                np.array(list(zip(*scaling_function(*zip(self.max_bound))))[0])
                if self.max_bound is not None
                else None
            )
            for node in self.get_nodes():
                node.value.center = np.array(
                    list(zip(*scaling_function(*zip(node.value.center))))[0]
                )
                if any(np.isnan(node.value.center)):
                    raise Exception("NaN found!")

        def get_nodes(self):
            return self.arbor.traverse()

        def get_regions(self):
            for node in self.get_nodes():
                yield node.value

        def recalculate_center_of_mass(self):
            center_of_mass_calculator = self.analyser.center_of_mass_calculator(True)
            center_of_mass_calculator.send(None)
            for node in self.get_nodes():
                center_of_mass_calculator.send(node)

        def get_center_of_mass(self, node):
            center_of_mass = node.value.center_of_mass
            if center_of_mass is not None:
                return center_of_mass[0], center_of_mass[1] / self.max_mass_v_mag
            else:
                return None

        def build_tree(self, id_to_data):
            """
            build the tree from an map of the form:
            id: node, pid, data
            Thus we can add edges to nodes to build the tree
            TODO: move node creation into this funciton so we just need id: pid, data maps
            """
            roots = []
            for nid, data in id_to_data.items():
                node, pid, value = data
                if node.value is None:
                    node.value = self.Region()
                if value is not None and value.get("center", None) is not None:
                    self.update_coords(value["center"])
                if value is not None and value.get("bounds", None) is not None:
                    self.update_bounds(value["bounds"])
                node.value.insert_values(value)
                parent = id_to_data.get(pid, None)
                if parent is None:
                    roots.append(node)
                else:
                    parent[0].add_child(node)
            if len(roots) == 1:
                self.arbor.root = roots[0]
            else:
                sizes = [len(list(node.traverse())) for node in roots]
                self.arbor.root = roots[sizes.index(max(sizes))]
                self.recalculate_bounds()

        def recalculate_bounds(self):
            """
            recalculate bounds of this skeleton volume
            """
            self.min_bound = None
            self.max_bound = None
            self.min_coord = None
            self.max_coord = None
            for region in self.get_regions():
                self.update_bounds(region.bounds)
                self.update_coords(region.center)

        def get_tree_bounds(self):
            """
            get bounds of this skeleton volume
            TODO: view_radius should be set on a skeleton volume and assumed
            to remain constant
            """
            self.min_coord = np.array([float("inf"), float("inf"), float("inf")])
            self.max_coord = np.array([0, 0, 0])
            for node in self.get_nodes():
                self.min_coord = np.min(
                    np.array([self.min_coord, node.value.center]), axis=0
                )
                self.max_coord = np.max(
                    np.array([self.max_coord, node.value.center]), axis=0
                )
            return (
                self.min_coord - self.fov_shape // 2,
                self.max_coord + self.fov_shape // 2 + 1,
            )

        def input_nodes(self, nodes):
            """
            create the tree by copying the nodes provided
            TODO: change this to just take the data from another
            trees nodes and build a new tree
            """
            node_map = {}
            for node in nodes:
                new_node, ids = node.clone()
                self.update_bounds(new_node.value.bounds)
                self.update_coords(new_node.value.center)
                if any(np.isnan(self.min_coord)):
                    raise Exception("NaN found!")
                node_map[ids[0]] = (new_node, ids)

            for nid in node_map:
                nid, pid, child_ids = node_map[nid][1]
                node = node_map[nid][0]
                if pid in node_map:
                    node.set_parent(node_map[pid][0])
                else:
                    self.arbor.root = node
                for child_id in child_ids:
                    if child_id in node_map:
                        node.add_child(node_map[child_id][0])

        def create_octrees(self, tree_bounds=None, block_shape=None):
            def _data_populator(bounds):
                return np.zeros(np.array(bounds[1]) - np.array(bounds[0]))

            if tree_bounds is None:
                tree_bounds = self.get_tree_bounds()
            if block_shape is None:
                block_shape = self.fov_shape
            self.segmentation = OctreeVolume(
                block_shape, tree_bounds, float, _data_populator
            )
            self.segmentation_counts = OctreeVolume(
                block_shape, tree_bounds, float, _data_populator
            )
            for node in self.get_nodes():
                if not self.is_filled(node.key):
                    continue
                node_bounds = node.value.bounds
                node_mask = node.value.mask
                print(
                    "Node covers {} - {}, with {} nonzero values".format(
                        node_bounds[0], node_bounds[1], np.sum(node_mask)
                    )
                )
                node_bounds = list(map(slice, node_bounds[0], node_bounds[1]))
                self.segmentation_counts[node_bounds] = (
                    self.segmentation_counts[node_bounds] + 1
                )
                if node_mask is not None:
                    self.segmentation[node_bounds] = (
                        self.segmentation[node_bounds] + node_mask
                    )
                    print(
                        "Seg Octree now has {} values in range {}".format(
                            np.sum(self.segmentation[node_bounds]), node_bounds
                        )
                    )

        def input_masks(
            self, data, axes=[2, 1, 0], c_o_m=True, tree_bounds=None, block_shape=None
        ):
            """
            c_o_m: whether or not to calculate the center of mass vectors for individual nodes
            """
            if c_o_m:
                nid_node_map = self.arbor.get_key_map()
                center_of_mass_calculator = self.analyser.center_of_mass_calculator()
                center_of_mass_calculator.send(None)
            if self.segmentation is None or self.segmentation_counts is None:
                self.create_octrees(tree_bounds=tree_bounds, block_shape=block_shape)
            for mask, bounds, nid, pid in data:
                mask = mask.transpose(axes)
                bounds = [[bound[axes[i]] for i in range(3)] for bound in bounds]
                if c_o_m:
                    node = nid_node_map[nid]
                    center_of_mass_calculator.send((node, mask))
                    self.max_mass_v_mag = max(
                        self.max_mass_v_mag, node.value.center_of_mass[1]
                    )
                self.segmentation[list(map(slice, bounds[0], bounds[1]))] += mask
                self.segmentation_counts[list(map(slice, bounds[0], bounds[1]))] += 1
            self.segmentation.bounds = self.segmentation.get_leaf_bounds()
            self.segmentation_counts.bounds = self.segmentation_counts.get_leaf_bounds()

        def input_masks_old(self, data):
            """
            input masks into the tree by providing a list of
            datapoints of the form (mask, bounds, nid, pid)
            TODO: This funciton should not need bounds or pids
            Should also insert mask data into the Octrees instead
            of nodes.
            NOTE: Instead of nids and masks, assuming constant shaped input,
            could aslo just take mask and bounds and insert into Octree.
            """
            id_to_data = {nid: [mask, bounds] for mask, bounds, nid, _ in data}
            for node in self.arbor.traverse():
                data = id_to_data.get(node.key, None)
                if data is not None:
                    self.insert_data(node, data[1], data[0], [2, 1, 0])

        def input_id_pairs(self, pairs):
            """
            build the tree by providing a list of (nid, pid) pairs. This is sufficient
            to build an arbor.
            TODO: rename function to reflect that this will build the arbor but nothing else.
            Also keep track of which parts have already been built to determine what functionality
            is or is not available at a give time
            """
            id_to_data = {nid: [self.arbor.Node(nid), pid, None] for nid, pid in pairs}
            self.build_tree(id_to_data)

        def input_nid_pid_x_y_z(self, nodes):
            """
            builds the arbor and includes node coordinates.
            TODO: rename to make it clearer what this functions provides.
            Also add a function to just insert node coordinates.
            """
            id_to_data = {
                nid: [
                    self.arbor.Node(nid),
                    pid,
                    {"center": np.array([x, y, z], dtype="float")},
                ]
                for nid, pid, x, y, z in nodes
            }
            self.build_tree(id_to_data)

        def input_center_of_mass(self, node, center_of_mass):
            """
            input a nodes center of mass, update the max center of mass magnitude for normalizing
            """
            node.value.center_of_mass = center_of_mass
            self.max_mass_v_mag = max(self.max_mass_v_mag, center_of_mass[1])

        def update_bounds(self, bounds):
            """
            update bounds of skeleton
            TODO: remove this or coords, since the field of view should be static and thus
            the two are trivially computable from the other
            """
            if bounds is None:
                return
            if self.min_bound is None:
                self.min_bound = bounds[0]
            else:
                self.min_bound = np.min(np.array([self.min_bound, bounds[0]]), axis=0)
            if self.max_bound is None:
                self.max_bound = bounds[1]
            else:
                self.max_bound = np.max(np.array([self.max_bound, bounds[1]]), axis=0)

        def update_coords(self, coord):
            """
            update coordinates of skeleton
            TODO: remove this or bounds, since the field of view should be static and thus
            the two are trivially computable from the other
            """
            if self.min_coord is None:
                self.min_coord = coord
            else:
                self.min_coord = np.min(np.array([self.min_coord, coord]), axis=0)
            if self.max_coord is None:
                self.max_coord = coord
            else:
                self.max_coord = np.max(np.array([self.max_coord, coord]), axis=0)

        def insert_data(self, node, bounds, mask, axes):
            """
            Inserts some data into a node. This is mostly here to make sure axes are kept
            reliable
            axes: [x,y,z]
                - if your data is in [z,y,x] order, axes should be [2,1,0] to show axes positions
            """
            mask = mask.transpose(axes)
            bounds = [bound[axes] for bound in bounds]
            self.update_bounds(bounds)
            if node.value is None:
                value = self.Region(bounds, mask)
                node.set_value(value)
            else:
                node.value.insert_values({"bounds": bounds, "mask": mask})

        def insert_id_data_triplets(self, triplets, axes):
            """
            insert masks and bounds into the tree via node ids.
            axes: [x,y,z]
                - if your data is in [z,y,x] order, axes should be [2,1,0] to show axes positions
            """
            id_to_data = {nid: (bounds, mask) for nid, bounds, mask in triplets}
            for node in self.arbor.traverse():
                data = id_to_data.get(node.key, None)
                if data is not None:
                    self.insert_data(node, data[0], data[1], axes)

        def get_interesting_nodes(self, root=True, leaves=True, branches=True):
            """
            This function extracts interesting nodes (root, leaves branches).
            TODO: move this funciton into the Arbor class since physical
            coordinates are irrelevant
            """

            if not (root or leaves or branches):
                return []
            for node in self.arbor.traverse():
                if root:
                    root = False
                    yield node
                elif branches and len(node.get_neighbors()) > 2:
                    yield node
                elif leaves and len(node.get_neighbors()) == 1:
                    yield node

        def get_segments(self):
            """
            gets all the sections of the tree that are single chains of nodes with no branches
            """
            for segment in self.arbor.traverse_segments():
                yield segment

        def get_minimal_subtree(self, ids):
            """
            get the smallest possible subtree containing all given ids
            """
            all_nodes = self.arbor.get_minimal_subtree(ids)
            new_skeleton = Skeleton()
            new_skeleton.io.input_nodes(all_nodes)
            return new_skeleton

        def get_c_o_m_series(self):
            segments = self.arbor.get_root_leaf_paths()
            for segment in segments:
                series = []
                for node in segment:
                    series.append(
                        list(node.value.center)
                        + list(node.value.center_of_mass)
                        + [0 if len(node.children) == 1 else 1]
                    )
                yield np.array(series)

        def get_topological_copy(self, keep_root=True):
            """
            removes all simple nodes with 2 adjacent edges. if keep_root is False
            the root can potentially be removed as well and will be replaced with
            the closest branch point
            """
            raise NotImplementedError

        def get_branches(self):
            """
            gets branch nodes
            TODO: remove duplicity with get interesting nodes
            """
            for node in self.get_nodes():
                if len(node.get_neighbors()) > 2:
                    yield node

        def get_leaves(self):
            """
            gets leaf nodes
            TODO: remove duplicity with get interesting nodes
            """
            for node in self.get_nodes():
                if len(node.get_neighbors()) == 1:
                    yield node

        def get_radius(self, node, radius):
            """
            get all nodes within a specific radius (physical distance) of a given node
            """
            origin = node.value.center
            all_nodes = [node]
            previous = [node]
            layer = node.get_neighbors()
            while len(layer) > 0:
                all_nodes += layer
                next_layer = []
                for node in layer:
                    neighbors = node.get_neighbors()
                    for neighbor in neighbors:
                        if (
                            neighbor not in previous
                            and sum((neighbor.value.center - origin) ** 2) < radius ** 2
                        ):
                            next_layer.append(neighbor)
                previous = layer[:]
                layer = next_layer[:]
            return all_nodes

        def save_data_n5(self, folder_path, dataset_path):
            """
            Save the data gathered in the n5 format.

            dependent on pyn5 library which is expected to change a lot.
            """
            datasets = {
                "segmentation": self.segmentation,
                "counts": self.segmentation_counts,
            }
            for name, data in datasets.items():
                print("Saving {} to n5!".format(name))
                print("Num leaves = {}".format(len(list(data.iter_leaves()))))
                data.write_to_n5(folder_path, dataset_path + "/" + name)

        class Region:
            """
            The region class contains information and operations specific to
            the neuron skeleton nodes.
            TODO: Think of different names to make this clearer
            """

            def __init__(self, bounds=None, mask=None):
                """
                bounds: [np.array, np.array]
                mask: np.array
                """
                self.bounds = bounds
                if bounds is not None:
                    self.center = sum(bounds) // 2
                else:
                    self.center = None
                self.mask = mask
                self.center_of_mass = None
                self.strahler = None

            def insert_values(self, values):
                if values is not None:
                    bounds = values.get("bounds", None)
                    if bounds is not None:
                        self.set_bounds(bounds)
                    center = values.get("center", None)
                    if center is not None:
                        self.set_center(center)
                    mask = values.get("mask", None)
                    if mask is not None:
                        self.set_mask(mask)

            def set_mask(self, mask):
                if self.bounds is not None:
                    assert all(
                        self.bounds[1] - self.bounds[0] == mask.shape
                    ), "incomming mask does not match this nodes shape"
                self.mask = mask

            def set_center(self, center):
                if self.bounds is not None:
                    assert all(
                        sum(self.bounds) // 2 == center
                    ), "incoming center does not match center of this nodes bounds"
                self.center = center

            def set_bounds(self, bounds):
                if self.center is not None:
                    assert all(
                        self.center == sum(bounds) // 2
                    ), "incomming bounds: {} do not have same center as this node: {}".format(
                        bounds, self.center
                    )
                else:
                    self.center = sum(bounds) // 2
                self.bounds = bounds

            def clone(self):
                new_region = type(self)(
                    [x.clone() for x in self.bounds], self.mask.clone()
                )
                new_region.center_of_mass = self.center_of_mass.clone()
                return self

