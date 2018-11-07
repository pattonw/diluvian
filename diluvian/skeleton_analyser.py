import numpy as np
from scipy.ndimage import gaussian_filter1d
from .octrees import OctreeVolume


class Analyser:
    def __init__(self, input_output):
        self.io = input_output
        self.io.analyser = self
        self.combined = None

    def _get_center_of_mass(self, data):
        x = np.linspace(-1, 1, data.shape[0])
        y = np.linspace(-1, 1, data.shape[1])
        z = np.linspace(-1, 1, data.shape[2])
        z_change = np.sum(np.dot(data, z))
        y_change = np.sum(np.dot(data.transpose(2, 0, 1), y))
        x_change = np.sum(np.dot(data.transpose(1, 2, 0), x))
        change_direction = np.array([x_change, y_change, z_change])
        change_mag = np.dot(change_direction, change_direction) ** 0.5
        change_direction = change_direction / change_mag
        return (change_direction, change_mag)

    def _create_sphere(self, shape):
        """
        This method assumes the shape is isotropic.
        TODO: Consider adding a resolution term to account
        for input shapes that are not isotropic
        NOTE: Isotropy is handled in part by the shape term,
        however this assumes that the input shape accounts
        perfectly for the resolution
        """

        def dist_to_center(i, j, k, shape):
            i = (i - shape[0] // 2) / shape[0]
            j = (j - shape[1] // 2) / shape[1]
            k = (k - shape[2] // 2) / shape[2]
            return (i ** 2 + j ** 2 + k ** 2) ** (0.5)

        sphere = np.ones(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if dist_to_center(i, j, k, shape) > 1:
                        sphere[i, j, k] = 0
        return sphere

    def get_connection(self, index_a, index_b):
        if self.combined is None:
            self.combine_segmentations_and_counts()
        a2b_matrix = self.combined[list(map(slice, index_a, index_b))]
        costs = np.zeros(a2b_matrix.size)
        for i in range(a2b_matrix.shape[0]):
            for j in range(a2b_matrix.shape[1]):
                for k in range(a2b_matrix.shape[2]):
                    current_score = min(a2b_matrix[i, j, k])
                    region = list(
                        map(
                            slice, [x - 1 if x > 0 else x for x in [i, j, k]], [i, j, k]
                        )
                    )
                    costs[i, j, k] = min(current_score, np.min(costs[region]))

    def combine_segmentations_and_counts(self, min_overlap_count=3):
        if self.combined is None:
            segmentation = self.io.segmentation
            counts = self.io.segmentation_counts
            combined = OctreeVolume(
                segmentation.leaf_shape,
                segmentation.bounds,
                segmentation.dtype,
                segmentation.populator,
            )
            for seg_leaf, count_leaf in zip(
                segmentation.iter_leaves(), counts.iter_leaves()
            ):
                seg_data = seg_leaf.data
                count_data = count_leaf.data
                if seg_data is not None and count_data is not None:
                    mask = np.copy(seg_data)
                    mask[seg_data < min_overlap_count] = 0
                    mask[seg_data >= min_overlap_count] = (
                        mask[seg_data >= min_overlap_count]
                        / count_data[seg_data >= min_overlap_count]
                    )
                    combined[
                        list(
                            map(
                                slice,
                                seg_leaf.bounds[0],
                                np.minimum(seg_leaf.bounds[1], combined.bounds[1]),
                            )
                        )
                    ] = mask[
                        list(
                            map(
                                slice,
                                [0, 0, 0],
                                np.minimum(seg_leaf.bounds[1], combined.bounds[1])
                                - seg_leaf.bounds[0],
                            )
                        )
                    ]
            self.combined = combined
        return self.combined

    def center_of_mass_calculator(self, use_combined=False):
        sphere = None
        if use_combined and self.combined is None:
            self.combine_segmentations_and_counts()
        while True:
            if use_combined:
                node = yield
                mask = self.combined[
                    list(
                        map(
                            slice,
                            node.value.center - np.array([48, 48, 12]),
                            node.value.center + np.array([49, 49, 13]),
                        )
                    )
                ]
            else:
                node, mask = yield
            if sphere is None:
                sphere = self._create_sphere(mask.shape)
            node.value.center_of_mass = self._get_center_of_mass(mask)

    def calculate_center_of_mass_vects(self):
        sphere = None
        for node in self.io.get_nodes():
            mask = node.value.mask
            if mask is None:
                continue
            data = np.copy(mask)
            if sphere is None:
                sphere = self._create_sphere(data.shape)
            data[sphere == 0] = 0
            direction, mag = self._get_center_of_mass(data)
            self.io.input_center_of_mass(node, (direction, mag))

    def resample_segments(self, delta, steps, sigma_fraction):
        """
        resample tree to have evenly spaced nodes. Gaussian smooth the curve
        and then sample at regular intervals.

        inputs: tree, nodes, regions or coords (nodes makes most sense I think)
        outputs: tree, nodes, regions or coords (coords makes most sense I think)
        io can then generate a new skeleton from coords
        """

        def handle_root(node, seen_roots, new_id, seen_tails):
            if node.key in seen_roots:
                return [], seen_roots, new_id
            else:
                if node.key in seen_tails:
                    new_node = []
                    seen_roots[node.key] = seen_tails[node.key]
                    del seen_tails[node.key]
                    return new_node, seen_roots, new_id
                else:
                    new_node = (
                        new_id,
                        None,
                        node.value.center[0],
                        node.value.center[1],
                        node.value.center[2],
                    )
                    seen_roots[node.key] = new_node
                    new_id += 1
                    return [new_node], seen_roots, new_id

        # store root and branch points to avoid moving or duplicating them
        seen_roots = {}
        # store tail points so that we can use them as roots
        seen_tails = {}
        # create a list of new node points of form (nid, pid, x, y, z)
        new_tree_nodes = []
        # new nodes will need new nids/pids thus we will reassign all nids starting at 0
        new_node_id = 0
        # get each straight segment
        for segment in self.io.get_segments():
            # make sure each point in the segment has coordinates
            assert all(
                [
                    node.value is not None and node.value.center is not None
                    for node in segment
                ]
            ), "segment contains some nodes with no center coordinates"

            # handle the root node (new_root will be empty list if not needed)
            new_root, seen_roots, new_node_id = handle_root(
                segment[0], seen_roots, new_node_id, seen_tails
            )
            # get interpolated nodes TAIL included but not Root
            new_interpolated_nodes, new_node_id = self.resample_segment(
                segment,
                delta,
                steps,
                sigma_fraction,
                new_node_id,
                seen_roots[segment[0].key],
            )
            # handle the tail node (new_tail will be empty list if not needed)
            seen_tails[segment[-1].key] = new_interpolated_nodes[-1]

            new_tree_nodes = new_tree_nodes + new_root + new_interpolated_nodes
        new_skeleton = self.io.new_skeleton()
        new_skeleton.io.input_nid_pid_x_y_z(new_tree_nodes)
        return new_skeleton

    def resample_segment(self, nodes, delta, steps, sigma_fraction, new_node_id, root):
        def get_smoothed(coords, steps=100, sigma_fraction=.001):
            x_y_z = list(zip(*coords))
            t = np.linspace(0, 1, len(coords))
            t2 = np.linspace(0, 1, steps)

            x_y_z_2 = list(map(lambda x: np.interp(t2, t, x), x_y_z))
            x_y_z_3 = list(
                map(
                    lambda x: gaussian_filter1d(
                        x, steps * sigma_fraction, mode="nearest"
                    ),
                    x_y_z_2,
                )
            )
            return zip(*x_y_z_3)

        def downsample(coords, delta, origin, end):
            previous = origin
            for coord in coords + [end]:
                sqdist = sum((np.array(coord) - np.array(previous)) ** 2)
                if sqdist < delta ** 2:
                    continue
                elif sqdist > (2 * delta) ** 2:
                    k = (sqdist ** 0.5) // delta
                    for i in range(0, int(k)):
                        new_coords = (
                            ((k - 1) - i) * np.array(previous)
                            + (i + 1) * np.array(coord)
                        ) / k

                        if any(np.isnan(new_coords)):
                            raise Exception("NAN FOUND")
                        yield list(new_coords)
                    previous = coord
                else:
                    yield coord
                    previous = coord

        coords = [node.value.center for node in nodes]
        smoothed_coords = list(get_smoothed(coords, steps, sigma_fraction))
        downsampled_coords = downsample(
            smoothed_coords, delta, root[2:], nodes[-1].value.center
        )
        previous_id = root[0]
        current_id = new_node_id
        downsampled_nodes = []
        for coord in downsampled_coords:
            node = (current_id, previous_id, coord[0], coord[1], coord[2])
            previous_id = current_id
            current_id += 1
            downsampled_nodes.append(node)
        downsampled_nodes.append(
            (current_id, previous_id, coords[-1][0], coords[-1][1], coords[-1][2])
        )

        return downsampled_nodes, new_node_id + len(downsampled_nodes)

    def get_regularness(self):
        """
        Get mean, std and outliers for distance between nodes. 
        Determine whether skeleton needs to be resampled or not
        """
        raise NotImplementedError("not done yet")

