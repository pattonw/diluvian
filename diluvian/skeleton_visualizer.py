import numpy as np
import matplotlib.pyplot as plt
import math


class Visualizer:
    def __init__(self, input_output, analyser):
        self.io = input_output
        self.analyser = analyser
        self.plot_shape = None
        self.ax = None
        self.fig = None

    def view_wire_frame(
        self,
        view_radius=np.array([48, 48, 12]),
        emph_nodes=None,
        c_o_m=False,
        num_graphs=1,
    ):
        self.build_graph(view_radius, num_graphs)
        self.add_skeleton_wire_frame(emph_nodes)
        if c_o_m:
            if emph_nodes is None:
                nodes = self.io.get_nodes()
            else:
                nodes = emph_nodes
            self.add_center_of_mass_vectors(nodes)
        if num_graphs > 1 and emph_nodes is not None:
            self.add_c_o_m_series(nodes)
        self.show()

    def add_c_o_m_series(self, nodes):
        directions = []
        magnitudes = []
        branch_nodes = []
        for node in nodes:
            if node.value.center_of_mass is not None:
                direction, magnitude = self.io.get_center_of_mass(node)
                directions.append(direction)
                magnitudes.append(magnitude)
                branch_nodes.append(1 if len(node.children) > 1 else 0)
        # directions = zip(
        #    [gaussian_filter1d(x, 0.01, mode="nearest") for x in zip(directions)]
        # )
        # directions = [direction[0].reshape([3]) for direction in directions]
        # magnitudes = gaussian_filter1d(magnitudes, 0.01, mode="nearest")
        angle_changes = []
        magnitude_changes = []
        for i in range(1, len(magnitudes)):
            angle_changes.append(
                math.acos(
                    min(
                        np.dot(directions[i - 1], directions[i])
                        / (sum(directions[i - 1] ** 2) * sum(directions[i] ** 2)),
                        1,
                    )
                )
            )
            magnitude_changes.append(magnitudes[i] - magnitudes[i - 1])

        def moving_average(a, n=3):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1 :] / n

        self.ax[1].plot(moving_average(angle_changes))
        self.ax[1].plot(moving_average(magnitude_changes))
        self.ax[1].plot(moving_average(branch_nodes))
        self.ax[1].legend(["Angle changes", "Magnitude changes", "Branch locations"])
        self.ax[1].set_title(["Segmentation center of mass offset from sample point"])

    def add_skeleton_wire_frame(self, emphasize_nodes=None, emphasize_branches=True):
        coords = []
        if emphasize_branches:
            branch_coords = []
        if emphasize_nodes is not None:
            emph_coords = []
        for node in self.io.get_nodes():
            if emphasize_branches and len(node.children) > 1:
                branch_coords.append(node.value.center)
            elif emphasize_nodes is not None and node in emphasize_nodes:
                emph_coords.append(node.value.center)
            else:
                coords.append(node.value.center)
            for child in node.children:
                self.ax[0].plot(
                    [node.value.center[0], child.value.center[0]],
                    [node.value.center[1], child.value.center[1]],
                    [node.value.center[2], child.value.center[2]],
                    color="black",
                )

        xs, ys, zs = zip(*coords)
        if emphasize_branches and len(branch_coords) > 0:
            bxs, bys, bzs = zip(*branch_coords)
            self.ax[0].scatter(bxs, bys, bzs, s=40, color="red")
        if emphasize_nodes is not None and len(emph_coords) > 0:
            exs, eys, ezs = zip(*emph_coords)
            self.ax[0].scatter(exs, eys, ezs, s=30, color="green")
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        self.ax[0].scatter(xs, ys, zs, s=10)

    def add_center_of_mass_vectors(self, nodes=None):
        def add_com_vector(ax, direction, mag, coords, shape):
            xs = [coords[0], coords[0] + direction[0] * shape[0] / 10 * mag]
            ys = [coords[1], coords[1] + direction[1] * shape[1] / 10 * mag]
            zs = [coords[2], coords[2] + direction[2] * shape[2] / 10 * mag]
            ax.plot(xs, ys, zs)

        if nodes is None:
            nodes = self.io.get_nodes()

        for node in nodes:
            if node.value.center_of_mass is None:
                continue
            direction, mag = self.io.get_center_of_mass(node)
            add_com_vector(
                self.ax[0], direction, mag, node.value.center, self.plot_shape
            )

    def build_graph(self, view_radius=[48, 48, 12], num_graphs=1, ion=False):
        self.fig = plt.figure(figsize=(10 * num_graphs, 10))
        if ion:
            plt.ion()
        self.ax = []
        for i in range(1, num_graphs + 1):
            if i == 1:
                ax = self.fig.add_subplot(1, num_graphs, i, projection="3d")

                view_dims = self.io.get_tree_bounds()
                self.plot_shape = view_dims[1] - view_dims[0]
                ax.set_xlim(view_dims[0][0], view_dims[1][0])
                ax.set_ylim(view_dims[0][1], view_dims[1][1])
                ax.set_zlim(view_dims[0][2], view_dims[1][2])
                self.ax.append(ax)
            else:
                ax = self.fig.add_subplot(1, num_graphs, i)
                self.ax.append(ax)

    def show(self):
        plt.show()

    def view_meshes_octree(self, min_overlap_count=3, percent_overlap_agreement=0.5):
        from mayavi import mlab
        import random

        epsilon = 0.000001  # for float comparisons

        fig = mlab.figure(size=(1280, 720))
        spacing = np.array([16, 16, 35])

        # Bounds and point_centers are kept track of because grid
        # spacing of volumes messes up relative positioning
        leaves = list(
            zip(
                self.io.segmentation.iter_leaves(),
                self.io.segmentation_counts.iter_leaves(),
            )
        )
        coords = [(leaf.bounds, np.sum(leaf.bounds, axis=0) // 2) for leaf, _ in leaves]
        bounds, centers = zip(*coords)
        centers = np.array(centers)
        lower_bounds = [bound[0] for bound in bounds if bound is not None]
        min_bound = np.min(np.array(lower_bounds), axis=0)
        centers = centers - min_bound
        centers = centers * spacing

        i = 0
        for seg_leaf, count_leaf in leaves:
            i += 1
            seg_data = seg_leaf.data
            count_data = count_leaf.data

            if seg_data is not None and count_data is not None:
                seg_data = np.copy(seg_data)
                count_data = np.copy(count_data)
                seg_data[seg_data < 3] = 0
                seg_data[seg_data >= 3] = (
                    seg_data[seg_data >= 3] / count_data[seg_data >= 3]
                )
                seg_data[seg_data > percent_overlap_agreement - epsilon] = 1
                seg_data[seg_data != 1] = 0

                grid = mlab.pipeline.scalar_field(seg_data)
                grid.spacing = list(spacing)
                grid.origin = (seg_leaf.bounds[0] - min_bound) * spacing

                colors = (random.random(), random.random(), random.random())

                mlab.pipeline.iso_surface(
                    grid, color=colors, contours=[1], opacity=0.05
                )

                mlab.orientation_axes(figure=fig, xlabel="Z", zlabel="X")
                mlab.view(
                    azimuth=45, elevation=30, focalpoint="auto", roll=90, figure=fig
                )
        centers = (
            np.array([node.value.center for node in self.io.get_nodes()]) - min_bound
        ) * spacing
        xs, ys, zs = zip(*centers)
        mlab.points3d(xs, ys, zs, scale_factor=100)
        mlab.show_pipeline(rich_view=False)
        mlab.show()

    def view_meshes(self, k, coverage_cutoff=0.5):
        """
        TODO.
        """
        from mayavi import mlab
        import random

        fig = mlab.figure(size=(1280, 720))
        spacing = np.array([4, 4, 35])

        # Bounds and point_centers are kept track of because grid
        # spacing of volumes messes up relative positioning
        coords = [
            (node.value.bounds, node.value.center) for node in self.io.get_nodes()
        ]
        bounds, centers = zip(*coords)
        centers = np.array(centers)
        lower_bounds = [bound[0] for bound in bounds if bound is not None]
        min_bound = np.min(np.array(lower_bounds), axis=0)
        centers = centers - min_bound
        centers = centers * spacing

        i = 0
        for node in self.io.get_nodes():
            i += 1
            if i % k != 0:
                continue
            mask_data = node.value.mask
            if mask_data is not None and True:
                mask = np.copy(mask_data)
                mask[mask > coverage_cutoff] = 1
                mask[mask != 1] = 0

                grid = mlab.pipeline.scalar_field(mask)
                grid.spacing = list(spacing)
                grid.origin = (node.value.bounds[0] - min_bound) * spacing

                colors = (random.random(), random.random(), random.random())

                mlab.pipeline.iso_surface(
                    grid, color=colors, contours=[1], opacity=0.05
                )

                mlab.orientation_axes(figure=fig, xlabel="Z", zlabel="X")
                mlab.view(
                    azimuth=45, elevation=30, focalpoint="auto", roll=90, figure=fig
                )
        xs, ys, zs = zip(*centers)
        mlab.points3d(xs, ys, zs, scale_factor=20)
        mlab.show_pipeline(rich_view=False)
        mlab.show()
