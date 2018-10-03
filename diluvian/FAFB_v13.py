# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

from collections import deque
import logging
from multiprocessing import Manager, Process
import os
from PIL import Image
from pathlib import Path
import random
import tensorflow as tf
import math

import numpy as np
import six
from six.moves import input as raw_input
from tqdm import tqdm

from .config import CONFIG
from .volumes import SubvolumeBounds, ImageStackVolume, DownsampledVolume
from .regions import Region
from .skeletons import Skeleton


BOUNDS = [
    np.array([0, 0, 0], dtype="int"),
    np.array([7063, 15850, 10600], dtype="int") - 1,
]
RES = np.array([35, 4, 4], dtype="int")
TRANS = np.array([0, 50900, 120200], dtype="int")
IFOV = np.array([25, 97, 97], dtype="int")


class FAFBStackVolume(ImageStackVolume):
    def image_populator(self, bounds):
        FAFB_imgs = (
            "/groups"
            + "/flyTEM"
            + "/flyTEM"
            + "/from_tier2"
            + "/eric"
            + "/working_sets"
            + "/150625_segmentation_samples"
            + "/sample_D_plus"
            + "/cutout_10600x15850"
        )
        image_subvol = np.zeros(tuple(bounds[1] - bounds[0]), dtype=np.float32)
        col_range = list(
            map(
                int,
                (
                    math.floor(bounds[0][self.DIM.X] / self.tile_width),
                    math.ceil(bounds[1][self.DIM.X] / self.tile_width),
                ),
            )
        )
        row_range = list(
            map(
                int,
                (
                    math.floor(bounds[0][self.DIM.Y] / self.tile_height),
                    math.ceil(bounds[1][self.DIM.Y] / self.tile_height),
                ),
            )
        )
        tile_size = np.array([1, self.tile_height, self.tile_width]).astype(np.int64)

        for z in range(bounds[0][self.DIM.Z], bounds[1][self.DIM.Z]):
            if z in self.missing_z:
                image_subvol[int(z - bounds[0][self.DIM.Z]), :, :] = 0
                continue
            img = Path(FAFB_imgs) / "crop.{:>08d}.png".format(int(z))
            try:
                im = np.array(Image.open(img))
                # If the image is multichannel, throw our hands up and
                # just use the first channel.
                if im.ndim > 2:
                    im = im[:, :, 0].squeeze()
                im = im / 256.0
            except IOError:
                logging.debug("Failed to load tile: %d", z)
                im = np.full(
                    (self.orig_bounds[self.DIM.Y], self.orig_bounds[self.DIM.X]),
                    0,
                    dtype=np.float32,
                )
            for r in range(*row_range):
                for c in range(*col_range):
                    tile_coord = np.array([z, r, c]).astype(np.int64)
                    tile_loc = np.multiply(tile_coord, tile_size)

                    subvol = (
                        np.maximum(np.zeros(3), tile_loc - bounds[0]).astype(np.int64),
                        np.minimum(
                            np.array(image_subvol.shape),
                            tile_loc + tile_size - bounds[0],
                        ).astype(np.int64),
                    )
                    tile_sub = (
                        np.maximum(np.zeros(3), bounds[0] - tile_loc).astype(np.int64),
                        np.minimum(tile_size, bounds[1] - tile_loc).astype(np.int64),
                    )

                    image_subvol[
                        subvol[0][self.DIM.Z],
                        subvol[0][self.DIM.Y] : subvol[1][self.DIM.Y],
                        subvol[0][self.DIM.X] : subvol[1][self.DIM.X],
                    ] = im[
                        tile_sub[0][self.DIM.Y]
                        + tile_loc[self.DIM.Y] : tile_sub[1][self.DIM.Y]
                        + tile_loc[self.DIM.Y],
                        tile_sub[0][self.DIM.X]
                        + tile_loc[self.DIM.X] : tile_sub[1][self.DIM.X]
                        + tile_loc[self.DIM.X],
                    ]

        return image_subvol

    def real_coord_to_pixel(self, a):
        return np.floor_divide(a, self.orig_resolution) - self.translation

    def pixel_coord_to_real(self, a):
        return np.matmul(a + self.translation, self.orig_resolution)

    def downsample(self, resolution):
        downsample = self._get_downsample_from_resolution(resolution)
        if np.all(np.equal(downsample, 0)):
            return self
        else:
            return DownsampledVolume(self, downsample)


def seeds_from_skeleton(filename):
    if filename.name[-4:] == "json":
        import json

        json_file = Path(filename)
        with json_file.open("r") as f:
            skeleton = json.load(f)

        nodes = skeleton[0]
        ids, pids, _, xs, ys, zs, _, _ = zip(*nodes)
        pids = list(pids)
        for i in range(len(ids)):
            if pids[i] is None:
                pids[i] = ids[i]
        skeleton = np.array([ids, pids, zs, ys, xs], dtype="int").T

        def valid(x, y, z):
            return (
                BOUNDS[0][2] + IFOV[2]
                <= x // RES[2] - TRANS[2]
                < BOUNDS[1][2] - IFOV[2]
                and BOUNDS[0][1] + IFOV[1]
                <= y // RES[1] - TRANS[1]
                < BOUNDS[1][1] - IFOV[1]
                and BOUNDS[0][0] + IFOV[0]
                <= z // RES[0] - TRANS[0]
                < BOUNDS[1][0] - IFOV[0]
            )

        skeleton = skeleton[[valid(x[4], x[3], x[2]) for x in skeleton]]

        return skeleton[:, 2:], skeleton[:, :2]
    elif filename.name[-3:] == "csv":
        import csv

        coords = []
        ids = []
        with open(str(filename), newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for row in reader:
                coords.append([int(float(x)) for x in row[2:]])
                if row[1].strip() == "null" or row[1].strip() == "none":
                    ids.append([int(float(row[0])), None])
                else:
                    ids.append([int(float(x)) for x in row[:2]])
        return coords, ids


def fill_skeleton_with_model_threaded(
    model_file,
    skeleton_file_path,
    volumes=None,
    partition=False,
    augment=False,
    bounds_input_file=None,
    bias=True,
    move_batch_size=1,
    max_moves=None,
    remask_interval=None,
    sparse=False,
    moves=None,
    num_workers=1,
    worker_prequeue=1,
    reject_early_termination=False,
    reject_non_seed_components=True,
    save_output_file=None,
):
    def worker(
        worker_id,
        set_devices,
        model_file,
        volume,
        region_shape,
        nodes,
        results,
        lock,
        revoked,
    ):
        lock.acquire()
        import tensorflow as tf

        if set_devices:
            # Only make one GPU visible to Tensorflow so that it does not allocate
            # all available memory on all devices.
            # See: https://stackoverflow.com/questions/37893755
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)

        with tf.device("/gpu:0"):
            # Late import to avoid Keras import until TF bindings are set.
            from .network import load_model

            logging.debug("Worker %s: loading model", worker_id)
            model = load_model(model_file, CONFIG.network)
        lock.release()

        def is_revoked(test_node):
            ret = False
            lock.acquire()
            if tuple(test_node) in revoked:
                ret = True
                revoked.remove(tuple(test_node))
            lock.release()
            return ret

        while True:
            node = nodes.get(True)

            if not isinstance(node, np.ndarray):
                logging.debug("Worker %s: got DONE", worker_id)
                break

            if is_revoked(node):
                results.put((node, None))
                continue

            def stopping_callback(region):
                stop = is_revoked(node)
                if (
                    reject_non_seed_components
                    and region.bias_against_merge
                    and region.mask[tuple(region.seed_vox)] < 0.5
                ):
                    stop = True
                return stop

            logging.debug("Worker %s: got seed %s", worker_id, str(node))

            logging.debug(
                "start: {0}".format(node[2:] - np.floor_divide(region_shape, 2))
            )
            logging.debug(
                "stop: {0}".format(node[2:] + np.floor_divide(region_shape, 2) + 1)
            )

            print(node)
            image = volume.get_subvolume(
                SubvolumeBounds(
                    start=(node[2:]) - np.floor_divide(region_shape, 2),
                    stop=(node[2:]) + np.floor_divide(region_shape, 2) + 1,
                    node_id=node[0:2],
                )
            ).image

            # Flood-fill and get resulting mask.
            # Allow reading outside the image volume bounds to allow segmentation
            # to fill all the way to the boundary.
            region = Region(
                image,
                seed_vox=np.floor_divide(np.array(image.shape), 2) + 1,
                sparse_mask=False,
                block_padding="reflect",
            )
            region.bias_against_merge = bias
            try:
                six.next(
                    region.fill(
                        model,
                        move_batch_size=move_batch_size,
                        max_moves=max_moves,
                        stopping_callback=stopping_callback,
                        remask_interval=remask_interval,
                    )
                )
            except Region.EarlyFillTermination:
                logging.debug("Worker %s: node %s failed to fill", worker_id, str(node))
            except StopIteration:
                pass
            logging.debug("Worker %s: node %s filled", worker_id, str(node))

            results.put((node, region.to_body()))

    """
    Generate regions to fill:

    Each region should be a small area around a point on the skeleton.
    After flood filling all regions, I will look for intersections or the
    lack thereof that will indicate false mergers, allong with heavy
    segmentation perpendicular to skeleton that may indicate missing
    branches.

    """

    volume_a = FAFBStackVolume(
        np.array([7063, 15850, 10600], dtype="int") - 1,
        np.array([35, 4, 4], dtype="int"),
        np.array([0, 50900, 120200], dtype="int"),
        530,
        634,
        None,
        zoom_level=0,
        missing_z=None,
        image_leaf_shape=None,
    )
    volume = volume_a.downsample([35, 16, 16])
    seeds, ids = seeds_from_skeleton(skeleton_file)
    seeds = [
        list(volume.world_coord_to_local(volume_a.real_coord_to_pixel(seed)))
        for seed in seeds
    ]
    nodes = [np.array(list(ids[i]) + seeds[i]) for i in range(len(seeds))]
    skel = Skeleton(ids)
    region_shape = (
        CONFIG.model.input_fov_shape
        + 4
        * CONFIG.model.output_fov_shape
        // CONFIG.model.output_fov_move_fraction
    )

    pbar = tqdm(desc="Node queue", total=len(nodes), miniters=1, smoothing=0.0)
    num_nodes = len(nodes)
    nodes = iter(nodes)

    manager = Manager()
    # Queue of seeds to be picked up by workers.
    node_queue = manager.Queue()
    # Queue of results from workers.
    results_queue = manager.Queue()
    # Dequeue of seeds that were put in seed_queue but have not yet been
    # combined by the main process.
    dispatched_nodes = deque()
    # Seeds that were placed in seed_queue but subsequently covered by other
    # results before their results have been processed. This allows workers to
    # abort working on these seeds by checking this list.
    revoked_nodes = manager.list()
    # Results that have been received by the main process but have not yet
    # been combined because they were not received in the dispatch order.
    unordered_results = {}

    def queue_next_node():
        total = 0
        for node in nodes:
            if skel.is_filled(node[0]):
                # This seed has already been filled.
                total += 1
                continue
            dispatched_nodes.append(node)
            node_queue.put(node)

            break

        return total

    for _ in range(min(num_nodes, num_workers * worker_prequeue)):
        processed_nodes = queue_next_node()
        pbar.update(processed_nodes)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        set_devices = False
        num_workers = 1
        logging.warn(
            "Environment variable CUDA_VISIBLE_DEVICES is set, so only one worker can be used.\n"
            "See https://github.com/aschampion/diluvian/issues/11"
        )
    else:
        set_devices = True

    workers = []
    loading_lock = manager.Lock()
    for worker_id in range(num_workers):
        w = Process(
            target=worker,
            args=(
                worker_id,
                set_devices,
                model_file,
                volume,
                region_shape,
                node_queue,
                results_queue,
                loading_lock,
                revoked_nodes,
            ),
        )
        w.start()
        workers.append(w)

    # For each seed, create region, fill, threshold, and merge to output volume.
    while dispatched_nodes:
        processed_nodes = 1
        expected_node = dispatched_nodes.popleft()
        logging.debug("Expecting node %s", np.array_str(expected_node))

        if tuple(expected_node) in unordered_results:
            logging.debug(
                "Expected node %s is in old results",
                np.array_str(expected_node),
            )
            node = expected_node
            body = unordered_results[tuple(node)]
            del unordered_results[tuple(node)]

        else:
            node, body = results_queue.get(True)
            processed_nodes += queue_next_node()

            while not np.array_equal(node, expected_node):
                logging.debug("Node %s is early, stashing", np.array_str(node))
                unordered_results[tuple(node)] = body
                node, body = results_queue.get(True)
                processed_nodes += queue_next_node()

        logging.debug("Processing node at %s", np.array_str(node))
        pbar.update(processed_nodes)

        if skel.is_filled(node[0]):
            # This seed has already been filled.
            logging.debug(
                "Node (%s) was filled but has been covered in the meantime.",
                np.array_str(node),
            )
            loading_lock.acquire()
            if tuple(node) in revoked_nodes:
                revoked_nodes.remove(tuple(node))
            loading_lock.release()
            continue

        if body is None:
            raise Exception("Body is None.")

        if not body.is_seed_in_mask():
            logging.debug(
                "Seed (%s) is not in its body.", np.array_str(node[2:])
            )

        mask, bounds = body._get_bounded_mask(
            CONFIG.postprocessing.closing_shape
        )

        body_size = np.count_nonzero(mask)

        if body_size == 0:
            logging.debug("Body is empty.")

        logging.debug("Adding body to prediction label volume.")

        orig_bounds = SubvolumeBounds(
            start=node[2:] - np.floor_divide(region_shape, 2),
            stop=node[2:] + np.floor_divide(region_shape, 2) + 1,
        )
        skel.fill(node[0], orig_bounds, body)

        logging.debug("Filled node (%s)", np.array_str(node))

    for _ in range(num_workers):
        node_queue.put("DONE")
    for wid, worker in enumerate(workers):
        worker.join()
    manager.shutdown()

    pbar.close()
    skel.save_skeleton_mask_meshes(skeleton_file.name.split(".")[0])


def run():
    CONFIG.from_toml("trained_models/pattonw-v0/pattonw-v0.toml")

    random.seed(CONFIG.random_seed)
    np.random.seed(CONFIG.random_seed)
    tf.set_random_seed(CONFIG.random_seed)

    model_file = "trained_models/pattonw-v0/pattonw-v0.hdf5"
    skeleton_file_path = Path("../tests/")
    for skeleton_file in skeleton_file_path.iterdir():
        if skeleton_file.name[:18] == "27884_downsampled_":
            fill_skeleton_with_model_threaded(
                model_file,
                skeleton_file,
                volumes=None,
                partition=False,
                augment=False,
                bounds_input_file=None,
                bias=True,
                move_batch_size=1,
                max_moves=None,
                remask_interval=None,
                sparse=False,
                moves=None,
                num_workers=8,
                worker_prequeue=1,
                reject_early_termination=False,
                reject_non_seed_components=True,
                save_output_file=None,
            )
