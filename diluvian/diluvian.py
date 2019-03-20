# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

from collections import deque
import itertools
import logging
from multiprocessing import (
        Manager,
        Process,
        )
import os
import random

import numpy as np
import pytoml as toml
import six
from six.moves import input as raw_input
from tqdm import tqdm

from .config import CONFIG
from . import preprocessing
from .training import augment_subvolume_generator
from .util import (
        get_color_shader,
        Roundrobin,
        WrappedViewer,
        )
from .volumes import (
        HDF5Volume,
        partition_volumes,
        SubvolumeBounds,
        )
from .regions import Region
import sarbor


def generate_subvolume_bounds(filename, volumes, num_bounds, sparse=False, moves=None):
    if '{volume}' not in filename:
        raise ValueError('CSV filename must contain "{volume}" for volume name replacement.')

    if moves is None:
        moves = 5
    else:
        moves = np.asarray(moves)
    subv_shape = CONFIG.model.input_fov_shape + CONFIG.model.move_step * 2 * moves

    if sparse:
        gen_kwargs = {'sparse_margin': subv_shape}
    else:
        gen_kwargs = {'shape': subv_shape}
    for k, v in six.iteritems(volumes):
        bounds = v.downsample(CONFIG.volume.resolution)\
                  .subvolume_bounds_generator(**gen_kwargs)
        bounds = itertools.islice(bounds, num_bounds)
        SubvolumeBounds.iterable_to_csv(bounds, filename.format(volume=k))


def get_skeleton_bounds(filename, volumes, num_bounds, sparse=False, moves=None):
    seeds, ids = seeds_from_skeleton(filename)

    if len(volumes.keys()) > 1:
        # TODO: support multiple volumes
        raise ValueError("WIP. multiple volumes not yet supported")
    if moves is None:
        moves = 5
    else:
        moves = np.asarray(moves)
    subv_shape = CONFIG.model.input_fov_shape + CONFIG.model.move_step * moves

    if sparse:
        gen_kwargs = {"sparse_margin": subv_shape}
    else:
        gen_kwargs = {"shape": subv_shape, "seeds": seeds, "ids": ids}
    subvolume_bounds = {}
    for k, v in volumes.items():
        bounds = v.downsample(CONFIG.volume.resolution).subvolume_bounds_generator(
            **gen_kwargs
        )
        bounds = itertools.islice(bounds, num_bounds)
        subvolume_bounds[k] = bounds
    return subvolume_bounds


def seeds_from_skeleton(filename):
    import csv

    coords = []
    ids = []
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            coords.append([int(float(x)) for x in row[2:]])
            if row[1].strip() == "null" or row[1].strip() == "none":
                ids.append([int(float(row[0])), None])
            elif row[0] == row[1]:
                ids.append([int(float(row[0])), None])
            else:
                ids.append([int(float(x)) for x in row[:2]])
    return coords[::], ids


def fill_volume_with_model(
        model_file,
        volume,
        resume_prediction=None,
        checkpoint_filename=None,
        checkpoint_label_interval=20,
        seed_generator='sobel',
        background_label_id=0,
        bias=True,
        move_batch_size=1,
        max_moves=None,
        max_bodies=None,
        num_workers=CONFIG.training.num_gpus,
        worker_prequeue=1,
        filter_seeds_by_mask=True,
        reject_non_seed_components=True,
        reject_early_termination=False,
        remask_interval=None,
        shuffle_seeds=True):
    subvolume = volume.get_subvolume(SubvolumeBounds(start=np.zeros(3, dtype=np.int64), stop=volume.shape))
    # Create an output label volume.
    if resume_prediction is None:
        prediction = np.full_like(subvolume.image, background_label_id, dtype=np.uint64)
        label_id = 0
    else:
        if resume_prediction.shape != subvolume.image.shape:
            raise ValueError('Resume volume prediction is wrong shape.')
        prediction = resume_prediction
        prediction.flags.writeable = True
        label_id = prediction.max()
    # Create a conflict count volume that tracks locations where segmented
    # bodies overlap. For now the first body takes precedence in the
    # predicted labels.
    conflict_count = np.full_like(prediction, 0, dtype=np.uint32)

    def worker(worker_id, set_devices, model_file, image, seeds, results, lock, revoked):
        lock.acquire()
        import tensorflow as tf

        if set_devices:
            # Only make one GPU visible to Tensorflow so that it does not allocate
            # all available memory on all devices.
            # See: https://stackoverflow.com/questions/37893755
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_id)

        with tf.device('/gpu:0'):
            # Late import to avoid Keras import until TF bindings are set.
            from .network import load_model

            logging.debug('Worker %s: loading model', worker_id)
            model = load_model(model_file, CONFIG.network)
        lock.release()

        def is_revoked(test_seed):
            ret = False
            lock.acquire()
            if tuple(test_seed) in revoked:
                ret = True
                revoked.remove(tuple(test_seed))
            lock.release()
            return ret

        while True:
            seed = seeds.get(True)

            if not isinstance(seed, np.ndarray):
                logging.debug('Worker %s: got DONE', worker_id)
                break

            if is_revoked(seed):
                results.put((seed, None))
                continue

            def stopping_callback(region):
                stop = is_revoked(seed)
                if reject_non_seed_components and \
                   region.bias_against_merge and \
                   region.mask[tuple(region.seed_vox)] < 0.5:
                    stop = True
                return stop

            logging.debug('Worker %s: got seed %s', worker_id, np.array_str(seed))

            # Flood-fill and get resulting mask.
            # Allow reading outside the image volume bounds to allow segmentation
            # to fill all the way to the boundary.
            region = Region(image, seed_vox=seed, sparse_mask=True, block_padding='reflect')
            region.bias_against_merge = bias
            early_termination = False
            try:
                six.next(region.fill(
                    model,
                    move_batch_size=move_batch_size,
                    max_moves=max_moves,
                    progress=2 + worker_id,
                    stopping_callback=stopping_callback,
                    remask_interval=remask_interval))
            except Region.EarlyFillTermination:
                early_termination = True
            except StopIteration:
                pass
            if reject_early_termination and early_termination:
                body = None
            else:
                body = region.to_body()
            logging.debug('Worker %s: seed %s filled', worker_id, np.array_str(seed))

            results.put((seed, body))

    # Generate seeds from volume.
    generator = preprocessing.SEED_GENERATORS[seed_generator]
    seeds = generator(subvolume.image, CONFIG.volume.resolution)

    if filter_seeds_by_mask and volume.mask_data is not None:
        seeds = [s for s in seeds if volume.mask_data[tuple(volume.world_coord_to_local(s))]]

    pbar = tqdm(desc='Seed queue', total=len(seeds), miniters=1, smoothing=0.0)
    label_pbar = tqdm(desc='Labeled vox', total=prediction.size, miniters=1, smoothing=0.0, position=1)
    num_seeds = len(seeds)
    if shuffle_seeds:
        random.shuffle(seeds)
    seeds = iter(seeds)

    manager = Manager()
    # Queue of seeds to be picked up by workers.
    seed_queue = manager.Queue()
    # Queue of results from workers.
    results_queue = manager.Queue()
    # Dequeue of seeds that were put in seed_queue but have not yet been
    # combined by the main process.
    dispatched_seeds = deque()
    # Seeds that were placed in seed_queue but subsequently covered by other
    # results before their results have been processed. This allows workers to
    # abort working on these seeds by checking this list.
    revoked_seeds = manager.list()
    # Results that have been received by the main process but have not yet
    # been combined because they were not received in the dispatch order.
    unordered_results = {}

    def queue_next_seed():
        total = 0
        for seed in seeds:
            if prediction[seed[0], seed[1], seed[2]] != background_label_id:
                # This seed has already been filled.
                total += 1
                continue
            dispatched_seeds.append(seed)
            seed_queue.put(seed)

            break

        return total

    for _ in range(min(num_seeds, num_workers * worker_prequeue)):
        processed_seeds = queue_next_seed()
        pbar.update(processed_seeds)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        set_devices = False
        num_workers = 1
        logging.warn('Environment variable CUDA_VISIBLE_DEVICES is set, so only one worker can be used.\n'
                     'See https://github.com/aschampion/diluvian/issues/11')
    else:
        set_devices = True

    workers = []
    loading_lock = manager.Lock()
    for worker_id in range(num_workers):
        w = Process(target=worker, args=(worker_id, set_devices, model_file, subvolume.image,
                                         seed_queue, results_queue, loading_lock, revoked_seeds))
        w.start()
        workers.append(w)

    last_checkpoint_label = label_id

    # For each seed, create region, fill, threshold, and merge to output volume.
    while dispatched_seeds:
        processed_seeds = 1
        expected_seed = dispatched_seeds.popleft()
        logging.debug('Expecting seed %s', np.array_str(expected_seed))

        if tuple(expected_seed) in unordered_results:
            logging.debug('Expected seed %s is in old results', np.array_str(expected_seed))
            seed = expected_seed
            body = unordered_results[tuple(seed)]
            del unordered_results[tuple(seed)]

        else:
            seed, body = results_queue.get(True)
            processed_seeds += queue_next_seed()

            while not np.array_equal(seed, expected_seed):
                logging.debug('Seed %s is early, stashing', np.array_str(seed))
                unordered_results[tuple(seed)] = body
                seed, body = results_queue.get(True)
                processed_seeds += queue_next_seed()

        logging.debug('Processing seed at %s', np.array_str(seed))
        pbar.set_description('Seed ' + np.array_str(seed))
        pbar.update(processed_seeds)

        if prediction[seed[0], seed[1], seed[2]] != background_label_id:
            # This seed has already been filled.
            logging.debug('Seed (%s) was filled but has been covered in the meantime.',
                          np.array_str(seed))
            loading_lock.acquire()
            if tuple(seed) in revoked_seeds:
                revoked_seeds.remove(tuple(seed))
            loading_lock.release()
            continue

        if body is None:
            logging.debug('Body was None.')
            continue

        if reject_non_seed_components and not body.is_seed_in_mask():
            logging.debug('Seed (%s) is not in its body.', np.array_str(seed))
            continue

        if reject_non_seed_components:
            mask, bounds = body.get_seeded_component(CONFIG.postprocessing.closing_shape)
        else:
            mask, bounds = body._get_bounded_mask()

        body_size = np.count_nonzero(mask)

        if body_size == 0:
            logging.debug('Body was empty.')
            continue

        # Generate a label ID for this region.
        label_id += 1
        if label_id == background_label_id:
            label_id += 1

        logging.debug('Adding body to prediction label volume.')
        bounds_shape = list(map(slice, bounds[0], bounds[1]))
        prediction_mask = prediction[bounds_shape] == background_label_id
        for seed in dispatched_seeds:
            if np.all(bounds[0] <= seed) and np.all(bounds[1] > seed) and mask[tuple(seed - bounds[0])]:
                loading_lock.acquire()
                if tuple(seed) not in revoked_seeds:
                    revoked_seeds.append(tuple(seed))
                loading_lock.release()
        conflict_count[bounds_shape][np.logical_and(np.logical_not(prediction_mask), mask)] += 1
        label_shape = np.logical_and(prediction_mask, mask)
        prediction[bounds_shape][np.logical_and(prediction_mask, mask)] = label_id

        label_pbar.set_description('Label {}'.format(label_id))
        label_pbar.update(np.count_nonzero(label_shape))
        logging.info('Filled seed (%s) with %s voxels labeled %s.',
                     np.array_str(seed), body_size, label_id)

        if max_bodies and label_id >= max_bodies:
            # Drain the queues.
            while not seed_queue.empty():
                seed_queue.get_nowait()
            break

        if checkpoint_filename is not None and label_id - last_checkpoint_label > checkpoint_label_interval:
            config = HDF5Volume.write_file(
                    checkpoint_filename + '.hdf5',
                    CONFIG.volume.resolution,
                    label_data=prediction)
            config['name'] = 'segmentation checkpoint'
            with open(checkpoint_filename + '.toml', 'wb') as tomlfile:
                tomlfile.write('# Filling model: {}\n'.format(model_file))
                tomlfile.write(str(toml.dumps({'dataset': [config]})))

    for _ in range(num_workers):
        seed_queue.put('DONE')
    for wid, worker in enumerate(workers):
        worker.join()
    manager.shutdown()

    label_pbar.close()
    pbar.close()

    return prediction, conflict_count


def fill_volumes_with_model(
        model_file,
        volumes,
        filename,
        resume_filename=None,
        partition=False,
        viewer=False,
        **kwargs):
    if '{volume}' not in filename:
        raise ValueError('HDF5 filename must contain "{volume}" for volume name replacement.')
    if resume_filename is not None and '{volume}' not in resume_filename:
        raise ValueError('TOML resume filename must contain "{volume}" for volume name replacement.')

    if partition:
        _, volumes = partition_volumes(volumes)

    for volume_name, volume in six.iteritems(volumes):
        logging.info('Filling volume %s...', volume_name)
        volume = volume.downsample(CONFIG.volume.resolution)
        if resume_filename is not None:
            resume_volume_filename = resume_filename.format(volume=volume_name)
            resume_volume = six.next(six.itervalues(HDF5Volume.from_toml(resume_volume_filename)))
            resume_prediction = resume_volume.to_memory_volume().label_data
        else:
            resume_prediction = None

        volume_filename = filename.format(volume=volume_name)
        checkpoint_filename = volume_filename + '_checkpoint'
        prediction, conflict_count = fill_volume_with_model(
                model_file,
                volume,
                resume_prediction=resume_prediction,
                checkpoint_filename=checkpoint_filename,
                **kwargs)

        config = HDF5Volume.write_file(
                volume_filename + '.hdf5',
                CONFIG.volume.resolution,
                label_data=prediction)
        config['name'] = volume_name + ' segmentation'
        with open(volume_filename + '.toml', 'wb') as tomlfile:
            tomlfile.write('# Filling model: {}\n'.format(model_file))
            tomlfile.write('# Filling kwargs: {}\n'.format(str(kwargs)))
            tomlfile.write(str(toml.dumps({'dataset': [config]})))

        if viewer:
            viewer = WrappedViewer(voxel_size=list(np.flipud(CONFIG.volume.resolution)))
            subvolume = volume.get_subvolume(SubvolumeBounds(start=np.zeros(3, dtype=np.int64), stop=volume.shape))
            viewer.add(subvolume.image, name='Image')
            viewer.add(prediction, name='Labels')
            viewer.add(conflict_count, name='Conflicts')

            viewer.print_view_prompt()


def fill_region_with_model(
        model_file,
        volumes=None,
        partition=False,
        augment=False,
        bounds_input_file=None,
        bias=True,
        move_batch_size=1,
        max_moves=None,
        remask_interval=None,
        sparse=False,
        moves=None):
    # Late import to avoid Keras import until TF bindings are set.
    from .network import load_model

    if volumes is None:
        raise ValueError('Volumes must be provided.')

    if partition:
        _, volumes = partition_volumes(volumes)

    if bounds_input_file is not None:
        gen_kwargs = {
                k: {'bounds_generator': iter(SubvolumeBounds.iterable_from_csv(bounds_input_file.format(volume=k)))}
                for k in volumes.iterkeys()}
    else:
        if moves is None:
            moves = 5
        else:
            moves = np.asarray(moves)
        subv_shape = CONFIG.model.input_fov_shape + CONFIG.model.move_step * 2 * moves

        if sparse:
            gen_kwargs = {
                    k: {'sparse_margin': subv_shape}
                    for k in volumes.iterkeys()}
        else:
            gen_kwargs = {
                    k: {'shape': subv_shape}
                    for k in volumes.iterkeys()}
    subvolumes = [
            v.downsample(CONFIG.volume.resolution)
             .subvolume_generator(**gen_kwargs[k])
            for k, v in six.iteritems(volumes)]
    if augment:
        subvolumes = map(augment_subvolume_generator, subvolumes)
    regions = Roundrobin(*[Region.from_subvolume_generator(v, block_padding='reflect') for v in subvolumes])

    model = load_model(model_file, CONFIG.network)

    for region in regions:
        region.bias_against_merge = bias
        try:
            six.next(region.fill(
                    model,
                    progress=True,
                    move_batch_size=move_batch_size,
                    max_moves=max_moves,
                    remask_interval=remask_interval))
        except (StopIteration, Region.EarlyFillTermination):
            pass
        body = region.to_body()
        viewer = region.get_viewer()
        try:
            mask, bounds = body.get_seeded_component(CONFIG.postprocessing.closing_shape)
            viewer.add(mask.astype(np.float32),
                       name='Body Mask',
                       offset=bounds[0],
                       shader=get_color_shader(2))
        except ValueError:
            logging.info('Seed not in body.')
        print(viewer)
        while True:
            s = raw_input('Press Enter to continue, '
                          'v to open in browser, '
                          'a to export animation, '
                          'r to 3D render body, '
                          'q to quit...')
            if s == 'q':
                return
            elif s == 'a':
                region_copy = region.unfilled_copy()
                # Must assign the animation to a variable so that it is not GCed.
                ani = region_copy.fill_animation(  # noqa
                        'export.mp4',
                        model,
                        progress=True,
                        move_batch_size=move_batch_size,
                        max_moves=max_moves,
                        remask_interval=remask_interval)
                s = raw_input("Press Enter when animation is complete...")
            elif s == 'r':
                region.render_body()
            elif s == 'ra':
                region_copy = region.unfilled_copy()
                region_copy.fill_render(
                        model,
                        progress=True,
                        move_batch_size=move_batch_size,
                        max_moves=max_moves,
                        remask_interval=remask_interval)
            elif s == 's':
                body.to_swc('{}.swc'.format('_'.join(map(str, tuple(body.seed)))))
            elif s == 'v':
                viewer.open_in_browser()
            else:
                break


def fill_skeleton_with_model_threaded_new(
    model_file,
    seeds,
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
    num_workers=CONFIG.training.num_gpus,
    worker_prequeue=1,
    reject_early_termination=False,
    reject_non_seed_components=True,
    save_output_file=None,
):
    """
    Floodfill small regions around a list of seed points.
    Necessary inputs:
    - model_file
    - seeds
    - volume
    - config

    returns Dict[seed: Mask]
    """
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

    if volumes is None:
        raise ValueError("Volumes must be provided.")
    elif len(volumes) > 1:
        # TODO: support multiple volumes
        raise ValueError("WIP. multiple volumes not yet supported")

    # Get Volume
    vol_name = list(volumes.keys())[0]
    volume = volumes[vol_name].downsample(CONFIG.volume.resolution)
    
    # Seeds come in real coordinates
    seeds = [
        list(volume.world_coord_to_local(volume.real_coord_to_pixel(seed)))
        for seed in seeds
    ]
    region_shape = CONFIG.model.input_fov_shape

    pbar = tqdm(desc="Seed queue", total=len(seeds), miniters=1, smoothing=0.0)
    num_nodes = len(seeds)
    seed_generator = iter(seeds)

    manager = Manager()
    # Queue of seeds to be picked up by workers.
    seed_queue = manager.Queue()
    # Queue of results from workers.
    results_queue = manager.Queue()
    # Dequeue of seeds that were put in seed_queue but have not yet been
    # combined by the main process.
    dispatched_seeds = deque()
    # Seeds that were placed in seed_queue but subsequently covered by other
    # results before their results have been processed. This allows workers to
    # abort working on these seeds by checking this list.
    revoked_seeds = manager.list()
    # Results that have been received by the main process but have not yet
    # been combined because they were not received in the dispatch order.
    unordered_results = {}

    final_results = {}

    def queue_next_seed():
        total = 0
        for seed in seed_generator:
            if unordered_results.get(seed) is not None:
                # This seed has already been filled.
                total += 1
                continue
            dispatched_seeds.append(seed)
            seed_queue.put(seed)

            break

        return total

    for _ in range(min(num_nodes, num_workers * worker_prequeue)):
        processed_nodes = queue_next_seed()
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
                seed_queue,
                results_queue,
                loading_lock,
                revoked_seeds,
            ),
        )
        w.start()
        workers.append(w)

    while dispatched_seeds:
        processed_seeds = 1
        expected_seed = dispatched_seeds.popleft()
        logging.debug("Expecting seed %s", np.array_str(expected_seed))

        if tuple(expected_seed) in unordered_results:
            logging.debug(
                "Expected seed %s is in old results", np.array_str(expected_seed)
            )
            seed = expected_seed
            body = unordered_results[tuple(seed)]
            del unordered_results[tuple(seed)]  # WHY

        else:
            seed, body = results_queue.get(True)
            processed_seeds += queue_next_seed()

            while not np.array_equal(seed, expected_seed):
                logging.debug("Node %s is early, stashing", np.array_str(seed))
                unordered_results[tuple(seed)] = body
                seed, body = results_queue.get(True)
                processed_seeds += queue_next_seed()

        logging.debug("Processing node at %s", np.array_str(seed))
        pbar.update(processed_seeds)

        if final_results.get(seed) is not None:
            # This seed has already been filled.
            logging.debug(
                "Seed (%s) was filled but has been covered in the meantime.",
                np.array_str(seed),
            )
            loading_lock.acquire()
            if tuple(seed) in revoked_seeds:
                revoked_seeds.remove(tuple(seed))
            loading_lock.release()
            continue

        if body is None:
            logging.debug("Body of Seed ({}) is None".format(seed))
            
            # REDO THIS SEED
            raise NotImplementedError
            continue

        if not body.is_seed_in_mask():
            logging.debug("Seed ({}) is not in its body.".format(seed))
            
            # REDO THIS SEED
            raise NotImplementedError
            continue

        mask, bounds = body._get_bounded_mask(CONFIG.postprocessing.closing_shape)

        body_size = np.count_nonzero(mask)

        if body_size == 0:
            logging.debug("Body of seed {} is empty.".format(seed))
            
            # REDO THIS SEED
            raise NotImplementedError
            continue

        final_results[seed] = mask
        logging.debug("Filled seed ({})".format(seed))

    for _ in range(num_workers):
        seed_queue.put("DONE")
    for wid, worker in enumerate(workers):
        worker.join()
    manager.shutdown()

    pbar.close()

    return final_results


def fill_skeleton_with_model_threaded(
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
    num_workers=CONFIG.training.num_gpus,
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
            from keras import backend

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

            try:
                image = volume.get_subvolume(
                    SubvolumeBounds(
                        start=(node[2:]) - np.floor_divide(region_shape, 2),
                        stop=(node[2:]) + np.floor_divide(region_shape, 2) + 1,
                        node_id=node[0:2],
                    )
                ).image
            except Exception as e:
                logging.debug(
                    "COULD NOT GET IMAGE DATA DUE TO FOLLOWING ERROR:\n{}".format(e)
                )
                results.put((node, None))
                continue

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
                try:
                    assert all(CONFIG.model.input_fov_shape == np.array([25, 97, 97])), CONFIG.model.input_fov_shape
                except AssertionError:
                    logging.debug("Config inpug_fov_shape changed! Resetting it to 25x97x97")
                    CONFIG.model.input_fov_shape = np.array([25,97,97])
                #  logging.warn(backend.tensorflow_backend._get_available_gpus())
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

    if volumes is None:
        raise ValueError("Volumes must be provided.")
    elif len(volumes) > 1:
        # TODO: support multiple volumes
        raise ValueError("WIP. multiple volumes not yet supported")

    """
    Generate regions to fill:

    Each region should be a small area around a point on the skeleton.
    After flood filling all regions, I will look for intersections or the
    lack thereof that will indicate false mergers, allong with heavy
    segmentation perpendicular to skeleton that may indicate missing
    branches.

    """
    assert all(CONFIG.model.input_fov_shape == np.array([25, 97, 97])), CONFIG.model.input_fov_shape
    logging.debug("input_fov_shape: {}".format(CONFIG.model.input_fov_shape))

    vol_name = list(volumes.keys())[0]
    volume = volumes[vol_name].downsample(CONFIG.volume.resolution)
    seeds, ids = seeds_from_skeleton(skeleton_file)
    original_nodes = [ids[i] + seeds[i] for i in range(len(seeds))]
    #N5Volume
    """
    seeds = [
       list(volume.world_coord_to_local(volume.parent.real_coord_to_world(seed)))
       for seed in seeds
    ]
    """
    # Image Stack

    seeds = [
       list(volume.world_coord_to_local(volume.real_coord_to_world(seed)))
       for seed in seeds
    ]
    
    nodes = [np.array(ids[i] + seeds[i]) for i in range(len(seeds))]
    skel = sarbor.Skeleton()
    skel.input_nid_pid_x_y_z(original_nodes)
    region_shape = CONFIG.model.input_fov_shape
    region_shape += 2 * (region_shape // 4)

    # Only for N5Volume
    """
    volume.parent.initialize_many(volume.parent.image_data, [
       list(volume.local_coord_to_world(seed))
       for seed in seeds
    ], region_shape * volume.scale)
    """


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
                "Expected node %s is in old results", np.array_str(expected_node)
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
            logging.info("Filling {} failed".format(node[0]))
            logging.debug("No body was created")
            continue

        if not body.is_seed_in_mask():
            logging.info("Filling {} failed".format(node[0]))
            logging.debug("Seed (%s) is not in its body.", np.array_str(node[2:]))
            continue

        mask, bounds = body._get_bounded_mask(CONFIG.postprocessing.closing_shape)

        body_size = np.count_nonzero(mask)

        if body_size == 0:
            logging.info("Filling {} failed".format(node[0]))
            logging.debug("Body is empty.")
            continue

        logging.debug("Adding body to prediction label volume.")

        skel.fill(node[0], mask)

        logging.info("Filling {} succeeded".format(node[0]))

    for _ in range(num_workers):
        node_queue.put("DONE")
    for wid, worker in enumerate(workers):
        worker.join()
    manager.shutdown()

    pbar.close()

    results = [(node.key, node.parent_key, node.value.center, node.value.mask) for node in skel.get_nodes()]

    if save_output_file:
        import pickle
        pickle.dump(results, open(save_output_file + "_masks.obj", "wb"))

def evaluate_volume(
        volumes,
        gt_name,
        pred_name,
        partition=False,
        border_threshold=None,
        use_gt_mask=True,
        relabel=False):
    # TODO: This is very intrusive into Volumes and should be refactored to
    # handle much of the partioned access and resampling there.

    import cremi

    if partition:
        _, volumes = partition_volumes(volumes, downsample=False)

    def labels_to_cremi(v):
        label_data = v.label_data.copy()
        if hasattr(v, 'bounds'):
            label_data = label_data[list(map(slice, list(v.bounds[0]), list(v.bounds[1])))]
        volume = cremi.Volume(label_data, resolution=v.resolution)

        return volume

    gt_vol = volumes[gt_name]
    pred_vol = volumes[pred_name]
    logging.info('GT shape: %s\t Prediction shape:%s', gt_vol.shape, pred_vol.shape)

    pred_upsample = gt_vol._get_downsample_from_resolution(pred_vol.resolution)
    if np.any(pred_upsample > 0):
        scale = np.exp2(pred_upsample).astype(np.int64)
        logging.warn('Segmentation is different resolution than groundtruth. Upsampling by %s.', scale)

        pred_data = pred_vol.label_data
        if hasattr(pred_vol, 'bounds'):
            pred_data = pred_data[list(map(slice, list(pred_vol.bounds[0]), list(pred_vol.bounds[1])))]
        orig_shape = pred_data.shape
        pred_data = np.lib.stride_tricks.as_strided(pred_data,
                                                    [b for a in zip(orig_shape, scale) for b in a],
                                                    [b for a in zip(pred_data.strides, [0, 0, 0]) for b in a])
        new_shape = np.array(orig_shape) * scale
        pred_data = pred_data.reshape(list(new_shape))

        padding = np.array(gt_vol.shape) - new_shape
        if np.any(padding > 0):
            logging.warn('Padding segmentation (%s) to be groundtruth size (%s)', new_shape, gt_vol.shape)
            pred_data = np.pad(pred_data, zip([0, 0, 0], list(padding)), 'edge')

        pred = cremi.Volume(pred_data, resolution=gt_vol.resolution)
    else:
        pred = labels_to_cremi(pred_vol)

    gt = labels_to_cremi(gt_vol)

    # Some augmented CREMI volumes have not just a uint64 -1 as background, but
    # several large values. Set these all to background to avoid breaking
    # coo_matrix.
    gt.data[gt.data > np.uint64(-10)] = np.uint64(-1)
    background_label_id = 0
    pred.data[pred.data > np.uint64(-10)] = background_label_id

    if use_gt_mask and gt_vol.mask_data is not None:
        logging.warn('Groundtruth has a mask channel that will be applied to segmentation.')
        mask_data = gt_vol.mask_data
        if hasattr(gt_vol, 'bounds'):
            mask_data = mask_data[list(map(slice, list(gt_vol.bounds[0]), list(gt_vol.bounds[1])))]

        if relabel:
            mask_exiting_bodies = np.unique(pred.data[np.logical_not(mask_data)])

        pred.data[np.logical_not(mask_data)] = background_label_id

        if relabel:
            from skimage import morphology

            pred_copy = np.zeros_like(pred.data)
            exiting_bodies_mask = np.isin(pred.data, mask_exiting_bodies)
            pred_copy[exiting_bodies_mask] = pred.data[exiting_bodies_mask]

            new_pred = morphology.label(pred_copy, background=background_label_id, connectivity=2)

            pred.data[exiting_bodies_mask] = new_pred[exiting_bodies_mask]

    gt_neuron_ids = cremi.evaluation.NeuronIds(gt, border_threshold=border_threshold)

    (voi_split, voi_merge) = gt_neuron_ids.voi(pred)
    adapted_rand = gt_neuron_ids.adapted_rand(pred)

    print('VOI split         :', voi_split)
    print('VOI merge         :', voi_merge)
    print('Adapted Rand-index:', adapted_rand)
    print('CREMI             :', np.sqrt((voi_split + voi_merge) * adapted_rand))


def view_volumes(volumes, partition=False):
    """Display a set of volumes together in a neuroglancer viewer.

    Parameters
    ----------
    volumes : dict
        Dictionary mapping volume name to diluvian.volumes.Volume.
    partition : bool
        If true, partition the volumes and put the view origin at the validaiton
        partition origin.
    """

    if partition:
        _, volumes = partition_volumes(volumes, downsample=False)

    viewer = WrappedViewer()

    for volume_name, volume in six.iteritems(volumes):
        resolution = list(np.flipud(volume.resolution))
        offset = getattr(volume, 'bounds', [np.zeros(3, dtype=np.int32)])[0]
        offset = np.flipud(-offset)

        viewer.add(volume.image_data,
                   name='{} (Image)'.format(volume_name),
                   voxel_size=resolution,
                   voxel_offset=offset)
        if volume.label_data is not None:
            viewer.add(volume.label_data,
                       name='{} (Labels)'.format(volume_name),
                       voxel_size=resolution,
                       voxel_offset=offset)
        if volume.mask_data is not None:
            viewer.add(volume.mask_data,
                       name='{} (Mask)'.format(volume_name),
                       voxel_size=resolution,
                       voxel_offset=offset)

    viewer.print_view_prompt()
