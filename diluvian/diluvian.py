# -*- coding: utf-8 -*-


import importlib
import itertools

import matplotlib as mpl
# Use the 'Agg' backend to allow the generation of plots even if no X server
# is available. The matplotlib backend must be set before importing pyplot.
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import neuroglancer
import numpy as np

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model

from .config import CONFIG
from .network import compile_network
from .third_party.multi_gpu import make_parallel
from .util import extend_keras_history, get_color_shader, roundrobin, write_keras_history_to_csv
from .volumes import SubvolumeBounds, static_training_generator, moving_training_generator
from .regions import DenseRegion


def plot_history(history):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    fig.suptitle('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper right')

    return fig


class PredictionCopy(Callback):
    """Keras batch end callback to run prediction on input from a kludge.

    Used to predict masks for FOV moving. Surprisingly this is faster than
    using a custom Keras training function to copy model predictions at the
    same time as gradient updates.
    """
    def __init__(self, kludge):
        self.kludge = kludge

    def on_batch_end(self, batch, logs={}):
        if self.kludge['inputs'] and self.kludge['outputs'] is None:
            self.kludge['outputs'] = self.model.predict(self.kludge['inputs'])


def generate_subvolume_bounds(filename, volumes, num_bounds, sparse=False):
    if '{volume}' not in filename:
        raise ValueError('CSV filename must contain "{volume}" for volume name replacement.')

    if sparse:
        gen_kwargs = {'sparse_margin': CONFIG.model.training_subv_shape * 4 - 3}
    else:
        gen_kwargs = {'shape': CONFIG.model.training_subv_shape * 4 - 3}
    for k, v in volumes.iteritems():
        bounds = v.downsample(CONFIG.volume.resolution)\
                  .subvolume_bounds_generator(**gen_kwargs)
        bounds = itertools.islice(bounds, num_bounds)
        SubvolumeBounds.iterable_to_csv(bounds, filename.format(volume=k))


def fill_region_from_model(model_file, volumes=None, bounds_input_file=None,
                           bias=True, move_batch_size=1,
                           max_moves=None, multi_gpu_model_kludge=None, sparse=False):
    if volumes is None:
        raise ValueError('Volumes must be provided.')

    if bounds_input_file is not None:
        gen_kwargs = {k: {
                  'bounds_generator': iter(SubvolumeBounds.iterable_from_csv(bounds_input_file.format(volume=k)))}
                  for k in volumes.iterkeys()}
    else:
        if sparse:
            gen_kwargs = {k: {'sparse_margin': CONFIG.model.training_subv_shape * 4 - 3} for k in volumes.iterkeys()}
        else:
            gen_kwargs = {k: {'shape': CONFIG.model.training_subv_shape * 4 - 3} for k in volumes.iterkeys()}
    regions = roundrobin(*[
            DenseRegion.from_subvolume_generator(
                v.downsample(CONFIG.volume.resolution)
                 .subvolume_generator(**gen_kwargs[k]))
            for k, v in volumes.iteritems()])

    model = load_model(model_file)

    for region in regions:
        region.bias_against_merge = bias
        region.fill(model,
                    verbose=True,
                    move_batch_size=move_batch_size,
                    max_moves=max_moves,
                    multi_gpu_pad_kludge=multi_gpu_model_kludge)
        viewer = region.get_viewer()
        print viewer
        s = raw_input("Press Enter to continue, a to export animation, q to quit...")
        if s == 'q':
            break
        elif s == 'a':
            region_copy = region.unfilled_copy()
            # Must assign the animation to a variable so that it is not GCed.
            ani = region_copy.fill_animation(model, 'export.mp4', verbose=True) # noqa
            s = raw_input("Press Enter when animation is complete...")
        elif s == 's':
            body = region.to_body()
            body.to_swc('{}.swc'.format('_'.join(map(str, tuple(body.seed)))))


def train_network(model_file=None, volumes=None, static_validation=True,
                  model_output_filebase=None, model_checkpoint_file=None,
                  tensorboard=False, viewer=False, metric_plot=False):
    if model_file is None:
        factory_mod_name, factory_func_name = CONFIG.network.factory.rsplit('.', 1)
        factory_mod = importlib.import_module(factory_mod_name)
        factory = getattr(factory_mod, factory_func_name)
        ffn = factory(CONFIG.model.fov_shape, CONFIG.network)
    else:
        ffn = load_model(model_file)

    # Multi-GPU models are saved as a single-GPU model prior to compilation,
    # so if loading from such a model file it will need to be recompiled.
    if not hasattr(ffn, 'optimizer'):
        if CONFIG.training.num_gpus > 1:
            ffn = make_parallel(ffn, CONFIG.training.num_gpus)
        compile_network(ffn, CONFIG.optimizer)

    if model_output_filebase is None:
        model_output_filebase = 'model_output'

    if volumes is None:
        raise ValueError('Volumes must be provided.')

    CONFIG.to_toml(model_output_filebase + '.toml')

    f_a_bins = CONFIG.training.fill_factor_bins

    num_volumes = len(volumes)

    training_volumes = {
            k: v.partition(CONFIG.training.partitions, CONFIG.training.training_partition)
                .downsample(CONFIG.volume.resolution)
            for k, v in volumes.iteritems()}
    validation_volumes = {
            k: v.partition(CONFIG.training.partitions, CONFIG.training.validation_partition)
                .downsample(CONFIG.volume.resolution)
            for k, v in volumes.iteritems()}

    if static_validation:
        validation_data = {k: static_training_generator(
                v.subvolume_generator(shape=CONFIG.model.fov_shape),
                CONFIG.training.batch_size,
                CONFIG.training.validation_size,
                f_a_bins=f_a_bins) for k, v in validation_volumes.iteritems()}
    else:
        validation_kludges = {k: {'inputs': None, 'outputs': None} for k in volumes.iterkeys()}
        validation_data = {k: moving_training_generator(
                v.subvolume_generator(shape=CONFIG.model.training_subv_shape),
                CONFIG.training.batch_size,
                CONFIG.training.validation_size,
                validation_kludges[k],
                f_a_bins=f_a_bins) for k, v in validation_volumes.iteritems()}
    validation_data = roundrobin(*validation_data.values())

    # Pre-train
    training_data = {k: static_training_generator(
            v.subvolume_generator(shape=CONFIG.model.fov_shape),
            CONFIG.training.batch_size,
            CONFIG.training.training_size,
            f_a_bins=f_a_bins) for k, v in training_volumes.iteritems()}
    training_data = roundrobin(*training_data.values())
    history = ffn.fit_generator(
            training_data,
            samples_per_epoch=CONFIG.training.training_size * num_volumes,
            nb_epoch=CONFIG.training.static_train_epochs,
            validation_data=validation_data,
            nb_val_samples=CONFIG.training.validation_size * num_volumes)

    # Moving training
    kludges = {k: {'inputs': None, 'outputs': None} for k in volumes.iterkeys()}
    callbacks = [PredictionCopy(kludge) for kludge in kludges.values()]
    callbacks.append(ModelCheckpoint(model_output_filebase + '.hdf5', save_best_only=True))
    if model_checkpoint_file:
        callbacks.append(ModelCheckpoint(model_checkpoint_file))
    callbacks.append(EarlyStopping(patience=CONFIG.training.patience))
    if tensorboard:
        callbacks.append(TensorBoard())

    training_data = {k: moving_training_generator(
            v.subvolume_generator(shape=CONFIG.model.training_subv_shape),
            CONFIG.training.batch_size,
            CONFIG.training.training_size,
            kludges[k],
            f_a_bins=f_a_bins) for k, v in training_volumes.iteritems()}
    training_data = roundrobin(*training_data.values())
    moving_history = ffn.fit_generator(
            training_data,
            samples_per_epoch=CONFIG.training.training_size * num_volumes,
            nb_epoch=CONFIG.training.total_epochs,
            initial_epoch=CONFIG.training.static_train_epochs,
            max_q_size=num_volumes,
            nb_worker=1,
            callbacks=callbacks,
            validation_data=validation_data,
            nb_val_samples=CONFIG.training.validation_size * num_volumes)
    extend_keras_history(history, moving_history)

    write_keras_history_to_csv(history, model_output_filebase + '.csv')

    if viewer:
        # for _ in itertools.islice(training_data, 12):
        #     continue
        dupe_data = static_training_generator(
                volumes[list(volumes.keys())[0]].subvolume_generator(shape=CONFIG.model.fov_shape),
                CONFIG.training.batch_size,
                CONFIG.training.training_size)
        viz_ex = itertools.islice(dupe_data, 1)

        for inputs, targets in viz_ex:
            viewer = neuroglancer.Viewer(voxel_size=list(CONFIG.volume.resolution))
            viewer.add(np.transpose(inputs['image_input'][0, :, :, :, 0]),
                       name='Image')
            viewer.add(np.transpose(inputs['mask_input'][0, :, :, :, 0]),
                       name='Mask Input',
                       shader=get_color_shader(2))
            viewer.add(np.transpose(targets[0][0, :, :, :, 0]),
                       name='Mask Target',
                       shader=get_color_shader(0))
            output = ffn.predict(inputs)
            viewer.add(np.transpose(output[0, :, :, :, 0]),
                       name='Mask Output',
                       shader=get_color_shader(1))
            print viewer

            raw_input("Press any key to exit...")

    if metric_plot:
        fig = plot_history(history)
        fig.savefig(model_output_filebase + '.png')

    return history
