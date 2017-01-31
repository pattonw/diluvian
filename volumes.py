# -*- coding: utf-8 -*-


import h5py
import numpy as np
import pytoml as toml

from keras.utils.data_utils import get_file

from config import CONFIG
from regions import DenseRegion
from util import pad_dims


class HDF5Volume(object):
    @staticmethod
    def from_toml(filename):
        volumes = {}
        with open(filename, 'rb') as fin:
            datasets = toml.load(fin).get('dataset', [])
            for dataset in datasets:
                hdf5_file = dataset['hdf5_file']
                if dataset.get('use_keras_cache', False):
                    hdf5_file = get_file(hdf5_file, dataset['download_url'], md5_hash=dataset.get('download_md5', None))
                volumes[dataset['name']] = HDF5Volume(hdf5_file,
                                                      dataset['image_dataset'],
                                                      dataset['label_dataset'])

        return volumes

    def __init__(self, orig_file, image_dataset, label_dataset):
        self.file = h5py.File(orig_file, 'r')
        self.image_data = self.file[image_dataset]
        self.label_data = self.file[label_dataset]

    def simple_training_generator(self, subvolume_size, batch_size, training_size, f_a_bins=None, partition=None):
        subvolumes = self.SubvolumeGenerator(self, subvolume_size, CONFIG.volume.downsample, partition)

        mask_input = np.full(np.append(subvolume_size, (1,)), CONFIG.model.v_false, dtype='float32')
        mask_input[tuple(np.array(mask_input.shape) / 2)] = CONFIG.model.v_true
        mask_input = np.tile(mask_input, (batch_size, 1, 1, 1, 1))

        if f_a_bins is not None:
            f_a_counts = np.zeros_like(f_a_bins, dtype='uint64')
        f_as = np.zeros(batch_size)

        sample_num = 0
        while 1:
            if sample_num >= training_size:
                subvolumes.reset()
                sample_num = 0

            batch_image_input = [None] * batch_size
            batch_mask_target = [None] * batch_size

            for batch_ind in range(0, batch_size):
                subvolume = subvolumes.next()

                batch_image_input[batch_ind] = pad_dims(subvolume['image'])
                batch_mask_target[batch_ind] = pad_dims(subvolume['mask_target'])
                f_as[batch_ind] = subvolume['f_a']

            batch_image_input = np.concatenate(batch_image_input)
            batch_mask_target = np.concatenate(batch_mask_target)

            sample_num += batch_size

            if f_a_bins is None:
                yield ({'image_input': batch_image_input,
                        'mask_input': mask_input},
                       [batch_mask_target])
            else:
                f_a_inds = np.digitize(f_as, f_a_bins) - 1
                inds, counts = np.unique(f_a_inds, return_counts=True)
                f_a_counts[inds] += counts.astype('uint64')
                sample_weights = np.reciprocal(f_a_counts[f_a_inds], dtype='float64')
                yield ({'image_input': batch_image_input,
                        'mask_input': mask_input},
                       [batch_mask_target],
                       sample_weights)

    def moving_training_generator(self, subvolume_size, batch_size, training_size, callback_kludge, f_a_bins=None, partition=None):
        subvolumes = self.SubvolumeGenerator(self, subvolume_size, CONFIG.volume.downsample, partition)

        regions = [None] * batch_size
        region_pos = [None] * batch_size

        if f_a_bins is not None:
            f_a_counts = np.zeros_like(f_a_bins, dtype='uint64')
        f_as = np.zeros(batch_size)

        sample_num = 0
        while 1:
            if sample_num >= training_size:
                subvolumes.reset()
                sample_num = 0

            # Before clearing last batches, reuse them to predict mask outputs
            # for move training. Add mask outputs to regions.
            active_regions = [n for n, region in enumerate(regions) if region is not None]
            if active_regions and callback_kludge['outputs'] is not None:
                for n in active_regions:
                    assert np.array_equal(callback_kludge['inputs']['image_input'][n, 0, 0, :, 0], batch_image_input[n, 0, 0, :, 0])
                    regions[n].add_mask(callback_kludge['outputs'][n, :, :, :, 0], region_pos[n])

            batch_image_input = [None] * batch_size
            batch_mask_input = [None] * batch_size
            batch_mask_target = [None] * batch_size

            for r, region in enumerate(regions):
                if region is None or region.queue.empty():
                    subvolume = subvolumes.next()

                    regions[r] = DenseRegion(subvolume['image'], subvolume['mask_target'])
                    region = regions[r]

                block_data = region.get_next_block()

                batch_image_input[r] = pad_dims(block_data['image'])
                batch_mask_input[r] = pad_dims(block_data['mask'])
                batch_mask_target[r] = pad_dims(block_data['target'])
                region_pos[r] = block_data['position']
                f_as[r] = subvolume['f_a']

            batch_image_input = np.concatenate(batch_image_input)
            batch_mask_input = np.concatenate(batch_mask_input)
            batch_mask_target = np.concatenate(batch_mask_target)

            sample_num += batch_size
            inputs = {'image_input': batch_image_input,
                      'mask_input': batch_mask_input}
            callback_kludge['inputs'] = inputs
            callback_kludge['outputs'] = None

            if f_a_bins is None:
                yield (inputs,
                       [batch_mask_target])
            else:
                f_a_inds = np.digitize(f_as, f_a_bins) - 1
                inds, counts = np.unique(f_a_inds, return_counts=True)
                f_a_counts[inds] += counts.astype('uint64')
                sample_weights = np.reciprocal(f_a_counts[f_a_inds], dtype='float64')
                yield (inputs,
                       [batch_mask_target],
                       sample_weights)

    def region_generator(self, subvolume_size, partition=None, seed_margin=None):
        subvolumes = self.SubvolumeGenerator(self, subvolume_size, CONFIG.volume.downsample, partition)

        if seed_margin is None:
            seed_margin = 10.0

        margin = np.ceil(np.reciprocal(np.array(CONFIG.volume.resolution), dtype='float64') * seed_margin).astype('int64')

        while 1:
            subvolume = subvolumes.next()
            mask_target = subvolume['mask_target']
            ctr = np.array(mask_target.shape) / 2
            seed_region = mask_target[ctr[0] - margin[0]:
                                      ctr[0] + margin[0] + 1,
                                      ctr[1] - margin[1]:
                                      ctr[1] + margin[1] + 1,
                                      ctr[2] - margin[2]:
                                      ctr[2] + margin[2] + 1]
            if not np.unique(seed_region).size == 1:
                print 'Rejecting region with seed margin too small.'
                continue
            region = DenseRegion(subvolume['image'], mask_target)
            yield region


    class SubvolumeGenerator(object):
        def __init__(self, volume, size_zoom, downsample, partition=None):
            if partition is None:
                partition = (np.array((1, 1, 1)), np.array((0, 0, 0)))
            self.volume = volume
            self.partition = partition
            self.zoom = np.exp2(downsample).astype('int64')
            self.size_zoom = size_zoom
            self.size_orig = np.multiply(self.size_zoom, self.zoom)
            self.margin = np.floor_divide(self.size_orig, 2)
            # HDF5 coordinates are z, y, x
            self.partition_size = np.floor_divide(np.flipud(np.array(self.volume.image_data.shape)), self.partition[0])
            self.ctr_min = np.multiply(self.partition_size, self.partition[1]) + self.margin
            self.ctr_max = np.multiply(self.partition_size, self.partition[1] + 1) - self.margin - 1
            self.random = np.random.RandomState(0)

        def __iter__(self):
            return self

        def reset(self):
            self.random.seed(0)

        def next(self):
            ctr = tuple(self.random.randint(self.ctr_min[n], self.ctr_max[n]) for n in range(0, 3))
            subvol = ((ctr[2] - self.margin[2], ctr[2] + self.margin[2] + (self.size_orig[2] % 2)),
                      (ctr[1] - self.margin[1], ctr[1] + self.margin[1] + (self.size_orig[1] % 2)),
                      (ctr[0] - self.margin[0], ctr[0] + self.margin[0] + (self.size_orig[0] % 2)))
            image_subvol = self.volume.image_data[subvol[0][0]:subvol[0][1],
                                      subvol[1][0]:subvol[1][1],
                                      subvol[2][0]:subvol[2][1]]
            label_subvol = self.volume.label_data[subvol[0][0]:subvol[0][1],
                                      subvol[1][0]:subvol[1][1],
                                      subvol[2][0]:subvol[2][1]]

            image_subvol = np.transpose(image_subvol.astype('float32')) / 256.0
            label_subvol = np.transpose(label_subvol)
            label_id = label_subvol[tuple(np.array(label_subvol.shape) / 2)]
            label_mask = label_subvol == label_id

            if np.any(self.zoom > 1):
                image_subvol = image_subvol.reshape([self.size_zoom[0], self.zoom[0],
                                                     self.size_zoom[1], self.zoom[1],
                                                     self.size_zoom[2], self.zoom[2]]).mean(5).mean(3).mean(1)
                label_mask = label_mask.reshape([self.size_zoom[0], self.zoom[0],
                                                 self.size_zoom[1], self.zoom[1],
                                                 self.size_zoom[2], self.zoom[2]]).all(5).all(3).all(1)
                # A higher fidelity alternative would be to use the mode label
                # for each downsample block. However, this is prohibitively
                # slow using the scipy code preserved below as an example:
                # label_mask = label_mask.reshape([self.size_zoom[0], self.zoom[0],
                #                                  self.size_zoom[1], self.zoom[1],
                #                                  self.size_zoom[2], self.zoom[2]])
                # label_mask = stats.mode(label_mask, 5)[0]
                # label_mask = stats.mode(label_mask, 3)[0]
                # label_mask = np.squeeze(stats.mode(label_mask, 1)[0])

            assert image_subvol.shape == tuple(self.size_zoom), 'Image wrong size: {}'.format(image_subvol.shape)
            assert label_mask.shape == tuple(self.size_zoom), 'Labels wrong size: {}'.format(label_mask.shape)

            f_a = np.count_nonzero(label_mask) / float(label_mask.size)
            mask_target = np.full_like(label_mask, CONFIG.model.v_false, dtype='float32')
            mask_target[label_mask] = CONFIG.model.v_true

            return {'image': image_subvol, 'mask_target': mask_target, 'f_a': f_a}
