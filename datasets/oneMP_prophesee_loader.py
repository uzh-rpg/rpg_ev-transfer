import os
import tqdm
import h5py
import torch
import IPython
import numpy as np
from os import listdir
from torch.utils.data import Dataset
from numpy.lib import recfunctions as rfn

import datasets.data_util as data_util


class OneMPProphesee(Dataset):
    def __init__(self, root, height=None, width=None, nr_events_window=None, augmentation=False, mode='train',
                 event_representation='histogram', nr_temporal_bins=5):
        """
        Creates an iterator over the 1 Mp Prophesee object recognition dataset.

        :param root: path to dataset root
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'train', 'test' or 'val'
        :param event_representation: 'histogram' or 'voxel_grid'
        """
        self.root = root
        self.files = listdir(os.path.join(self.root, mode))
        # Get numpy bounding box files
        self.files = [file_name[:-9] for file_name in self.files if file_name[-4:] == '.npy']

        self.mode = mode
        self.width = width
        self.height = height
        self.original_width = 1280
        self.original_height = 720
        self.augmentation = augmentation
        self.event_representation = event_representation
        self.nr_events_window = nr_events_window

        self.nr_classes = 2
        self.class_list = ['Car']
        # {"0": "pedestrian", "1": "two wheeler", "2": "car", "3": "truck", "4": "bus", "5": "traffic sign", "6": "traffic light"}
        self.class_to_use = [2]

        self.createAllBBoxDataset()
        self.nr_samples = len(self.files)

        if self.mode == 'val':
            import random
            random.seed(7)
            random.shuffle(self.files)
            self.files = self.files[:7000]

        else:
            import random
            random.seed(7)
            random.shuffle(self.files)

        self.valid_samples = np.ones([self.__len__()])

    def __len__(self):
        return len(self.files)

    def createAllBBoxDataset(self):
        """
        Iterates over the files and stores for each unique bounding box timestep the file name and the index of the
         unique indices file.
        """
        print('Building 1MP Prophesee Dataset')
        pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)

        # If not already created, create a look up table between bounding boxes and event index
        original_shape_hw = [self.original_height, self.original_width]
        target_ratio = float(self.height) / float(self.width)
        unscaled_target_height = int(original_shape_hw[1] * target_ratio)
        cropped_height = int(original_shape_hw[0] - unscaled_target_height)

        for i_file, file_name in enumerate(self.files):
            bbox_to_event_idx_name = os.path.join(self.root, 'bbox_to_idx', self.mode,
                                                  file_name + '_bbox_to_event_idx.npy')
            if os.path.isfile(bbox_to_event_idx_name):
                pbar.update(1)
                continue
            bbox_to_event_idx = []

            bbox_file_name = os.path.join(self.root, self.mode, file_name + '_bbox.npy')
            event_file_name = os.path.join(self.root, self.mode, file_name + '_td.h5')

            bboxes = np.load(bbox_file_name)
            t_bboxes = bboxes['t']

            unique_ts, unique_indices = np.unique(t_bboxes, return_index=True)
            for i, unique_time in enumerate(unique_ts):
                bbox_array = self.get_bounding_boxes(bbox_file_name, i, original_shape_hw,
                                                     unscaled_target_height, cropped_height)
                if bbox_array.shape[0] > 0:
                    sequence_start = self.searchClosestEventTimestamp(event_file_name, unique_time)
                    bbox_to_event_idx.append([i, sequence_start])

            np.save(bbox_to_event_idx_name, np.array(bbox_to_event_idx))
            pbar.update(1)

        pbar.close()

        file_name_frame_bbox_id = []
        for i_file, file_name in enumerate(self.files):
            bbox_to_event_idx = np.load(os.path.join(self.root, 'bbox_to_idx', self.mode,
                                                     file_name + '_bbox_to_event_idx.npy'))
            if bbox_to_event_idx.shape[0] == 0:
                continue
            file_name_frame_bbox_id += [[file_name, idx[0], idx[1]] for idx in bbox_to_event_idx]

        self.files = file_name_frame_bbox_id

    def scale_bounding_boxes(self, bbox_array, original_shape_wh, cropped_height):
        """Adjusts and scales the bounding boxes to the specified height and shape considering the center crop"""
        bbox_array[:, [0, 2]] = bbox_array[:, [0, 2]] - cropped_height
        bbox_array[:, [0, 2]] = bbox_array[:, [0, 2]] * float(self.height) / float(original_shape_wh[1])
        bbox_array[:, [1, 3]] = bbox_array[:, [1, 3]] * float(self.width) / float(original_shape_wh[0])

        bbox_array = self.check_bbox_for_boundaries(bbox_array)

        # Convert from x_max, y_max to width and height
        bbox_array[:, [2, 3]] = bbox_array[:, [2, 3]] - bbox_array[:, [0, 1]]

        # Remove Bounding Boxes with diagonal smaller than specifies square pixels
        valid_bbox = (bbox_array[:, 2]**2 + bbox_array[:, 3]**2) > 500
        valid_bbox = valid_bbox * (bbox_array[:, 2] > 5) * (bbox_array[:, 3] > 5)
        bbox_array = bbox_array[valid_bbox, :]

        return bbox_array.astype(np.int32)

    def __getitem__(self, idx):
        """
        returns event in the specified representation, labels ['x', 'y', 'w', 'h', 'class_id'].

        :param idx:
        """
        if self.valid_samples[idx] == 0:
            return self.__getitem__(idx - 1)

        bbox_file_path = os.path.join(self.root, self.mode, self.files[idx][0] + '_bbox.npy')
        event_file_path = os.path.join(self.root, self.mode, self.files[idx][0] + '_td.h5')

        # ----- Events -----
        closest_event_index = self.files[idx][2]
        events = self.readEventFile(event_file_path, closest_event_index, nr_window_events=self.nr_events_window)
        event_tensor = data_util.generate_input_representation(events, event_representation=self.event_representation,
                                                               shape=(self.original_height, self.original_width))
        original_shape_hw = [self.original_height, self.original_width]
        target_ratio = float(self.height) / float(self.width)
        unscaled_target_height = int(original_shape_hw[1] * target_ratio)
        cropped_height = int(original_shape_hw[0] - unscaled_target_height)

        event_tensor = torch.from_numpy(data_util.normalize_event_tensor(event_tensor[:, cropped_height:, :]))
        event_tensor = torch.nn.functional.interpolate(event_tensor[None, :, :, :],
                                                       size=(self.height, self.width),
                                                       mode='bilinear', align_corners=True)[0, :, :, :]

        # Filtering out not well-distributed event samples
        nr_samples_quad = event_tensor.view(2, 2, self.height // 2, 2, self.width // 2).sum(dim=(0, 2, 4)) / event_tensor.sum()
        skip_bool = nr_samples_quad < 0.01

        if skip_bool.any():
            self.valid_samples[idx] = 0

            return self.__getitem__(idx - 1)


        bbox_time_idx = self.files[idx][1]
        bbox_array = self.get_bounding_boxes(bbox_file_path, bbox_time_idx, original_shape_hw, unscaled_target_height,
                                             cropped_height)

        return event_tensor, bbox_array.astype(np.int32)

    def searchClosestEventTimestamp(self, event_file_name, bbox_time):
        """
        Code adapted from:
        https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/io/psee_loader.py

        go to the time final_time inside the file. This is implemented using a binary search algorithm
        :param final_time: expected time
        :param term_cirterion: (nb event) binary search termination criterion
        it will load those events in a buffer and do a numpy searchsorted so the result is always exact
        """
        event_h5_file = h5py.File(event_file_name, 'r')
        term_criterion = 1000
        event_time_h5 = event_h5_file['events']['t']
        nr_events = event_time_h5.shape[0]
        low = 0
        high = nr_events

        # binary search
        while high - low > term_criterion:
            middle = (low + high) // 2
            mid = event_time_h5[middle]

            if mid > bbox_time:
                high = middle
            elif mid < bbox_time:
                low = middle + 1
            else:
                low = middle
                break

        event_h5_file.close()

        return low

    def readEventFile(self, event_file, event_index, nr_window_events=250000):
        event_h5_file = h5py.File(event_file, 'r')
        start_index = max(event_index - nr_window_events // 2, 0)
        start_index = min(start_index, event_h5_file['events']['x'].shape[0] - (nr_window_events + 1))
        x = event_h5_file['events']['x'][start_index:(start_index+nr_window_events)]
        y = event_h5_file['events']['y'][start_index:(start_index+nr_window_events)]
        p = event_h5_file['events']['p'][start_index:(start_index+nr_window_events)]
        t = event_h5_file['events']['t'][start_index:(start_index+nr_window_events)]

        event_h5_file.close()
        events_np = np.stack([x, y, t, p], axis=-1)

        return events_np

    def get_bounding_boxes(self, bbox_file_path, bbox_time_idx, original_shape_hw, unscaled_target_height,
                           cropped_height):
        # ----- Bounding Box -----
        # [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'),
        # ('class_confidence', '<f4'), ('track_id', '<u4')]
        bboxes_structured = np.load(bbox_file_path)

        unique_ts, unique_indices = np.unique(bboxes_structured['t'], return_index=True)
        nr_unique_ts = unique_ts.shape[0]

        # Get bounding multiple boxes at current timestep
        start_idx = unique_indices[bbox_time_idx]
        if bbox_time_idx == (nr_unique_ts - 1):
            end_idx = bboxes_structured['t'].shape[0]
        else:
            end_idx = unique_indices[bbox_time_idx+1]

        bboxes = bboxes_structured[start_idx:end_idx]

        # Stored Dimensions ['u', 'v', 'w', 'h', 'class_id']
        bbox_array = rfn.structured_to_unstructured(bboxes)[:, [1, 2, 3, 4, 5]]

        # Get Car Class
        bbox_array = bbox_array[bbox_array[:, -1] == 2, :]
        bbox_array[:, -1] = 0

        # Change to ['x_min', 'y_min', 'x_max', 'y_max', 'class_id']
        bbox_array = bbox_array[:, [1, 0, 3, 2, 4]]
        bbox_array[:, [2, 3]] += bbox_array[:, :2]

        return self.scale_bounding_boxes(bbox_array, (original_shape_hw[1], unscaled_target_height), cropped_height)

    def check_bbox_for_boundaries(self, bbox_array):
        """Check bounding boxes for dimensions"""
        bbox_array[:, :4] = np.maximum(bbox_array[:, :4], 0)
        bbox_array[:, 2] = np.minimum(bbox_array[:, 2], self.height - 1)
        bbox_array[:, 3] = np.minimum(bbox_array[:, 3], self.width - 1)

        return bbox_array
