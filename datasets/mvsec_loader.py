import os
import torch
import h5py
import numpy as np

from torch.utils.data import Dataset
import datasets.data_util as data_util


class MVSEC_Events(Dataset):
    def __init__(self, root, height=None, width=None, nr_events_window=None, augmentation=False, mode='train',
                 event_representation=None, nr_temporal_bins=5):
        """
        Creates an iterator over the mvsec event object recognition dataset.

        :param root: path to dataset root
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: not used in this class
        :param augmentation: not used in this class
        :param mode: 'train', 'test' or 'val'
        :param event_representation: not used in this class
        """
        self.root = root
        self.event_representation = event_representation
        self.nr_events_window = nr_events_window
        self.nr_temporal_bins = nr_temporal_bins
        self.mode = mode
        self.augmentation = augmentation

        if mode == 'train':
            self.sequence_list = ['outdoor_day1_data', 'outdoor_day2_data']
            self.use_labels = False
        elif mode == 'val':
            # For the ground truth labels,
            self.sequence_list = ['outdoor_day1_data']
            self.use_labels = True
        elif mode == 'test':
            self.sequence_list = ['outdoor_day2_data']
            self.use_labels = True

        self.height = height
        self.width = width
        self.original_height = 260
        self.original_width = 346

        self.class_list = ['Car']

        self.frame_file_list = []
        self.frame_bbox_list = []
        if self.use_labels:
            self.extract_labeled_data()
        else:
            self.extract_unlabeled_data()

    def __len__(self):
        return len(self.frame_file_list)

    def extract_labeled_data(self):
        """Extracts the bounding boxes and the corresponding events"""
        # Extract bounding boxes
        gt_path = os.path.join(self.root, self.sequence_list[0], 'labels')
        ground_truth_file_list = os.listdir(gt_path)
        ground_truth_file_list.sort()
        gt_file_id = []
        for file in ground_truth_file_list:
            with open(os.path.join(gt_path, file)) as f:
                frame_list = []
                for line in f:
                    splitted_line = line.split(' ')
                    y_min, x_min, y_max, x_max = float(splitted_line[-4]), float(splitted_line[-3]), \
                                                 float(splitted_line[-2]), float(splitted_line[-1])
                    if splitted_line[0] == 'Car':
                        class_id = 0
                        frame_list.append(np.array([x_min, y_min, x_max, y_max, class_id]))
                if len(frame_list) != 0:
                    gt_file_id.append(float(file[:-4]))
                    self.frame_bbox_list.append(frame_list)

                elif self.mode == 'test':
                    gt_file_id.append(float(file[:-4]))
                    self.frame_bbox_list.append([])

        # Extract corresponding event data
        dataset = h5py.File(os.path.join(self.root, self.sequence_list[0], 'processed_data.hdf5'), "r")
        num_frames = dataset["davis/left"]["image_raw"].shape[0]
        num_events = dataset["davis/left"]["events"].shape[0]

        for i_file in gt_file_id:
            closest_event_id = dataset["davis/left"]["image_raw_event_inds"][int(i_file)]
            start_event_id = max(closest_event_id - self.nr_events_window // 2, 0)
            if closest_event_id + self.nr_events_window // 2 >= num_events:
                start_event_id = num_frames - self.nr_events_window
            frame_list = [self.sequence_list[0], int(i_file), start_event_id]
            self.frame_file_list.append(frame_list)

    def extract_unlabeled_data(self):
        for sequence in self.sequence_list:
            dataset = h5py.File(os.path.join(self.root, sequence, 'processed_data.hdf5'), "r")
            num_frames = dataset["davis/left"]["image_raw"].shape[0]
            num_events = dataset["davis/left"]["events"].shape[0]

            for i_frame in range(num_frames):
                closest_event_id = dataset["davis/left"]["image_raw_event_inds"][i_frame]
                start_event_id = max(closest_event_id - self.nr_events_window//2, 0)
                if closest_event_id + self.nr_events_window//2 >= num_events:
                    start_event_id = num_frames - self.nr_events_window

                frame_list = [sequence, i_frame, start_event_id]
                self.frame_file_list.append(frame_list)

    def scale_bounding_boxes(self, bbox_array, original_shape_wh, cropped_height):
        """Adjusts and scales the bounding boxes to the specified height and shape considering the center crop"""
        bbox_array[:, [0, 2]] = bbox_array[:, [0, 2]] - (cropped_height // 2)
        bbox_array[:, [0, 2]] = bbox_array[:, [0, 2]] * float(self.height) / float(original_shape_wh[1])
        bbox_array[:, [1, 3]] = bbox_array[:, [1, 3]] * float(self.width) / float(original_shape_wh[0])

        bbox_array = self.check_bbox_for_boundaries(bbox_array)

        # Convert from x_max, y_max to width and height
        bbox_array[:, [2, 3]] = bbox_array[:, [2, 3]] - bbox_array[:, [0, 1]]

        if self.mode != 'test':
            # Remove Bounding Boxes with diagonal smaller than specifies square pixels
            valid_bbox = (bbox_array[:, 2]**2 + bbox_array[:, 3]**2) > 500
            valid_bbox = valid_bbox * (bbox_array[:, 2] > 5) * (bbox_array[:, 3] > 5)
            bbox_array = bbox_array[valid_bbox, :]

        return bbox_array.astype(np.int32)

    def check_bbox_for_boundaries(self, bbox_array):
        """Check bounding boxes for dimensions"""
        bbox_array[:, :4] = np.maximum(bbox_array[:, :4], 0)
        bbox_array[:, 2] = np.minimum(bbox_array[:, 2], self.height - 1)
        bbox_array[:, 3] = np.minimum(bbox_array[:, 3], self.width - 1)

        return bbox_array

    def __getitem__(self, idx):
        """
        returns event frame and labels ['x', 'y', 'h', 'w', 'class_id']

        :param idx:
        """
        sequence, frame_id, start_event_id = self.frame_file_list[idx]
        dataset = h5py.File(os.path.join(self.root, sequence, 'processed_data.hdf5'), "r")

        events = dataset["davis/left"]["events"][start_event_id:(start_event_id + self.nr_events_window)][()]
        # Convert polarity from [0, 1] to [-1, 1]
        events[:, -1] = 2 * events[:, -1] - 1

        event_tensor = data_util.generate_input_representation(events,
                                                               event_representation=self.event_representation,
                                                               shape=(self.original_height, self.original_width),
                                                               nr_temporal_bins=self.nr_temporal_bins)
        # Mask out the car bonnet
        event_tensor = event_tensor[:, :198, :]

        original_shape_hw = event_tensor.shape[-2:]
        event_tensor = data_util.normalize_event_tensor(event_tensor)

        target_ratio = float(self.height) / float(self.width)
        unscaled_target_height = int(original_shape_hw[1] * target_ratio)
        cropped_height = int(original_shape_hw[0] - unscaled_target_height)

        # cropped_width = 0
        event_tensor = torch.from_numpy(event_tensor[:, (cropped_height//2):-(cropped_height//2), :])
        event_tensor = torch.nn.functional.interpolate(event_tensor[None, :, :, :],
                                                       size=(self.height, self.width),
                                                       mode='bilinear', align_corners=True)[0, :, :, :]

        if self.augmentation:
            if torch.bernoulli(torch.tensor(0.5)):
                event_tensor = torch.flip(event_tensor, dims=[2])

        if self.use_labels:
            if self.mode == 'test' and len(self.frame_bbox_list[idx]) == 0:
                return event_tensor, np.zeros([0, 5], dtype=np.int32)

            # Bounding Box [xmin, ymin, xmax, ymax, class_id]
            bbox_array = np.array(self.frame_bbox_list[idx])
            if self.augmentation:
                if torch.bernoulli(torch.tensor(0.5)):
                    bbox_array[:, 1] = self.width - bbox_array[:, 1] - bbox_array[:, 3] - 1

            bbox_array = self.scale_bounding_boxes(bbox_array, (original_shape_hw[1], unscaled_target_height),
                                                   cropped_height)

            return event_tensor, bbox_array
        else:
            return event_tensor, torch.zeros([0, 5])
