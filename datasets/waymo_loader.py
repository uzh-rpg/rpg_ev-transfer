import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import utils


from torch.utils.data import Dataset


class WaymoRGB(Dataset):
    def __init__(self, root, height=None, width=None, nr_events_window=None, augmentation=False, mode='train',
                 event_representation=None, nr_temporal_bins=5):
        """
        Creates an iterator over the Waymo object recognition dataset.

        :param root: path to dataset root
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: not used in this class
        :param augmentation: not used in this class
        :param mode: 'train', 'test' or 'val'
        :param event_representation: not used in this class
        """
        self.root = root
        self.height = height
        self.width = width
        self.mode = mode
        self.augmentation = augmentation

        self.initial_original_shape_wh = [1920, 1280]
        # ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
        self.cameras_to_use = [dataset_pb2.CameraName.FRONT]
        # ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_CYCLIST']
        self.classes_to_use = [label_pb2.Label.Type.TYPE_VEHICLE]
        self.class_id_to_class = {label_pb2.Label.Type.TYPE_VEHICLE: 0}

        self.class_list = ['Vehicle']

        assert os.path.exists(self.root), 'Specified Root Path not found: {}'.format(self.root)

        self.frame_samples = []
        self.extract_frame_samples()

        if self.mode == 'val':
            import random
            random.seed(7)
            random.shuffle(self.frame_samples)
            self.frame_samples = self.frame_samples[:7000]

    def __len__(self):
        return len(self.frame_samples)

    def extract_frame_samples(self):
        """Goes through specified data path and tf records files to list the keyframes"""
        tf_record_files = [file for file in os.listdir(os.path.join(self.root, self.mode)) if file[:8] == 'segment-']

        # Check if images contain bounding boxes of the specified classes
        for file_name in tqdm(tf_record_files):
            bbox_to_event_idx_name = os.path.join(self.root, 'bbox_frame_idx', self.mode,
                                                  file_name[:-9] + '_frame_idx.npy')
            if os.path.isfile(bbox_to_event_idx_name):
                continue
            current_frame_samples = self.extract_frame_information(file_name)

            np.save(bbox_to_event_idx_name, np.array(current_frame_samples))

        for file_name in tf_record_files:
            bbox_to_img_idx = np.load(os.path.join(self.root, 'bbox_frame_idx', self.mode,
                                                  file_name[:-9] + '_frame_idx.npy'))
            if bbox_to_img_idx.shape[0] == 0:
                continue
            self.frame_samples += [[file_name, idx[0], idx[1]] for idx in bbox_to_img_idx]

    def extract_frame_information(self, file_name):
        """Extract and checks for each frame and camera the needed bounding box"""
        datafile = WaymoDataFileReader(os.path.join(self.root, self.mode, file_name))
        current_frame_samples = []

        for i_frame, frame in enumerate(datafile):
            for camera_id in self.cameras_to_use:
                bbox_to_use = self.read_bboxes(frame, camera_id)

                in_frame = self.checkBBoxCroppedOut(bbox_to_use)
                if in_frame:
                    current_frame_samples.append([i_frame, camera_id])

        return current_frame_samples

    def checkBBoxCroppedOut(self, bbox_list):
        """Checks if the bounding box is still in the frame after rescaling and cropping"""
        if len(bbox_list) == 0:
            return False

        target_ratio = float(self.height) / float(self.width)
        unscaled_target_height = int(self.initial_original_shape_wh[0] * target_ratio)
        cropped_height = int(self.initial_original_shape_wh[1] - unscaled_target_height)

        # Bounding Box
        bbox_array = np.array(bbox_list)
        # Convert from [umin, vmin, umax, vmax, class_id] to [xmin, ymin, xmax, ymax, class_id]
        bbox_array = bbox_array[:, [1, 0, 3, 2, 4]]
        bbox_array = self.scale_bounding_boxes(bbox_array, (self.initial_original_shape_wh[0], unscaled_target_height),
                                               cropped_height)

        return bbox_array.shape[0] > 0

    def scale_bounding_boxes(self, bbox_array, original_shape_wh, cropped_dist, width_crop=False):
        """Adjusts and scales the bounding boxes to the specified height and shape considering the center crop"""
        if width_crop:
            bbox_array[:, [1, 3]] = bbox_array[:, [1, 3]] - (cropped_dist // 2)
        else:
            bbox_array[:, [0, 2]] = bbox_array[:, [0, 2]] - (cropped_dist // 2)

        bbox_array[:, [0, 2]] = bbox_array[:, [0, 2]] * float(self.height) / float(original_shape_wh[1])
        bbox_array[:, [1, 3]] = bbox_array[:, [1, 3]] * float(self.width) / float(original_shape_wh[0])

        bbox_array = self.check_bbox_for_boundaries(bbox_array)

        # Convert from x_max, y_max to width and height
        bbox_array[:, [2, 3]] = bbox_array[:, [2, 3]] - bbox_array[:, [0, 1]]

        # Remove Bounding Boxes with diagonal smaller than specifies square pixels
        # If the diagonal length is changed, the preprocessing needs to be redone
        valid_bbox = (bbox_array[:, 2]**2 + bbox_array[:, 3]**2) > 500
        # valid_bbox = (bbox_array[:, 2] ** 2 + bbox_array[:, 3] ** 2) > 100
        valid_bbox = valid_bbox * (bbox_array[:, 2] > 5) * (bbox_array[:, 3] > 5)
        bbox_array = bbox_array[valid_bbox, :]

        return bbox_array.astype(np.int32)

    def check_bbox_for_boundaries(self, bbox_array):
        """Check bounding boxes for dimensions"""
        bbox_array[:, :4] = np.maximum(bbox_array[:, :4], 0)
        bbox_array[:, 2] = np.minimum(bbox_array[:, 2], self.height-1)
        bbox_array[:, 3] = np.minimum(bbox_array[:, 3], self.width-1)

        return bbox_array

    def getImgTransforms(self, crop_size):
        img_transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                            transforms.Resize((self.height, self.width)),
                                            transforms.ToTensor()])
        return img_transform

    def read_bboxes(self, frame, camera_id):
        current_camera_bbox = utils.get(frame.camera_labels, camera_id).labels
        bbox_to_use = []
        for bbox in current_camera_bbox:
            # IPython.embed()
            if bbox.type in self.classes_to_use:
                coords = bbox.box
                bbox_to_use.append([coords.center_x - coords.length / 2,
                                    coords.center_y - coords.width / 2,
                                    coords.center_x + coords.length / 2,
                                    coords.center_y + coords.width / 2,
                                    self.class_id_to_class[bbox.type]])
        return bbox_to_use

    def getSensorMeasurements(self, path, frame_id, camera_id):
        datafile = WaymoDataFileReader(path)
        table = datafile.get_record_table()
        datafile.seek(table[frame_id])
        frame = datafile.read_record()

        camera = utils.get(frame.images, camera_id)
        sensor_measurement = utils.decode_image(camera)

        target_ratio = float(self.height) / float(self.width)
        unscaled_target_height = int(self.initial_original_shape_wh[0] * target_ratio)
        cropped_height = int(self.initial_original_shape_wh[1] - unscaled_target_height)

        img_transform = self.getImgTransforms((unscaled_target_height, self.initial_original_shape_wh[0]))
        sensor_measurement = img_transform(Image.fromarray(sensor_measurement))

        bbox_array = np.array(self.read_bboxes(frame, camera_id))

        # Convert from [umin, vmin, umax, vmax, class_id] to [xmin, ymin, xmax, ymax, class_id]
        bbox_array = bbox_array[:, [1, 0, 3, 2, 4]]
        bbox_array = self.scale_bounding_boxes(bbox_array, (self.initial_original_shape_wh[0], unscaled_target_height),
                                               cropped_height)

        return sensor_measurement, bbox_array

    def __getitem__(self, idx):
        """
        returns frame and labels ['x', 'y', 'h', 'w', 'class_id']

        :param idx:
        """
        sensor_measurement_path = self.frame_samples[idx][0]
        frame_id = self.frame_samples[idx][1]
        camera_id = self.frame_samples[idx][2]

        sensor_measurement_path = os.path.join(self.root, self.mode, sensor_measurement_path)
        # Bounding Box
        sensor_measurement, bbox_array = self.getSensorMeasurements(sensor_measurement_path, frame_id, camera_id)

        if self.augmentation:
            if torch.bernoulli(torch.tensor(0.5)):
                sensor_measurement = torch.flip(sensor_measurement, dims=[2])
                bbox_array[:, 1] = self.width - bbox_array[:, 1] - bbox_array[:, 3] - 1

        return sensor_measurement, bbox_array


class WaymoGray(WaymoRGB):
    def getImgTransforms(self, crop_size):
        img_transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                            transforms.Resize((self.height, self.width)),
                                            transforms.Grayscale(),
                                            transforms.ToTensor()])
        return img_transform
