import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import datasets.data_util as data_util


class Caltech101RGB(Dataset):
    def __init__(self, root, height=None, width=None, nr_events_window=None, augmentation=False, mode='train',
                 event_representation='histogram', nr_temporal_bins=5):
        """
        Creates an iterator over the N_Caltech101 dataset.

        :param root: path to dataset root
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events summed in the sliding histogram
        :param augmentation: flip, shift and random window start for training
        :param mode: 'train', 'test' or 'val'
        """
        # self.extended_data is set to False for event dataloader in getRootPath function
        self.extended_data = mode != 'test'
        self.augmentation = augmentation
        self.class_list = []
        self.height = height
        self.width = width
        self.nr_events_window = nr_events_window
        self.event_representation = event_representation
        self.nr_temporal_bins = nr_temporal_bins
        self.root = self.getRootPath(root)

        if self.extended_data:
            dir_name = 'split_train_0.5_val_0.25_test_0.25_new_extended'
        else:
            dir_name = 'split_train_0.5_val_0.25_test_0.25_new'

        with open(os.path.join(root, dir_name, mode + '.txt')) as f:
            self.data_split = f.read().split('\n')[:-1]

        self.class_list = os.listdir(self.root)
        self.class_list.sort()
        self.files = []
        self.labels = []

        self.getModeSplit()

        zipped_lists = list(zip(self.files, self.labels))
        random.seed(7)
        random.shuffle(zipped_lists)
        self.files, self.labels = zip(*zipped_lists)

    def __len__(self):
        return len(self.files)

    def getRootPath(self, root):
        """Function makes it easier to handle child of this class e.g. N-Caltech101 dataloader"""
        return os.path.join(root, 'Caltech101')

    def getModeSplit(self):
        for i, object_class in enumerate(self.class_list):
            idx_list = []
            for idx in self.data_split[(self.data_split.index(object_class + ':') + 1):]:
                if idx[0] != '-':
                    break
                idx = idx.split(' ')[-1]
                if idx[:5] == 'label':
                    idx_list.append(float(idx.split('_')[1]))
                else:
                    idx_list.append(float(idx))

            data_list = os.listdir(os.path.join(self.root, object_class))
            data_list.sort()

            data_check_idx = []
            for image_name in data_list:
                if image_name[:5] == 'image':
                    data_check_idx.append(float(image_name.split('_')[1][:-4]))
                elif image_name.split('_')[1] == 'events':
                    data_check_idx.append(float(image_name.split('_')[-1][:4]))
                else:
                    data_check_idx.append(float(image_name.split('_')[1]))

            new_files = [os.path.join(object_class, f) for i, f in enumerate(data_list)
                         if (data_check_idx[i] in idx_list)]

            self.files += new_files
            self.labels += [i] * len(new_files)

    def __getitem__(self, idx):
        """
        Returns one data sample

        :param idx:
        :return: image or event,  label or bounding box
        """
        label = self.labels[idx]
        sensor_measurement = Image.open(os.path.join(self.root, self.files[idx])).convert('RGB')

        img_width, img_height = sensor_measurement.size
        min_scale = min(self.height / img_height, self.width / img_width, 1)
        scaled_height, scaled_width = int(img_height * min_scale), int(img_width * min_scale)

        if self.augmentation:
            img_transform = transforms.Compose([
                                                transforms.Resize(size=[scaled_height, scaled_width]),
                                                transforms.Pad(padding=(0, 0, self.width - scaled_width,
                                                                        self.height - scaled_height)),
                                                transforms.RandomAffine(degrees=0, translate=[0.1, 0.1]),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor()
                                                ])

        else:
            img_transform = transforms.Compose([transforms.Resize(size=[scaled_height, scaled_width]),
                                                transforms.Pad(padding=(0, 0, self.width - scaled_width,
                                                                        self.height - scaled_height)),
                                                transforms.ToTensor()])

        sensor_measurement = img_transform(sensor_measurement)

        if self.augmentation:
            mid_point_u = sensor_measurement[0, :, :].sum(axis=1).nonzero(as_tuple=False).squeeze(dim=0).squeeze(dim=-1)
            mid_point_v = sensor_measurement[0, :, :].sum(axis=0).nonzero(as_tuple=False).squeeze(dim=0).squeeze(dim=-1)
            mid_point = [mid_point_u[mid_point_u.shape[0] // 2], mid_point_v[mid_point_v.shape[0] // 2]]
            sensor_measurement = data_util.random_crop_resize(sensor_measurement, mid_point)

        return sensor_measurement, label


class Caltech101Gray(Caltech101RGB):
    def __getitem__(self, idx):
        """
        Returns one data sample

        :param idx:
        :return: image or event,  label or bounding box
        """
        label = self.labels[idx]
        sensor_measurement = Image.open(os.path.join(self.root, self.files[idx])).convert('RGB')

        img_width, img_height = sensor_measurement.size
        min_scale = min(self.height / img_height, self.width / img_width, 1)
        scaled_height, scaled_width = int(img_height * min_scale), int(img_width * min_scale)

        if self.augmentation:
            img_transform = transforms.Compose([
                transforms.Resize(size=[scaled_height, scaled_width]),
                transforms.Pad(padding=(0, 0, self.width - scaled_width,
                                        self.height - scaled_height)),
                transforms.RandomAffine(degrees=0, translate=[0.1, 0.1]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Grayscale(),
                transforms.ToTensor()
            ])

        else:
            img_transform = transforms.Compose([transforms.Resize(size=[scaled_height, scaled_width]),
                                                transforms.Pad(padding=(0, 0, self.width - scaled_width,
                                                                        self.height - scaled_height)),
                                                transforms.Grayscale(),
                                                transforms.ToTensor()])

        sensor_measurement = img_transform(sensor_measurement)

        if self.augmentation:
            mid_point_u = sensor_measurement[0, :, :].sum(axis=1).nonzero(as_tuple=False).squeeze(dim=0).squeeze(dim=-1)
            mid_point_v = sensor_measurement[0, :, :].sum(axis=0).nonzero(as_tuple=False).squeeze(dim=0).squeeze(dim=-1)
            mid_point = [mid_point_u[mid_point_u.shape[0] // 2], mid_point_v[mid_point_v.shape[0] // 2]]
            sensor_measurement = data_util.random_crop_resize(sensor_measurement, mid_point)

        return sensor_measurement, label
