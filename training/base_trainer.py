from __future__ import division

import torch
from tqdm import tqdm
tqdm.monitor_interval = 0
import numpy as np
from tensorboardX import SummaryWriter

from utils.saver import CheckpointSaver
from datasets.object_det_loader import ObjectDetLoader
from datasets.wrapper_dataloader import WrapperDataset
import utils.viz_utils as viz_utils


class BaseTrainer(object):
    """BaseTrainer class to be inherited"""
    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.do_val_training_epoch = True

        # override this function to define your model, optimizers etc.
        self.init_fn()
        self.createDataLoaders()

        self.models_dict = {k: v.to(self.device) for k, v in self.models_dict.items()}

        # tensorboardX SummaryWriter for use in train_summaries
        self.summary_writer = SummaryWriter(self.settings.ckpt_dir)

        # Load the latest checkpoints
        load_optimizer = False
        if self.settings.resume_training:
            # load_optimizer = True
            load_optimizer = False

            self.saver = CheckpointSaver(save_dir=settings.ckpt_dir)
            self.checkpoint = self.saver.load_checkpoint(self.models_dict,
                                                         self.optimizers_dict,
                                                         checkpoint_file=self.settings.resume_ckpt_file,
                                                         load_optimizer=load_optimizer)
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['step_count']

        else:
            self.saver = CheckpointSaver(save_dir=settings.ckpt_dir)
            self.epoch_count = 0
            self.step_count = 0
            self.checkpoint = None

        self.epoch = self.epoch_count
        self.val_confusion_matrix = np.zeros([len(self.object_classes), len(self.object_classes)])

        optimizer_epoch_count = self.epoch_count if load_optimizer else 0
        self.lr_schedulers = {k: torch.optim.lr_scheduler.ExponentialLR(v, gamma=self.settings.lr_decay,
                                                                        last_epoch=optimizer_epoch_count-1)
                              for k, v in self.optimizers_dict.items()}


    def init_fn(self):
        """Model is constructed in child class"""
        pass

    def getDataloader(self, dataset_name):
        """Returns the dataset loader specified in the settings file"""
        if dataset_name == 'Caltech101_gray':
            from datasets.caltech101_loader import Caltech101Gray
            return Caltech101Gray
        elif dataset_name == 'Waymo_gray':
            from datasets.waymo_loader import WaymoGray
            return WaymoGray
        elif dataset_name == 'NCaltech101_events':
            from datasets.ncaltech101_loader import NCaltech101Events
            return NCaltech101Events
        elif dataset_name == 'OneMpProphesee_events':
            from datasets.oneMP_prophesee_loader import OneMPProphesee
            return OneMPProphesee
        elif dataset_name == 'MVSEC_events':
            from datasets.mvsec_loader import MVSEC_Events
            return MVSEC_Events

    def createDataset(self, dataset_name, dataset_path, img_size, batch_size, nr_events_window, event_representation,
                      nr_temporal_bins):
        """
        Creates the validation and the training data based on the provided paths and parameters.
        """
        dataset_builder = self.getDataloader(dataset_name)

        train_dataset = dataset_builder(dataset_path,
                                        height=img_size[0],
                                        width=img_size[1],
                                        nr_events_window=nr_events_window,
                                        augmentation=True,
                                        mode='train',
                                        event_representation=event_representation,
                                        nr_temporal_bins=nr_temporal_bins)

        val_dataset = dataset_builder(dataset_path,
                                      height=img_size[0],
                                      width=img_size[1],
                                      nr_events_window=nr_events_window,
                                      augmentation=False,
                                      mode='val',
                                      event_representation=event_representation,
                                      nr_temporal_bins=nr_temporal_bins)

        self.object_classes = train_dataset.class_list

        if dataset_name in ['Waymo_gray', 'MVSEC_events', 'OneMpProphesee_events']:
            dataset_loader = ObjectDetLoader
            train_loader_sensor = dataset_loader(train_dataset, batch_size=batch_size,
                                                 num_workers=self.settings.num_cpu_workers,
                                                 pin_memory=False, shuffle=True, drop_last=True)
            val_loader_sensor = dataset_loader(val_dataset, batch_size=batch_size,
                                               num_workers=self.settings.num_cpu_workers,
                                               pin_memory=False, shuffle=False, drop_last=True)

        else:
            dataset_loader = torch.utils.data.DataLoader
            train_loader_sensor = dataset_loader(train_dataset, batch_size=batch_size,
                                                 num_workers=self.settings.num_cpu_workers,
                                                 pin_memory=False, shuffle=True, drop_last=True)
            val_loader_sensor = dataset_loader(val_dataset, batch_size=batch_size,
                                               num_workers=self.settings.num_cpu_workers,
                                               pin_memory=False, shuffle=False, drop_last=True)

        return train_loader_sensor, val_loader_sensor

    def combineDataloaders(self):
        """Combines two dataloader to one dataloader."""
        self.train_loader = WrapperDataset(self.train_loader_sensor_a, self.train_loader_sensor_b, self.device)

    def createDataLoaders(self):
        out = self.createDataset(self.settings.dataset_name_a,
                                 self.settings.dataset_path_a,
                                 self.settings.img_size_a,
                                 self.settings.batch_size_a,
                                 self.settings.nr_events_window_a,
                                 self.settings.event_representation_a,
                                 self.settings.input_channels_a // 2)
        self.train_loader_sensor_a, self.val_loader_sensor_a = out

        out = self.createDataset(self.settings.dataset_name_b,
                                 self.settings.dataset_path_b,
                                 self.settings.img_size_b,
                                 self.settings.batch_size_b,
                                 self.settings.nr_events_window_b,
                                 self.settings.event_representation_b,
                                 self.settings.input_channels_b // 2)
        self.train_loader_sensor_b, self.val_loader_sensor_b = out

        self.combineDataloaders()

    def train(self):
        """Main training and validation loop"""
        val_epoch_step = 2
        if self.settings.dataset_name_b in ['MVSEC_events', 'OneMpProphesee_events']:
            val_epoch_step = 1

        for _ in tqdm(range(self.epoch_count, self.settings.num_epochs), total=self.settings.num_epochs,
                      initial=self.epoch_count):

            if (self.epoch_count % val_epoch_step) == 0:
                self.validationEpochs()

            self.trainEpoch()

            if self.epoch_count % val_epoch_step == 0:
                self.saver.save_checkpoint(self.models_dict,
                                           self.optimizers_dict, self.epoch_count, self.step_count,
                                           self.settings.batch_size_a,
                                           self.settings.batch_size_b)
                tqdm.write('Checkpoint saved')

            # apply the learning rate scheduling policy
            for opt in self.optimizers_dict:
                self.lr_schedulers[opt].step()
            self.epoch_count += 1

        self.validationEpochs()

    def trainEpoch(self):
        self.pbar = tqdm(total=self.train_loader.__len__(), unit='Batch', unit_scale=True)
        self.train_loader.createIterators()
        for model in self.models_dict:
            self.models_dict[model].train()

        for i_batch, sample_batched in enumerate(self.train_loader):
            out = self.train_step(sample_batched)

            self.train_summaries(out[0])

            self.step_count += 1
            self.pbar.set_postfix(TrainLoss='{:.2f}'.format(out[-1].data.cpu().numpy()))
            self.pbar.update(1)

        self.pbar.close()

    def validationEpochs(self):
        self.resetValidationStatistics()

        with torch.no_grad():
            for model in self.models_dict:
                self.models_dict[model].eval()

            self.validationEpoch(self.val_loader_sensor_a, 'sensor_a')
            self.validationEpoch(self.val_loader_sensor_b, 'sensor_b')

            if self.do_val_training_epoch:
                self.trainDatasetStatisticsEpoch('sensor_a', self.train_loader_sensor_a)
                self.trainDatasetStatisticsEpoch('sensor_b', self.train_loader_sensor_b)

            self.resetValidationStatistics()

        self.pbar.close()

    def validationEpoch(self, data_loader, sensor_name):
        val_dataset_length = data_loader.__len__()
        self.pbar = tqdm(total=val_dataset_length, unit='Batch', unit_scale=True)
        tqdm.write("Validation on " + sensor_name)
        cumulative_losses = {}
        total_nr_steps = None

        for i_batch, sample_batched in enumerate(data_loader):
            self.validationBatchStep(sample_batched, sensor_name, i_batch, cumulative_losses, val_dataset_length)
            self.pbar.update(1)
            total_nr_steps = i_batch
        self.val_summaries(cumulative_losses, total_nr_steps + 1)
        self.pbar.close()
        if self.val_confusion_matrix.sum() != 0:
            self.addValidationMatrix(sensor_name)

        self.saveValStatistics('val', sensor_name)

    def validationBatchStep(self, sample_batched, sensor, i_batch, cumulative_losses, val_dataset_length):
        nr_reconstr_vis = 3
        vis_step_size = max(val_dataset_length // nr_reconstr_vis, 1)
        vis_reconstr_idx = i_batch // vis_step_size if (i_batch % vis_step_size) == vis_step_size - 1 else -1

        if type(sample_batched[0]) is list:
            sample_batched = [[tensor.to(self.device) for tensor in sensor_batch] for sensor_batch in sample_batched]
        else:
            sample_batched = [tensor.to(self.device) for tensor in sample_batched]

        out = self.val_step(sample_batched, sensor, i_batch, vis_reconstr_idx)

        for k, v in out[0].items():
            if k in cumulative_losses:
                cumulative_losses[k] += v
            else:
                cumulative_losses[k] = v

    def trainDatasetStatisticsEpoch(self, sensor, data_loader):
        cumulative_losses = {}
        total_nr_steps = 0

        self.pbar = tqdm(total=data_loader.__len__(), unit='Batch', unit_scale=True)
        for i_batch, sample_batched in enumerate(data_loader):
            sample_batched = [tensor.to(self.device) for tensor in sample_batched]
            self.val_train_stats_step(sample_batched, sensor, i_batch, cumulative_losses)
            self.pbar.update(1)
            total_nr_steps = i_batch

        self.pbar.close()
        self.val_summaries(cumulative_losses, total_nr_steps + 1)
        self.saveValStatistics('val_training', sensor)

    def visualize_epoch(self):
        if self.settings.dataset_name_b == 'OneMpProphesee_events':
            viz_ratio = 0.1
        elif self.settings.dataset_name_b == 'MVSEC_events':
            viz_ratio = 0.5
        else:
            viz_ratio = 3

        return ((self.step_count - self.settings.disc_iter) / (self.settings.disc_iter + 1)) % \
                int((viz_ratio * int(self.train_loader.__len__() / (self.settings.disc_iter + 1)))) == 0

    def addValidationMatrix(self, sensor):
        self.val_confusion_matrix = self.val_confusion_matrix / (np.sum(self.val_confusion_matrix, axis=-1,
                                                                        keepdims=True) + 1e-9)
        plot_confusion_matrix = viz_utils.visualizeConfusionMatrix(self.val_confusion_matrix)
        tag = 'val/Confusion_Matrix_' + sensor
        tag.replace('sensor_a', self.settings.sensor_a_name).replace('sensor_b', self.settings.sensor_b_name)
        self.summary_writer.add_image(tag, plot_confusion_matrix, self.epoch_count, dataformats='HWC')

        self.val_confusion_matrix = np.zeros([len(self.object_classes), len(self.object_classes)])

    def summaries(self, losses, mode="train"):
        self.summary_writer.add_scalar("{}/learning_rate".format(mode),
                                       self.get_lr(),
                                       self.step_count)
        for k, v in losses.items():
            tag = k.replace('sensor_a', self.settings.sensor_a_name).replace('sensor_b', self.settings.sensor_b_name)
            self.summary_writer.add_scalar("{}/{}".format(mode, tag), v, self.step_count)

    def train_summaries(self, losses):
        nr_steps_avg = 50

        # Update sums
        for key, value in losses.items():
            # if key in keys_to_average:
            if key in self.train_statistics:
                self.train_statistics[key][0] += value
                self.train_statistics[key][1] += 1
            else:
                self.train_statistics[key] = [value, 1]

        if self.step_count % nr_steps_avg == (nr_steps_avg - 1):
            for key, _ in self.train_statistics.items():
                losses[key] = (self.train_statistics[key][0]) / self.train_statistics[key][1]
            self.train_statistics = {}
            self.summaries(losses, mode="train")

    def val_summaries(self, statistics, total_nr_steps):
        for k, v in statistics.items():
            tag = k.replace('sensor_a', self.settings.sensor_a_name).replace('sensor_b', self.settings.sensor_b_name)
            self.summary_writer.add_scalar("val/{}".format(tag), v / total_nr_steps, self.epoch_count)

    def img_summaries(self, tag, img, step=None):
        tag = tag.replace('sensor_a', self.settings.sensor_a_name).replace('sensor_b', self.settings.sensor_b_name)
        self.summary_writer.add_image(tag, img, step)
        self.summary_writer.flush()

    def get_lr(self):
        return next(iter(self.lr_schedulers.values())).get_last_lr()[0]

    def resetValidationStatistics(self):
        """If wanted, needs to be implement in a child class"""
        pass

    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def val_step(self, input_batch, sensor, i_batch, vis_reconstr_idx):
        raise NotImplementedError('You need to provide a _train_step method')

    def val_train_stats_step(self, input_batch, sensor, i_batch, cumulative_losses):
        raise NotImplementedError('You need to provide a val_train_acc_step method')

    def saveValStatistics(self, mode, sensor):
        """If wanted, needs to be implement in a child class"""
        pass

    def test(self):
        pass