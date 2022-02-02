import os
import time
import yaml
import torch
import shutil


class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- hardware ---
            hardware = settings['hardware']
            gpu_device = hardware['gpu_device']

            self.gpu_device = torch.device("cpu") if gpu_device == "cpu" else torch.device("cuda:" + str(gpu_device))

            self.num_cpu_workers = hardware['num_cpu_workers']
            if self.num_cpu_workers < 0:
                self.num_cpu_workers = os.cpu_count()

            # --- Model ---
            model = settings['model']
            self.model_name = model['model_name']
            self.use_decoder_a = model['use_decoder_a']
            self.use_decoder_b = model['use_decoder_b']
            self.use_task_a = model['use_task_a']
            self.use_task_b = model['use_task_b']
            self.use_cycle_a_b = model['use_cycle_a_b']
            self.use_cycle_b_a = model['use_cycle_b_a']
            self.use_cycle_task = model['use_cycle_task']
            self.cross_refinement_a = model['cross_refinement_a']
            self.cross_refinement_b = model['cross_refinement_b']

            if not (self.use_decoder_a or self.use_decoder_b) and (self.use_cycle_a_b or self.use_cycle_b_a):
                raise ValueError("Cycle loss requires decoder output")

            # --- dataset sensor a ---
            dataset = settings['dataset']
            self.dataset_name_a = dataset['name_a']
            self.sensor_a_name = self.dataset_name_a.split('_')[-1]
            self.nr_events_window_a = None
            self.event_representation_a = None
            if self.dataset_name_a == 'Caltech101_gray':
                dataset_specs = dataset['caltech101']
                self.input_channels_a = 1
            elif self.dataset_name_a == 'Waymo_gray':
                dataset_specs = dataset['waymo']
                self.input_channels_a = 1
            elif self.dataset_name_a in ['NCaltech101_events', 'MVSEC_events', 'OneMpProphesee_events']:
                dataset_specs = dataset[self.dataset_name_a.split('_')[0].lower()]
                self.nr_events_window_a = dataset_specs['nr_events_window']
                self.event_representation_a = dataset_specs['event_representation']
                if self.event_representation_a == 'voxel_grid':
                    self.input_channels_a = dataset_specs['nr_temporal_bins'] * 2
                else:
                    self.input_channels_a = 2
            else:
                raise ValueError("Specified Dataset Sensor A: %s is not implemented" % self.dataset_name_a)

            self.img_size_a = dataset_specs['shape']
            self.dataset_path_a = dataset_specs['dataset_path']
            assert os.path.isdir(self.dataset_path_a)

            # --- dataset sensor b ---
            dataset = settings['dataset']
            self.dataset_name_b = dataset['name_b']
            self.sensor_b_name = self.dataset_name_b.split('_')[-1]
            self.nr_events_window_b = None
            self.event_representation_b = None
            if self.dataset_name_b == 'Caltech101_gray':
                dataset_specs = dataset['caltech101']
                self.input_channels_b = 1
            elif self.dataset_name_b == 'Waymo_gray':
                dataset_specs = dataset['waymo']
                self.input_channels_b = 1
            elif self.dataset_name_b in ['NCaltech101_events', 'MVSEC_events', 'OneMpProphesee_events']:
                dataset_specs = dataset[self.dataset_name_b.split('_')[0].lower()]
                self.nr_events_window_b = dataset_specs['nr_events_window']
                self.event_representation_b = dataset_specs['event_representation']
                if self.event_representation_b == 'voxel_grid':
                    self.input_channels_b = dataset_specs['nr_temporal_bins'] * 2
                else:
                    self.input_channels_b = 2
            else:
                raise ValueError("Specified Dataset Sensor B: %s is not implemented" % self.dataset_name_b)

            self.img_size_b = dataset_specs['shape']
            self.dataset_path_b = dataset_specs['dataset_path']

            assert os.path.isdir(self.dataset_path_b)

            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']

            # --- directories ---
            directories = settings['dir']
            log_dir = directories['log']

            # --- logs ---
            if generate_log:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                log_dir = os.path.join(log_dir, timestr)
                os.makedirs(log_dir)
                settings_copy_filepath = os.path.join(log_dir, os.path.split(settings_yaml)[-1])
                shutil.copyfile(settings_yaml, settings_copy_filepath)
                self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
                os.mkdir(self.ckpt_dir)
                self.vis_dir = os.path.join(log_dir, 'visualization')
                os.mkdir(self.vis_dir)
            else:
                self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
                self.vis_dir = os.path.join(log_dir, 'visualization')

            # --- optimization ---
            optimization = settings['optim']
            self.batch_size_a = int(optimization['batch_size_a'])
            self.batch_size_b = int(optimization['batch_size_b'])
            self.lr_discriminator = float(optimization['lr_discriminator'])
            self.lr_front = float(optimization['lr_front'])
            self.lr_back = float(optimization['lr_back'])
            self.lr_decoder = float(optimization['lr_decoder'])
            self.lr_decay = float(optimization['lr_decay'])
            self.disc_iter = int(optimization['disc_iter'])
            self.front_iter = int(optimization['front_iter'])
            self.num_epochs = int(optimization['num_epochs'])
            self.weight_generator_loss = float(optimization['weight_generator_loss'])
            self.weight_discriminator_loss = float(optimization['weight_discriminator_loss'])
            self.weight_task_loss = float(optimization['weight_task_loss'])
            self.weight_KL_loss = float(optimization['weight_KL_loss'])
            self.weight_reconstruction_sensor_a_loss = float(optimization['weight_reconstruction_sensor_a_loss'])
            self.weight_reconstruction_sensor_b_loss = float(optimization['weight_reconstruction_sensor_b_loss'])
            self.weight_refinement_loss = float(optimization['weight_refinement_loss'])
            self.weight_cycle_loss = float(optimization['weight_cycle_loss'])
            self.weight_smoothness_loss = float(optimization['weight_smoothness_loss'])
