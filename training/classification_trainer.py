import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.models as models

from utils import pytorch_ssim, radam
import utils.viz_utils as viz_utils
from datasets.wrapper_dataloader import WrapperDataset
from utils.loss_functions import generator_loss_two_sensors, discriminator_loss, generator_loss, \
    event_reconstruction_loss
from models import refinement_net
from models.style_networks import StyleEncoder, StyleDecoder, ContentDiscriminator, CrossDiscriminator
import training.base_trainer
from utils.sobel_filter import NormGradient, GaussianSmoothing


class ClassificationModel(training.base_trainer.BaseTrainer):
    def __init__(self, settings, train=True):
        self.is_training = train
        super(ClassificationModel, self).__init__(settings)

        self.norm_gradient_layer = NormGradient(self.device, ignore_border=True)
        self.gaussian_layer = GaussianSmoothing(channels=1, kernel_size=[3, 3], sigma=[1, 1], device=self.device)

    def init_fn(self):
        self.buildModels()
        self.createOptimizerDict()

        # Decoder Loss
        self.ssim = pytorch_ssim.SSIM()
        secondary_l1 = nn.L1Loss(reduction="mean")
        image_loss = lambda x, y: secondary_l1(x, y) - self.ssim(x, y)
        reconst_loss_dict = {'gray': image_loss, 'events': event_reconstruction_loss}

        self.reconst_loss_dict = {'sensor_a': reconst_loss_dict.get(self.settings.sensor_a_name),
                                  'sensor_b': reconst_loss_dict.get(self.settings.sensor_b_name)}

        self.cycle_content_loss = torch.nn.L1Loss()

        self.cycle_attribute_loss = torch.nn.L1Loss()

        # Task Loss
        self.task_loss = nn.CrossEntropyLoss()
        self.train_statistics = {}

    def buildModels(self):
        attribute_channels = 8

        # Shared Encoder Layers
        self.front_end_shared = list(models.resnet18(pretrained=True).children())[5][1]

        # Front End Sensor A
        self.front_end_sensor_a = StyleEncoder(self.settings.input_channels_a, self.front_end_shared,
                                               attribute_channels, self.settings.use_decoder_a)

        # Front End Sensor B
        self.front_end_sensor_b = StyleEncoder(self.settings.input_channels_b, self.front_end_shared,
                                               attribute_channels, self.settings.use_decoder_b)

        # Discriminator
        self.discriminator = ContentDiscriminator(nr_channels=128)

        self.models_dict = {"front_sensor_a": self.front_end_sensor_a,
                            "front_sensor_b": self.front_end_sensor_b,
                            "front_shared": self.front_end_shared,
                            "dis": self.discriminator}

        # Task Backend
        if self.settings.use_task_a or self.settings.use_task_b:
            self.task_backend = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[6:-1] +
                                                [nn.Flatten(), nn.Linear(512, 101)]))
            self.models_dict["back_end"] = self.task_backend

        # Decoders
        if self.settings.use_decoder_a:
            self.decoder_sensor_a = StyleDecoder(input_c=128, output_c=self.settings.input_channels_a,
                                                 attribute_channels=attribute_channels,
                                                 sensor_name=self.settings.sensor_a_name)
            self.models_dict["decoder_sensor_a"] = self.decoder_sensor_a

        if self.settings.use_decoder_b:
            self.decoder_sensor_b = StyleDecoder(input_c=128, output_c=2,
                                                 attribute_channels=attribute_channels,
                                                 sensor_name=self.settings.sensor_b_name)
            self.models_dict["decoder_sensor_b"] = self.decoder_sensor_b

        # Cross Refinement Networks
        if self.settings.cross_refinement_a:
            self.cross_refinement_net_a = refinement_net.StyleRefinementNetwork(input_c=2,
                                                                                output_c=self.settings.input_channels_a,
                                                                                sensor=self.settings.sensor_a_name,
                                                                                channel_list=[16, 8],
                                                                                last_layer_pad=1,
                                                                                device=self.device)
            self.refinement_discr_a = CrossDiscriminator(input_dim=self.settings.input_channels_a, n_layer=6)

            self.models_dict["cross_refinement_net_sensor_a"] = self.cross_refinement_net_a
            self.models_dict["refinement_discr_sensor_a"] = self.refinement_discr_a

        if self.settings.cross_refinement_b:
            self.cross_refinement_net_b = refinement_net.StyleRefinementNetwork(input_c=2,
                                                                                output_c=self.settings.input_channels_b,
                                                                                sensor=self.settings.sensor_b_name,
                                                                                channel_list=[16, 8],
                                                                                last_layer_pad=1,
                                                                                device=self.device)
            self.refinement_discr_b = CrossDiscriminator(input_dim=self.settings.input_channels_b, n_layer=6)

            self.models_dict["cross_refinement_net_sensor_b"] = self.cross_refinement_net_b
            self.models_dict["refinement_discr_sensor_b"] = self.refinement_discr_b

    def createOptimizerDict(self):
        """Creates the dictionary containing the optimizer for the the specified subnetworks"""
        if not self.is_training:
            self.optimizers_dict = {}
            return

        dis_params = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        front_sensor_a_params = filter(lambda p: p.requires_grad, self.front_end_sensor_a.parameters())
        front_sensor_b_params = filter(lambda p: p.requires_grad, self.front_end_sensor_b.parameters())
        front_shared_params = filter(lambda p: p.requires_grad, self.front_end_shared.parameters())

        weight_decay = 0.01
        optimizer_dis = radam.RAdam(dis_params,
                                    lr=self.settings.lr_discriminator,
                                    weight_decay=weight_decay,
                                    betas=(0., 0.999))
        optimizer_front_sensor_a = radam.RAdam(front_sensor_a_params,
                                               lr=self.settings.lr_front,
                                               weight_decay=weight_decay,
                                               betas=(0., 0.999))
        optimizer_front_sensor_b = radam.RAdam(front_sensor_b_params,
                                               lr=self.settings.lr_front,
                                               weight_decay=weight_decay,
                                               betas=(0., 0.999))
        optimizer_front_shared = radam.RAdam(front_shared_params,
                                             lr=self.settings.lr_front,
                                             weight_decay=weight_decay,
                                             betas=(0., 0.999))

        self.optimizers_dict = {"optimizer_front_sensor_a": optimizer_front_sensor_a,
                                "optimizer_front_sensor_b": optimizer_front_sensor_b,
                                "optimizer_front_shared": optimizer_front_shared,
                                "optimizer_dis": optimizer_dis}

        # Task
        if self.settings.use_task_a or self.settings.use_task_b:
            back_params = filter(lambda p: p.requires_grad, self.task_backend.parameters())
            optimizer_back = radam.RAdam(back_params,
                                         lr=self.settings.lr_back,
                                         weight_decay=weight_decay,
                                         betas=(0., 0.999))
            self.optimizers_dict["optimizer_back"] = optimizer_back

        # Decoder Task
        if self.settings.use_decoder_a:
            decoder_sensor_a_params = filter(lambda p: p.requires_grad, self.decoder_sensor_a.parameters())
            optimizer_decoder_sensor_a = radam.RAdam(decoder_sensor_a_params,
                                                     lr=self.settings.lr_decoder,
                                                     weight_decay=weight_decay,
                                                     betas=(0., 0.999))

            self.optimizers_dict["optimizer_decoder_sensor_a"] = optimizer_decoder_sensor_a

        if self.settings.use_decoder_b:
            decoder_sensor_b_params = filter(lambda p: p.requires_grad, self.decoder_sensor_b.parameters())
            optimizer_decoder_sensor_b = radam.RAdam(decoder_sensor_b_params,
                                                     lr=self.settings.lr_decoder,
                                                     weight_decay=weight_decay,
                                                     betas=(0., 0.999))
            self.optimizers_dict["optimizer_decoder_sensor_b"] = optimizer_decoder_sensor_b

        # Refinement Task
        if self.settings.cross_refinement_a:
            refinement_a_params = filter(lambda p: p.requires_grad, self.cross_refinement_net_a.parameters())
            refinement_discr_a_params = filter(lambda p: p.requires_grad, self.refinement_discr_a.parameters())
            optimizer_refinement_a = radam.RAdam(refinement_a_params,
                                                 lr=self.settings.lr_decoder,
                                                 weight_decay=weight_decay,
                                                 betas=(0., 0.999))
            optimizer_refinement_discr_a = radam.RAdam(refinement_discr_a_params,
                                                       lr=self.settings.lr_discriminator,
                                                       weight_decay=weight_decay,
                                                       betas=(0., 0.999))
            self.optimizers_dict["optimizer_refinement_a"] = optimizer_refinement_a
            self.optimizers_dict["optimizer_refinement_discr_a"] = optimizer_refinement_discr_a

        if self.settings.cross_refinement_b:
            refinement_b_params = filter(lambda p: p.requires_grad, self.cross_refinement_net_b.parameters())
            refinement_discr_b_params = filter(lambda p: p.requires_grad, self.refinement_discr_b.parameters())
            optimizer_refinement_b = radam.RAdam(refinement_b_params,
                                                 lr=self.settings.lr_decoder,
                                                 weight_decay=weight_decay,
                                                 betas=(0., 0.999))
            optimizer_refinement_discr_b = radam.RAdam(refinement_discr_b_params,
                                                       lr=self.settings.lr_discriminator,
                                                       weight_decay=weight_decay,
                                                       betas=(0., 0.999))
            self.optimizers_dict["optimizer_refinement_b"] = optimizer_refinement_b
            self.optimizers_dict["optimizer_refinement_discr_b"] = optimizer_refinement_discr_b

    def train_step(self, input_batch):
        if not input_batch or input_batch[0][1].shape[0] == 0:
            print('Empty Labels  %s' % input_batch[0][1].shape[0])
            return {}, {}
        # alternate between gen loss and dis loss
        mod_step = self.step_count % (self.settings.front_iter + self.settings.disc_iter)

        if mod_step < self.settings.disc_iter:
            # Discriminator Step
            optimizers_list = ['optimizer_dis']
            if self.settings.cross_refinement_a:
                optimizers_list.append('optimizer_refinement_discr_a')
            if self.settings.cross_refinement_b:
                optimizers_list.append('optimizer_refinement_discr_b')

            for key_word in optimizers_list:
                optimizer_key_word = self.optimizers_dict[key_word]
                optimizer_key_word.zero_grad()

            d_final_loss, d_losses, d_outputs = self.discriminator_train_step(input_batch)

            d_final_loss.backward()
            for key_word in optimizers_list:
                optimizer_key_word = self.optimizers_dict[key_word]
                optimizer_key_word.step()

            return d_losses, d_outputs, d_final_loss
        else:
            # Front End Step
            optimizers_list = ['optimizer_front_sensor_a', 'optimizer_front_sensor_b', 'optimizer_front_shared']
            if self.settings.use_decoder_a:
                optimizers_list.append('optimizer_decoder_sensor_a')
            if self.settings.use_decoder_b:
                optimizers_list.append('optimizer_decoder_sensor_b')
            if self.settings.cross_refinement_a:
                optimizers_list.append('optimizer_refinement_a')
            if self.settings.cross_refinement_b:
                optimizers_list.append('optimizer_refinement_b')
            if self.settings.use_task_a or self.settings.use_task_b:
                optimizers_list.append('optimizer_back')

            for key_word in optimizers_list:
                optimizer_key_word = self.optimizers_dict[key_word]
                optimizer_key_word.zero_grad()

            g_final_loss, g_losses, g_outputs = self.generator_train_step(input_batch)
            g_final_loss.backward()

            for key_word in optimizers_list:
                optimizer_key_word = self.optimizers_dict[key_word]
                optimizer_key_word.step()

            return g_losses, g_outputs, g_final_loss

    def generator_train_step(self, batch):
        data_a = batch[0][0]
        labels_a = batch[0][1]
        data_b = batch[1][0]
        labels_b = batch[1][1]

        # Set BatchNorm Statistics to Train
        for model in self.models_dict:
            self.models_dict[model].train()

        gen_model_sensor_a = self.models_dict['front_sensor_a']
        gen_model_sensor_b = self.models_dict['front_sensor_b']

        losses = {}
        outputs = {}
        g_loss = 0.
        # Train generator.
        # Generator output.
        content_sensor_a, mu_sensor_a, logvar_sensor_a, attribute_sensor_a = gen_model_sensor_a(data_a)
        content_sensor_b, mu_sensor_b, logvar_sensor_b, attribute_sensor_b = gen_model_sensor_b(data_b)

        if self.settings.use_decoder_a:
            losses['var_content_space_sensor_a'] = content_sensor_a.var(dim=(-2, -1)).mean()
            losses['var_attribute_space_sensor_a'] = attribute_sensor_a.var(dim=(-1)).mean()
        if self.settings.use_decoder_b:
            losses['var_content_space_sensor_b'] = content_sensor_b.var(dim=(-2, -1)).mean()
            losses['var_attribute_space_sensor_b'] = attribute_sensor_b.var(dim=(-1)).mean()

        # Get discriminator prediction.
        g_loss += self.trainDiscriminatorStep(content_sensor_a, content_sensor_b, losses)

        if self.settings.use_task_a:
            g_loss += self.trainTaskStep('sensor_a', content_sensor_a, labels_a, losses)

        if self.settings.use_task_b:
            print("Label B")
            g_loss += self.trainTaskStep('sensor_b', content_sensor_b, labels_b, losses)

        if self.settings.use_cycle_a_b:
            out = self.trainCycleStep('sensor_a', 'sensor_b', content_sensor_a, mu_sensor_b,
                                      attribute_sensor_b, data_a, labels_a, losses, self.settings.use_task_a,
                                      self.settings.cross_refinement_b)
            g_loss += out[0]
            cross_decoder_output_b = out[1]

        if self.settings.use_cycle_b_a:
            g_loss += self.trainCycleStep('sensor_b', 'sensor_a', content_sensor_b, mu_sensor_a,
                                          attribute_sensor_a, data_b, labels_b, losses, self.settings.use_task_b,
                                          self.settings.cross_refinement_a)

        if self.settings.use_decoder_b and self.settings.sensor_b_name == 'events':
            g_loss += self.augmentFlowAttribute('sensor_b', cross_decoder_output_b, data_a, content_sensor_a, losses,
                                                translation=True)

        return g_loss, losses, outputs

    def trainDiscriminatorStep(self, content_sensor_a, content_sensor_b, losses):
        dis_model = self.models_dict['dis']
        input_disc = torch.cat([content_sensor_a, content_sensor_b], dim=0)
        logits = dis_model(input_disc)
        logits_sensor_a = logits[:self.settings.batch_size_a]
        logits_sensor_b = logits[self.settings.batch_size_a:]
        # Compute GAN loss.
        generator_loss_a = generator_loss_two_sensors("hinge", sensor_a=logits_sensor_a) * \
                           self.settings.weight_generator_loss
        generator_loss_b = generator_loss_two_sensors("hinge", sensor_b=logits_sensor_b) * \
                           self.settings.weight_generator_loss
        discr_loss = generator_loss_a + generator_loss_b
        losses['generator_loss'] = discr_loss.detach()
        losses['generator_sensor_a_loss'] = generator_loss_a.detach()
        losses['generator_sensor_b_loss'] = generator_loss_b.detach()

        return discr_loss

    def trainCycleStep(self, sensor_name, second_sensor_name, content_first_sensor, attribute_mu_second_sensor,
                       attribute_second_sensor, data_first_sensor, labels_first_sensor, losses, use_task_first_sensor,
                       cross_refinement_second_sensor):
        decoder_second_sensor = self.models_dict['decoder_' + second_sensor_name]
        gen_model_second_sensor = self.models_dict['front_' + second_sensor_name]

        decoder_output = decoder_second_sensor.forward(content_first_sensor, attribute_second_sensor)

        g_loss = 0
        if cross_refinement_second_sensor:
            cross_refinement_net = self.models_dict['cross_refinement_net_' + second_sensor_name]
            refinement_discr = self.models_dict['refinement_discr_' + second_sensor_name]

            out = cross_refinement_net.forward(decoder_output, data_first_sensor,
                                               return_clean_reconst=True, return_flow=True)
            reconst_second_sensor_input, clean_reconst, flow_map = out

            # Image Gradient Loss
            g_loss += self.trainImageGradientStep(data_first_sensor, clean_reconst, flow_map, second_sensor_name,
                                                  losses)

            # Flow Smoothness Loss
            smoothness_loss = self.flowSmoothnessLoss(flow_map) * self.settings.weight_smoothness_loss
            g_loss += smoothness_loss
            losses['optical_flow_smoothness_' + second_sensor_name + '_loss'] = smoothness_loss.detach()

            # Discriminator Step
            refinement_logits = refinement_discr(reconst_second_sensor_input)
            refined_generator_loss = generator_loss(loss_func='hinge', fake=refinement_logits) * \
                                     self.settings.weight_refinement_loss
            g_loss += refined_generator_loss
            losses['gen_refinement_' + second_sensor_name + '_loss'] = refined_generator_loss.detach()

        content_cycle, attribute_mu_cycle_second_sensor, _, _ = gen_model_second_sensor(reconst_second_sensor_input)

        # Cycle Content Loss
        cycle_content_loss = self.cycle_content_loss(content_first_sensor, content_cycle) * \
                             self.settings.weight_cycle_loss
        g_loss += cycle_content_loss
        cycle_name = sensor_name + '_to_' + second_sensor_name
        losses['cycle_content_' + cycle_name + '_loss'] = cycle_content_loss.cpu().detach()

        # Cycle Attribute Loss
        cycle_attribute_loss = self.cycle_attribute_loss(attribute_mu_second_sensor,
                                                         attribute_mu_cycle_second_sensor) * \
                               self.settings.weight_cycle_loss
        g_loss += cycle_attribute_loss
        cycle_name = sensor_name + '_to_' + second_sensor_name
        losses['cycle_attribute_' + cycle_name + '_loss'] = cycle_attribute_loss.detach()

        if self.settings.use_cycle_task and use_task_first_sensor:
            if sensor_name == 'sensor_b':
                print("Label B")
            g_loss += self.trainCycleAccuracyStep(sensor_name, reconst_second_sensor_input, labels_first_sensor, losses)

        if sensor_name == 'sensor_a':
            return g_loss, decoder_output
        return g_loss

    def trainCycleAccuracyStep(self, cycle_name, reconst_second_sensor_input, labels, losses):
        task_backend = self.models_dict["back_end"]
        event_encoder = self.models_dict['front_sensor_b']

        content_cycle, _, _, _ = event_encoder(reconst_second_sensor_input.detach())
        pred_sensor_cycle = task_backend(content_cycle)

        loss_pred = self.task_loss(pred_sensor_cycle, target=labels) * 0.1
        losses['classification_cycle_' + cycle_name + '_loss'] = loss_pred.detach()
        losses['task_cycle_' + cycle_name + '_acc'] = torch.mean(torch.eq(torch.argmax(pred_sensor_cycle, dim=-1),
                                                                          labels).float()).detach()
        return loss_pred

    def trainTaskStep(self, sensor_name, content_features, labels, losses):
        task_backend = self.models_dict["back_end"]
        pred_sensor = task_backend(content_features)
        loss_pred = self.task_loss(pred_sensor, target=labels) * self.settings.weight_task_loss
        losses['classification_' + sensor_name + '_loss'] = loss_pred.detach()
        losses['task_sensor_' + sensor_name + '_acc'] = torch.mean(torch.eq(torch.argmax(pred_sensor, dim=-1),
                                                                            labels).float()).detach()

        return loss_pred

    def trainImageGradientStep(self, data_first_sensor, clean_reconst, flow_map, second_sensor_name, losses):
        norm_gradient = self.norm_gradient_layer.forward(data_first_sensor)
        torch_smoothed = self.gaussian_layer(norm_gradient)
        gradient_loss = 0

        summed_events = torch.sum(clean_reconst, dim=1, keepdim=True)

        # Positive Loss
        pos_loss_bool = torch_smoothed > 0.3
        pos_loss_spatial = nn.functional.relu(0.7 - summed_events) * norm_gradient
        pos_loss = pos_loss_spatial[pos_loss_bool].mean()

        losses['pos_gradient_' + second_sensor_name + '_loss'] = pos_loss.detach()
        gradient_loss += pos_loss * 5

        if self.visualize_epoch():
            nrow = 4
            # spatial_loss = (pos_loss_bool * pos_loss_spatial - neg_loss_bool.int())
            spatial_loss = pos_loss_bool * pos_loss_spatial
            spatial_loss = (spatial_loss - spatial_loss.min()) / spatial_loss.max()
            spatial_loss = spatial_loss[:nrow].expand(-1, 3, -1, -1)
            viz_flow = viz_utils.visualizeFlow(flow_map[:nrow])

            viz_tensors = torch.cat((data_first_sensor[:nrow].expand(-1, 3, -1, -1),
                                     viz_flow,
                                     viz_utils.createRGBImage(clean_reconst[:nrow]),
                                     spatial_loss), dim=0)
            rgb_grid = torchvision.utils.make_grid(viz_tensors, nrow=nrow)
            self.img_summaries('train/image_gradient_' + second_sensor_name + '_img', rgb_grid, self.step_count)

        return gradient_loss

    def flowSmoothnessLoss(self, flow_map):
        """Computes the smoothness loss in a neighbour hood of 2 for each pixel based on the Charbonnier loss"""
        displ = flow_map
        displ_c = displ[..., 1:-1, 1:-1]

        displ_u = displ[..., 1:-1, 2:]
        displ_d = displ[..., 1:-1, 0:-2]
        displ_l = displ[..., 2:, 1:-1]
        displ_r = displ[..., 0:-2, 1:-1]

        displ_ul = displ[..., 2:, 2:]
        displ_dr = displ[..., 0:-2, 0:-2]
        displ_dl = displ[..., 0:-2, 2:]
        displ_ur = displ[..., 2:, 0:-2]

        loss = self.charbonnier_loss(displ_l - displ_c) +\
               self.charbonnier_loss(displ_r - displ_c) +\
               self.charbonnier_loss(displ_d - displ_c) +\
               self.charbonnier_loss(displ_u - displ_c) +\
               self.charbonnier_loss(displ_dl - displ_c) +\
               self.charbonnier_loss(displ_dr - displ_c) +\
               self.charbonnier_loss(displ_ul - displ_c) +\
               self.charbonnier_loss(displ_ur - displ_c)
        loss /= 8
        return loss

    def charbonnier_loss(self, delta, exponent=0.45, eps=1e-3):
        # alpha = 0.25
        # epsilon = 1e-8
        return (delta.pow(2) + eps**2).pow(exponent).mean()

    def augmentFlowAttribute(self, sensor_name, cross_decoder_output, data_first_sensor, content_first_sensor, losses,
                             translation=False):
        flow_map = cross_decoder_output[:, :2, :, :]
        height, width = self.settings.img_size_b[0], self.settings.img_size_b[1]

        # --- Flow Augmentation
        if not translation:
            # Flow is in Camera Coordinates: (x, y) = (u. v)
            b = flow_map.shape[0]
            random_x_center = (torch.rand(b, device=flow_map.device) * width).long()
            random_y_center = (torch.rand(b, device=flow_map.device) * (height / 4) + (height - height / 4) / 2).long()
            x_direction = (torch.arange(width, device=flow_map.device)[None, :] - random_x_center[:, None]) / (width / 2)
            y_direction = (torch.arange(height, device=flow_map.device)[None, :] - random_y_center[:, None]) / (height / 2)

        # --- Translation
        else:
            x_direction = (torch.rand([self.settings.batch_size_a, 1], device=flow_map.device) * 2 - 1).repeat([1, width])
            y_direction = (torch.rand([self.settings.batch_size_a, 1], device=flow_map.device) * 2 - 1).repeat([1, height])

        augmented_flow_vectors = torch.stack([x_direction[:, None, :].repeat([1, height, 1]),
                                              y_direction[:, :, None].repeat([1, 1, width])], dim=1)

        augmented_flow_vectors = nn.functional.normalize(augmented_flow_vectors, p=2, dim=1)

        pred_flow_magnitude = torch.sqrt((flow_map ** 2).sum(1, keepdim=True))
        augmented_flow = augmented_flow_vectors * pred_flow_magnitude
        # ------

        front_end_sensor_b = self.models_dict['front_sensor_b']
        augmented_input = torch.cat([augmented_flow, cross_decoder_output[:, 2:, :, :]], dim=1)

        cross_refinement_net = self.models_dict['cross_refinement_net_' + sensor_name]
        augmented_event_histo = cross_refinement_net.forward(augmented_input.detach(), data_first_sensor)
        augmented_attribute_f, _, _ = front_end_sensor_b(augmented_event_histo, attribute_only=True)

        decoder_network = self.models_dict['decoder_' + sensor_name]
        augmented_decoder_output = decoder_network.forward(content_first_sensor, augmented_attribute_f)

        reconst_loss = self.reconst_loss_dict[sensor_name](augmented_decoder_output[:, :2, :, :], augmented_flow) * \
                                self.settings.weight_reconstruction_sensor_b_loss

        losses['augmented_reconstruction_grayscale_' + sensor_name + '_loss'] = reconst_loss.detach()


        if self.visualize_epoch():
            nrow = 4
            viz_org_flow = viz_utils.visualizeFlow(flow_map[:nrow])
            viz_flow_augmented = viz_utils.visualizeFlow(augmented_flow[:nrow])
            viz_flow_predicted = viz_utils.visualizeFlow(augmented_decoder_output[:nrow, :2, :, :])

            viz_tensors = torch.cat((data_first_sensor[:nrow].expand(-1, 3, -1, -1),
                                     viz_org_flow,
                                     viz_flow_augmented,
                                     viz_flow_predicted,
                                     viz_utils.createRGBImage(augmented_event_histo[:nrow])), dim=0)
            rgb_grid = torchvision.utils.make_grid(viz_tensors, nrow=nrow)
            self.img_summaries('train/flow_augmentation' + sensor_name + '_img', rgb_grid, self.step_count)

        return reconst_loss

    def discriminator_train_step(self, batch):
        data_a = batch[0][0]
        data_b = batch[1][0]

        gen_model_sensor_a = self.models_dict['front_sensor_a']
        gen_model_sensor_b = self.models_dict['front_sensor_b']
        dis_model = self.models_dict['dis']

        with torch.no_grad():
            content_f_sensor_a, _, _, attribute_f_sensor_a = gen_model_sensor_a(data_a)
            content_f_sensor_b, _, _, attribute_f_sensor_b = gen_model_sensor_b(data_b)

        input_disc = torch.cat([content_f_sensor_a, content_f_sensor_b], dim=0)
        logits = dis_model(input_disc)
        logits_sensor_a = logits[:self.settings.batch_size_a]
        logits_sensor_b = logits[self.settings.batch_size_a:]

        d_loss = discriminator_loss("hinge", logits_sensor_a, logits_sensor_b) * self.settings.weight_discriminator_loss

        losses = {'discriminator_loss': d_loss,
                  'disc_sensor_a_acc': torch.mean(torch.eq(torch.sign(logits_sensor_a),
                                                           torch.ones(logits_sensor_a.size()).to(
                                                               self.device)).float()).detach(),
                  'disc_sensor_b_acc': torch.mean(torch.eq(torch.sign(logits_sensor_b),
                                                           -torch.ones(logits_sensor_b.size()).to(
                                                               self.device)).float()).detach()}

        if self.settings.cross_refinement_a:
            d_loss += self.discriminatorCrossStep(data_a, data_b, content_f_sensor_b, attribute_f_sensor_a,
                                                  'sensor_a', losses)

        if self.settings.cross_refinement_b:
            d_loss += self.discriminatorCrossStep(data_b, data_a, content_f_sensor_a, attribute_f_sensor_b,
                                                  'sensor_b', losses)
        outputs = {}

        return d_loss, losses, outputs

    def discriminatorCrossStep(self, data_first_sensor, data_second_sensor, content_second_sensor,
                               attribute_first_sensor, first_sensor, losses):
        with torch.no_grad():
            decoder_sensor_first = self.models_dict['decoder_' + first_sensor]
            reconst_sensor_second_to_first = decoder_sensor_first.forward(content_second_sensor, attribute_first_sensor)

            cross_refinement_net_a = self.models_dict['cross_refinement_net_' + first_sensor]
            refined_sensor_b_a = cross_refinement_net_a.forward(reconst_sensor_second_to_first, data_second_sensor)

        refinement_discr_a = self.models_dict['refinement_discr_' + first_sensor]
        fake_logits = refinement_discr_a(refined_sensor_b_a.detach())
        real_logits = refinement_discr_a(data_first_sensor)

        refine_discr_loss = discriminator_loss('hinge', real=real_logits, fake=fake_logits) * \
                            self.settings.weight_refinement_loss

        losses['discr_refinement_' + first_sensor + '_loss'] = refine_discr_loss.detach()
        discr_accuracy = torch.mean(torch.eq(torch.sign(fake_logits),
                                             -torch.ones(fake_logits.size()).to(self.device)).float())
        discr_accuracy += torch.mean(torch.eq(torch.sign(real_logits),
                                              torch.ones(real_logits.size()).to(self.device)).float())
        losses['discr_refinement_' + first_sensor + '_acc'] = (discr_accuracy / 2.0).detach()

        return refine_discr_loss

    def validationEpochs(self):
        self.resetValidationStatistics()

        with torch.no_grad():
            for model in self.models_dict:
                self.models_dict[model].eval()

            self.validationEpoch(WrapperDataset(self.val_loader_sensor_a, self.val_loader_sensor_b, self.device,
                                                dataset_len_to_use='first'), 'sensor_a')
            self.validationEpoch(WrapperDataset(self.val_loader_sensor_b, self.val_loader_sensor_a, self.device,
                                                dataset_len_to_use='first'), 'sensor_b')

            if self.do_val_training_epoch:
                self.trainDatasetStatisticsEpoch('sensor_a', self.train_loader_sensor_a)
                self.trainDatasetStatisticsEpoch('sensor_b', self.train_loader_sensor_b)

            self.resetValidationStatistics()

        self.pbar.close()

    def val_step(self, input_batch, sensor, i_batch, vis_reconstr_idx):
        """Calculates the performance measurements based on the input"""
        data = input_batch[0][0]
        labels = input_batch[0][1]
        data_second_sensor = input_batch[1][0]

        gen_model = self.models_dict['front_' + sensor]
        dis_model = self.models_dict['dis']

        content_f_first_sensor, _, _, attribute_f_first_sensor = gen_model(data)
        logits_sensor = dis_model(content_f_first_sensor)

        losses = {}
        second_sensor = 'sensor_b'

        if sensor == 'sensor_a':
            losses['generator_sensor_a_loss'] = generator_loss_two_sensors("hinge", sensor_a=logits_sensor).detach()
            losses['discriminator_sensor_a_loss'] = discriminator_loss("hinge", real=logits_sensor).detach()
            losses['disc_sensor_a_acc'] = torch.mean(torch.eq(torch.sign(logits_sensor),
                                                              torch.ones(logits_sensor.size()).to(
                                                                  self.device)).float()).detach()
            cycle_loss_bool = self.settings.use_cycle_a_b
            cross_refinement_bool = self.settings.cross_refinement_b
            decoder_bool = self.settings.use_decoder_a
        else:
            losses['generator_sensor_b_loss'] = generator_loss_two_sensors("hinge", sensor_b=logits_sensor).detach()
            losses['discriminator_sensor_b_loss'] = discriminator_loss("hinge", fake=logits_sensor).detach()
            losses['disc_sensor_b_acc'] = torch.mean(torch.eq(torch.sign(logits_sensor),
                                                              -torch.ones(logits_sensor.size()).to(
                                                                  self.device)).float()).detach()
            second_sensor = 'sensor_a'
            cycle_loss_bool = self.settings.use_cycle_b_a
            cross_refinement_bool = self.settings.cross_refinement_a
            decoder_bool = self.settings.use_decoder_b

        if self.settings.use_task_a or self.settings.use_task_b:
            self.valTaskStep(content_f_first_sensor, labels, data, losses, sensor, vis_reconstr_idx=vis_reconstr_idx)

        if cycle_loss_bool:
            self.valCycleStep(content_f_first_sensor, data, data_second_sensor, labels, losses, sensor, second_sensor,
                              cross_refinement_bool)

        if vis_reconstr_idx != -1:
            self.visualizeReconstructions(data, data_second_sensor, content_f_first_sensor, attribute_f_first_sensor,
                                          vis_reconstr_idx, sensor, second_sensor, decoder_bool, cycle_loss_bool,
                                          cross_refinement_bool)

        return losses, None

    def visualizeReconstructions(self, data, data_second_sensor, content_f_first_sensor, attribute_f_first_sensor,
                                 vis_reconstr_idx, sensor, second_sensor, decoder_bool, cross_reconstruction,
                                 cross_refinement_bool):
        nrow = 4
        vis_tensors = [viz_utils.createRGBImage(data[:nrow])]

        if cross_reconstruction:
            encoder_second_sensor = self.models_dict["front_" + second_sensor]
            decoder_second_sensor = self.models_dict["decoder_" + second_sensor]
            _, _, _, attribute_f_second_sensor, = encoder_second_sensor(data_second_sensor)
            decoder_out = decoder_second_sensor(content_f_first_sensor, attribute_f_second_sensor)
            vis_tensors.append(viz_utils.visualizeFlow(decoder_out[:nrow, :2]))

            if cross_refinement_bool:
                cross_refinement_net_second_sensor = self.models_dict['cross_refinement_net_' + second_sensor]
                refined_second_sensor_img = cross_refinement_net_second_sensor(decoder_out,
                                                                               data)
                vis_tensors.append(viz_utils.createRGBImage(refined_second_sensor_img[:nrow]).clamp(0, 1))

        rgb_grid = torchvision.utils.make_grid(torch.cat(vis_tensors, dim=0), nrow=nrow)
        self.img_summaries('val_' + sensor + '/reconst_input_' + sensor + '_' + str(vis_reconstr_idx),
                           rgb_grid, self.epoch_count)

    def valTaskStep(self, content_f_first_sensor, labels, data, losses, sensor, vis_reconstr_idx):
        """Computes the task loss and visualizes the detected bounding boxes"""
        task_backend = self.models_dict["back_end"]
        pred = task_backend(content_f_first_sensor)

        loss_pred = self.task_loss(pred, target=labels)

        losses['classification_' + sensor + '_loss'] = loss_pred.detach()
        losses['task_sensor_' + sensor + '_acc'] = torch.mean(torch.eq(torch.argmax(pred, dim=-1),
                                                                       labels).float()).detach()
        np.add.at(self.val_confusion_matrix, (torch.argmax(pred, dim=-1).data.cpu().numpy(),
                                              labels.data.cpu().numpy()), 1)

    def valCycleStep(self, content_f_first_sensor, data, data_second_sensor, labels, losses, sensor, second_sensor,
                     cross_refinement_bool):
        """Computes the cycle loss"""
        decoder_second_sensor = self.models_dict["decoder_" + second_sensor]
        gen_second_sensor_model = self.models_dict['front_' + second_sensor]
        _, _, _, attribute_f_second_sensor = gen_second_sensor_model(data_second_sensor)
        reconst_second_sensor_img = decoder_second_sensor(content_f_first_sensor, attribute_f_second_sensor)

        if cross_refinement_bool:
            cross_refinement_net_second_sensor = self.models_dict['cross_refinement_net_' + second_sensor]
            reconst_second_sensor_img = cross_refinement_net_second_sensor(reconst_second_sensor_img,
                                                                           data)
            refinement_discr_second_sensor = self.models_dict['refinement_discr_' + second_sensor]
            fake_discr_logits = refinement_discr_second_sensor(reconst_second_sensor_img)
            refined_generator_loss = generator_loss(loss_func='hinge', fake=fake_discr_logits) * \
                                     self.settings.weight_refinement_loss
            losses['refinement_discr_' + second_sensor] = refined_generator_loss.detach()

        cycle_content_first_second, _, _, _ = gen_second_sensor_model(reconst_second_sensor_img)
        cycle_loss_first_second = self.cycle_content_loss(content_f_first_sensor, cycle_content_first_second) * \
                                  self.settings.weight_cycle_loss
        cycle_name = 'cycle_' + sensor + '_to_' + second_sensor
        losses[cycle_name + '_loss'] = cycle_loss_first_second.detach()

        if self.settings.use_cycle_task:
            self.valCycleTask(cycle_content_first_second, labels, cycle_name, losses)

    def valCycleTask(self, cycle_content_first_second, labels, cycle_name, losses):
        """Computes the task performance of the cylce reconstruction"""
        task_backend = self.models_dict["back_end"]
        pred_second_sensor = task_backend(cycle_content_first_second)
        pred_class_second_sensor = torch.argmax(pred_second_sensor, dim=-1)
        losses[cycle_name + '_acc'] = torch.mean(torch.eq(labels, pred_class_second_sensor).float()).detach()

    def val_train_stats_step(self, input_batch, sensor, i_batch, cumulative_losses):
        """Calculates the performance measurements based on the input"""
        data, labels = input_batch
        gen_model = self.models_dict['front_' + sensor]
        task_backend = self.models_dict["back_end"]
        content_f_first_sensor, _, _, attribute_f_first_sensor, = gen_model(data)

        pred = task_backend(content_f_first_sensor)

        key = 'task_training_sensor_' + sensor + '_acc'
        if key in cumulative_losses:
            cumulative_losses[key] += torch.mean(torch.eq(torch.argmax(pred, dim=-1), labels).float()).detach()
        else:
            cumulative_losses[key] = torch.mean(torch.eq(torch.argmax(pred, dim=-1), labels).float()).detach()
