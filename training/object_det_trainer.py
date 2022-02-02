import torch
import numpy as np
import torchvision.models as models

import utils.viz_utils as viz_utils
from models import refinement_net, yolov3, yolov3_modules
from models.style_networks import StyleEncoder, StyleDecoder, ContentDiscriminator, CrossDiscriminator
from utils import utils, object_detection_eval
import training.classification_trainer


class ObjectDetModel(training.classification_trainer.ClassificationModel):
    def __init__(self, settings, train=True):
        self.is_training = train
        super(ObjectDetModel, self).__init__(settings)

        self.train_statistics = {}
        # Mean Average Precision
        self.val_statistics = {}
        self.do_val_training_epoch = False
        for sensor_name in ['sensor_a', 'sensor_b']:
            self.val_statistics[sensor_name + '_gt_bboxes_mAP'] = []
            self.val_statistics[sensor_name + '_det_bboxes_mAP'] = []

    def buildModels(self):
        attribute_channels = 8
        config_path = 'models/yolov3_singleclass.cfg'
        # Shared Layers in the middle of the network
        self.front_end_shared = list(models.resnet18(pretrained=True).children())[5][1]

        # Front End Sensor A
        self.front_end_sensor_a = StyleEncoder(self.settings.input_channels_a, self.front_end_shared,
                                               attribute_channels, use_attributes=self.settings.use_decoder_a)

        # Front End Sensor B
        self.front_end_sensor_b = StyleEncoder(self.settings.input_channels_b, self.front_end_shared,
                                               attribute_channels, use_attributes=self.settings.use_decoder_b)

        # Discriminator
        self.discriminator = ContentDiscriminator(nr_channels=128, smaller_input=True)

        self.models_dict = {"front_sensor_a": self.front_end_sensor_a,
                            "front_sensor_b": self.front_end_sensor_b,
                            "front_shared": self.front_end_shared,
                            "dis": self.discriminator}

        # Task Backend
        if self.settings.use_task_a or self.settings.use_task_b:
            self.task_backend = yolov3.YoloTask(config_path,
                                                img_height=self.settings.img_size_a[0],
                                                img_width=self.settings.img_size_a[1])

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
            self.refinement_discr_a = CrossDiscriminator(input_dim=self.settings.input_channels_a, n_layer=5)
            self.models_dict["cross_refinement_net_sensor_a"] = self.cross_refinement_net_a
            self.models_dict["refinement_discr_sensor_a"] = self.refinement_discr_a

        if self.settings.cross_refinement_b:
            self.cross_refinement_net_b = refinement_net.StyleRefinementNetwork(input_c=2,
                                                                                output_c=self.settings.input_channels_b,
                                                                                sensor=self.settings.sensor_b_name,
                                                                                channel_list=[16, 8],
                                                                                last_layer_pad=1,
                                                                                device=self.device)
            self.refinement_discr_b = CrossDiscriminator(input_dim=self.settings.input_channels_b, n_layer=5)
            self.models_dict["cross_refinement_net_sensor_b"] = self.cross_refinement_net_b
            self.models_dict["refinement_discr_sensor_b"] = self.refinement_discr_b

    def generator_train_step(self, batch):
        data_a = batch[0][0]
        labels_a = batch[0][1]
        data_b = batch[1][0]
        labels_b = batch[1][1]

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
            g_loss += self.trainTaskStep('sensor_a', data_a, content_sensor_a, labels_a, losses)

        if self.settings.use_task_b:
            print("Label B")
            g_loss += self.trainTaskStep('sensor_b', data_b, content_sensor_b, labels_b, losses)

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
            g_loss += self.augmentFlowAttribute('sensor_b', cross_decoder_output_b, data_a, content_sensor_a, losses)

        return g_loss, losses, outputs

    def trainCycleAccuracyStep(self, cycle_name, reconst_second_sensor_input, labels, losses):
        task_backend = self.models_dict["back_end"]
        event_encoder = self.models_dict['front_sensor_b']

        content_cycle, _, _, _ = event_encoder(reconst_second_sensor_input.detach())
        loss_pred, _, step_metric = task_backend(content_cycle, labels)

        loss_pred = loss_pred * self.settings.weight_task_loss
        losses['object_det_cycle_' + cycle_name + '_loss'] = loss_pred.detach().cpu().numpy()

        avg_cycle_recall, avg_cycle_precision = self.extractAvgRecallPrecision(step_metric)
        losses['task_cycle_recall_' + cycle_name + '_acc'] = avg_cycle_recall
        losses['task_cycle_precision_' + cycle_name + '_acc'] = avg_cycle_precision

        return loss_pred

    def trainTaskStep(self, sensor_name, input_data, content_features, labels, losses):
        task_backend = self.models_dict["back_end"]
        loss_pred, pred, step_metric = task_backend(content_features, labels)
        loss_pred = loss_pred * self.settings.weight_task_loss

        avg_cycle_recall, avg_cycle_precision = self.extractAvgRecallPrecision(step_metric)
        losses['task_recall_' + sensor_name + '_acc'] = avg_cycle_recall
        losses['task_precision_' + sensor_name + '_acc'] = avg_cycle_precision
        losses['object_det_' + sensor_name + '_loss'] = loss_pred.detach().cpu().numpy()

        if self.step_count % (3 * int(self.train_loader.__len__())) == 0:
            img_tag = 'train/' + sensor_name + '_input_img'
            self.visualizeGtPredictedBbox(pred, labels, input_data, step=self.step_count, tag=img_tag)

        return loss_pred

    def valTaskStep(self, content_f_first_sensor, labels, data, losses, sensor_name, vis_reconstr_idx):
        """Computes the task loss and visualizes the detected bounding boxes"""
        task_backend = self.models_dict["back_end"]

        loss_pred, pred, step_metric = task_backend(content_f_first_sensor, labels)

        avg_cycle_recall, avg_cycle_precision = self.extractAvgRecallPrecision(step_metric)
        losses['object_det_' + sensor_name + '_loss'] = loss_pred.detach().cpu().numpy()
        losses['task_recall_' + sensor_name + '_acc'] = avg_cycle_recall
        losses['task_precision_' + sensor_name + '_acc'] = avg_cycle_precision

        img_shape = self.settings.img_size_a

        if sensor_name == 'sensor_b':
            img_shape = self.settings.img_size_b

        detected_bboxes = self.extractPredictedBBoxNMS(pred, img_shape)
        self.addBBoxmAP(detected_bboxes.cpu().numpy(), labels.cpu().numpy(), nr_imgs=data.shape[0], tag=sensor_name)

        if vis_reconstr_idx != -1:
            img_tag = 'val/' + sensor_name + '_input_img_' + str(vis_reconstr_idx)
            self.visualizeGtPredictedBbox(detected_bboxes, labels, data, step=self.epoch_count, tag=img_tag,
                                          nms_extracted_bbox=True)

    def valCycleTask(self, cycle_content_first_second, labels, cycle_name, losses):
        """Computes the task performance of the cylce reconstruction"""
        task_backend = self.models_dict["back_end"]
        loss_pred, _, step_metric = task_backend(cycle_content_first_second, labels)

        avg_cycle_recall, avg_cycle_precision = self.extractAvgRecallPrecision(step_metric)
        losses['object_det_' + cycle_name + '_loss'] = loss_pred.detach().cpu().numpy()
        losses['task_recall_' + cycle_name + '_acc'] = avg_cycle_recall
        losses['task_precision_' + cycle_name + '_acc'] = avg_cycle_precision

    def val_train_stats_step(self, input_batch, sensor, i_batch, cumulative_losses):
        """Calculates the performance measurements based on the input"""
        pass

    def saveValStatistics(self, mode, sensor):
        """Compute the mAP based on the VOC Pascal challenge"""
        self.computeMeanAveragePrecision(sensor, field_tag=mode)
        self.resetValidationStatistics()

    def resetValidationStatistics(self):
        """Creates empty lists again"""
        self.val_statistics = {}
        for sensor_name in ['sensor_a', 'sensor_b']:
            self.val_statistics[sensor_name + '_gt_bboxes_mAP'] = []
            self.val_statistics[sensor_name + '_det_bboxes_mAP'] = []

    def extractAvgRecallPrecision(self, metric):
        """Returns the average value for recall and precision for each detection scale in yolov3"""
        cycle_recall = 0
        cycle_precision = 0
        counter = 0
        for k, v in metric.items():
            if k[-12:] == 'recall50_acc':
                cycle_recall += v
            elif k[-13:] == 'precision_acc':
                cycle_precision += v
                counter += 1
        return cycle_recall / counter, cycle_precision / counter

    def extractPredictedBBoxNMS(self, pred, img_shape):
        """
        Extracts the bounding boxes with an objectness score above 0.5 and applies non maximum suppression

        :return filtered_boxes: [img_id, x_min, y_min, x_max. y_max, object_score, pred_class_id, class_score]
        """
        nr_img, nr_bbox = pred.shape[:2]
        detected_bboxes_bool = pred[:, :, 4] > 0.5  # Object score above 0.5

        if detected_bboxes_bool.sum() == 0:
            return torch.zeros([0, 8])

        img_id = torch.arange(end=nr_img, device=pred.device)[:, None].expand([-1, nr_bbox])
        detected_bboxes = pred[detected_bboxes_bool]
        detected_bboxes = utils.cropBboxToFrame(detected_bboxes, image_shape=img_shape)

        detected_bboxes_img_ids = img_id[detected_bboxes_bool]
        pred_class_score, pred_class = torch.max(detected_bboxes[:, 5:], dim=1)

        detected_bboxes = torch.cat([detected_bboxes_img_ids[:, None],
                                     detected_bboxes[:, :5],
                                     pred_class[:, None],
                                     pred_class_score[:, None]], dim=-1)

        filtered_bboxes = yolov3_modules.nonMaxSuppression(detected_bboxes, iou=0.5)

        return filtered_bboxes

    def addBBoxmAP(self, detected_bboxes, labels, nr_imgs, tag):
        """Adds the detected bboxes and labels to the corresponding lists"""
        det_bboxes = [[] for _ in range(nr_imgs)]
        gt_bboxes = [[] for _ in range(nr_imgs)]

        eval_bbox = detected_bboxes[:, [0, 1, 2, 3, 4, 6, 7]]
        eval_bbox[:, 3:5] = eval_bbox[:, 1:3] + eval_bbox[:, 3:5]

        for i_bbox, det_bbox in enumerate(eval_bbox):
            # det_bbox: [img_id, x_min, y_min, x_max, y_max, class_id, class_score]
            det_bboxes[int(det_bbox[0])].append(eval_bbox[i_bbox, 1:])

        labels[:, 3:5] = labels[:, 1:3] + labels[:, 3:5]
        # print(labels)
        for i_bbox, gt_bbox in enumerate(labels):
            # # Gt Bboxes: [x_min, y_min, x_max, y_max, class_id]
            gt_bboxes[int(gt_bbox[0])].append(gt_bbox[1:])

        self.val_statistics[tag + '_gt_bboxes_mAP'] = self.val_statistics[tag + '_gt_bboxes_mAP'] + gt_bboxes
        self.val_statistics[tag + '_det_bboxes_mAP'] = self.val_statistics[tag + '_det_bboxes_mAP'] + det_bboxes

    def visualizeGtPredictedBbox(self, detected_bbox, labels, input_data, step, tag='train', nms_extracted_bbox=False):
        """Visualizes the detected bounding box as well as the ground truth bounding boxes for the first image"""
        rgb_img = viz_utils.visualizeTensors(input_data[0, None, ...]).squeeze(0)
        viz_img = rgb_img.cpu().numpy().transpose([1, 2, 0])

        gt_bbox = labels[labels[:, 0] == 0]
        viz_img = viz_utils.drawBoundingBoxes(viz_img,
                                              bounding_boxes=gt_bbox[:, 1:5].int().cpu().numpy(),
                                              class_name=[self.object_classes[label] for label in gt_bbox[:, -1]],
                                              ground_truth=True)

        if not nms_extracted_bbox:
            detected_bboxes_bool = detected_bbox[0, :, 4] > 0.5
            if detected_bboxes_bool.sum() == 0:
                self.img_summaries(tag, viz_img.transpose([2, 0, 1]).astype(float), step)
                return

            bbox_to_viz = detected_bbox[0, detected_bboxes_bool, :4].int()
            bbox_to_viz = utils.cropBboxToFrame(bbox_to_viz, image_shape=self.settings.img_size_b).int().cpu().numpy()
            class_pred = torch.argmax(detected_bbox[0, detected_bboxes_bool, 5:], dim=-1).int().cpu().numpy()
        else:
            detected_bbox = detected_bbox.cpu().numpy()
            detected_bbox = detected_bbox[np.equal(detected_bbox[:, 0], 0), 1:]
            if detected_bbox.shape[0] == 0:
                self.img_summaries(tag, viz_img.transpose([2, 0, 1]).astype(float), step)
                return
            bbox_to_viz = detected_bbox[:, :4].astype(np.int)
            class_pred = detected_bbox[:, 5].astype(np.int)

        viz_img = viz_utils.drawBoundingBoxes(viz_img,
                                              bounding_boxes=bbox_to_viz,
                                              class_name=[self.object_classes[pred] for pred in class_pred],
                                              ground_truth=False)

        self.img_summaries(tag, viz_img.transpose([2, 0, 1]).astype(float), step)

    def computeMeanAveragePrecision(self, sensor_name, field_tag='val'):
        """Adds the average precision for each class as well as the mean average precision"""
        out = object_detection_eval.evaluatePascalVOCMetrics(self.val_statistics[sensor_name + '_gt_bboxes_mAP'],
                                                             self.val_statistics[sensor_name + '_det_bboxes_mAP'],
                                                             nr_classes=len(self.object_classes))
        class_AP = np.zeros(len(self.object_classes), dtype=np.float)

        sensor_modality_name = self.settings.sensor_b_name
        if sensor_name == 'sensor_a':
            sensor_modality_name = self.settings.sensor_a_name

        for i_class, class_name in enumerate(self.object_classes):
            sensor_tag = sensor_modality_name + '_class_' + class_name + '_AP_acc'
            self.summary_writer.add_scalar(field_tag + "/{}".format(sensor_tag), out[i_class]['AP'], self.epoch_count)
            class_AP[i_class] = out[i_class]['AP']

        print('----- mAP ------')
        print(sensor_name)
        print(np.mean(class_AP))
        print('----- - ------')

        sensor_tag = sensor_modality_name + '_class_mAP_acc'
        self.summary_writer.add_scalar(field_tag + "/{}".format(sensor_tag), np.mean(class_AP), self.epoch_count)

