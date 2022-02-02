"""
Code was adopted from: https://github.com/eriklindernoren/PyTorch-YOLOv3/
"""
import torch
import torch.nn as nn

from models.submodules import Normalize2D
from models.yolov3_modules import parse_model_config, build_targets


class YoloEncoder(nn.Module):
    def __init__(self, input_c, shared_layers, config_path):
        super(YoloEncoder, self).__init__()
        module_defs = parse_model_config(config_path, network_part='Encoder')
        module_defs = [{'input_channels': input_c}] + module_defs

        self.shared_layers = shared_layers
        self.module_list, self.module_defs = create_modules(module_defs)

    def forward(self, x):
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            layer_outputs.append(x)

        x = self.shared_layers(x)

        return x, None, None, None


class YoloShared(nn.Module):
    def __init__(self, config_path):
        super(YoloShared, self).__init__()
        module_defs = parse_model_config(config_path, network_part='Shared')
        self.module_list, self.module_defs = create_modules(module_defs)
        self.normalization_layer = Normalize2D()

    def forward(self, x):
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            else:
                print('Module {} is not defined'.format(module_def))
            layer_outputs.append(x)

        return self.normalization_layer(x)


class YoloTask(nn.Module):
    def __init__(self, config_path, img_height, img_width):
        super(YoloTask, self).__init__()
        module_defs = parse_model_config(config_path, network_part='TaskBranch')
        self.module_list, self.module_defs = create_modules(module_defs, img_height, img_width)

    def forward(self, x, targets=None):
        loss = 0
        layer_outputs, yolo_outputs = [], []
        scale_metrics = {}
        self.scale = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss, metrics = module[0](x, targets)
                if targets is not None:
                    self.addScaleMetrics(scale_metrics, metrics)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = torch.cat(yolo_outputs, 1).detach()

        if targets is None:
            return None, yolo_outputs, None
        else:
            return (loss, yolo_outputs, scale_metrics)

    def addScaleMetrics(self, scale_metrics, metrics):
        """Combines both dictionnaries"""
        scale_name = 'high_'
        if self.scale == 0:
            scale_name = 'low_'
        elif self.scale == 1:
            scale_name = 'middle_'
        self.scale += 1

        for key in metrics:
            scale_metrics[scale_name + key] = metrics[key]


def create_modules(module_defs, img_height=None, img_width=None):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    input_channels = module_defs.pop(0)
    output_filters = [int(input_channels["input_channels"])]
    filters = int(input_channels["input_channels"])
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            # pad = (kernel_size - 1) // 2
            pad = int(module_def["pad"])
            modules.add_module(f"conv_{module_i}",
                               nn.Conv2d(in_channels=output_filters[-1],
                                         out_channels=filters,
                                         kernel_size=kernel_size,
                                         stride=int(module_def["stride"]),
                                         padding=pad,
                                         bias=not bn))
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.01, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            if 'pad' in module_def:
                pad = int(module_def['pad'])
            else:
                pad = int((kernel_size - 1) // 2)
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=pad)
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height=img_height, img_width=img_width)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, module_defs


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, img_height, img_width):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.img_height = img_height
        self.img_width = img_width
        self.grid_size_h = None
        self.grid_size_w = None
        self.anchor_w = None
        self.anchor_h = None

    def compute_grid_offsets(self, grid_size_height, grid_size_width, cuda=True):
        self.grid_size_h = grid_size_height
        self.grid_size_w = grid_size_width
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride_h = self.img_height / self.grid_size_h
        self.stride_w = self.img_width / self.grid_size_w
        # Calculate offsets for each grid
        g_h = grid_size_height
        g_w = grid_size_width

        self.grid_x = torch.arange(g_h)[:, None].repeat(1, g_w).view([1, 1, g_h, g_w]).type(FloatTensor)
        self.grid_y = torch.arange(g_w)[None, :].repeat(g_h, 1).view([1, 1, g_h, g_w]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_h / self.stride_h, a_w / self.stride_w) for a_w, a_h in self.anchors])
        self.anchor_h = self.scaled_anchors[:, 0].view((1, self.num_anchors, 1, 1))
        self.anchor_w = self.scaled_anchors[:, 1].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None):
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        batch_size = x.size(0)
        grid_size_height = x.size(2)
        grid_size_width = x.size(3)

        prediction = (x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size_height, grid_size_width)
                      .permute(0, 1, 3, 4, 2).contiguous())

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        h = prediction[..., 2]  # Height
        w = prediction[..., 3]  # Width
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        if self.anchor_h is None:
            self.compute_grid_offsets(grid_size_height, grid_size_width, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 2] = torch.exp(h) * self.anchor_h * self.stride_h
        pred_boxes[..., 3] = torch.exp(w) * self.anchor_w * self.stride_w
        pred_boxes[..., 0] = (x + self.grid_x) * self.stride_h - (pred_boxes[..., 2] / 2)
        pred_boxes[..., 1] = (y + self.grid_y) * self.stride_w - (pred_boxes[..., 3] / 2)

        output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                            pred_conf.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, self.num_classes)), -1)

        if targets is None:
            return output, 0, None
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, th, tw, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
                img_shape=(self.img_height, self.img_width)
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            # iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            # recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            metrics = {
                "total_loss": (total_loss.detach().cpu()).item(),
                "x_loss": (loss_x.detach().cpu()).item(),
                "y_loss": (loss_y.detach().cpu()).item(),
                "w_loss": (loss_w.detach().cpu()).item(),
                "h_loss": (loss_h.detach().cpu()).item(),
                "object_confidence_loss": (loss_conf.detach().cpu()).item(),
                "classification_loss": (loss_cls.detach().cpu()).item(),
                "classification_acc": (cls_acc.detach().cpu()).item(),
                "recall50_acc": (recall50.detach().cpu()).item(),
                # "recall75": (recall75.detach().cpu()).item(),
                "precision_acc": (precision.detach().cpu()).item(),
                "conf_obj": (conf_obj.detach().cpu()).item(),
                "conf_noobj": (conf_noobj.detach().cpu()).item()
            }

            return output, total_loss, metrics
