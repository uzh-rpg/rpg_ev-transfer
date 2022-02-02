import torch
import torchvision


def bbox_wh_iou(hw1, hw2):
    hw2 = hw2.t()
    h1, w1 = hw1[0], hw1[1]
    h2, w2 = hw2[0], hw2[1]
    inter_area = torch.min(h1, h2) * torch.min(w1, w2)
    union_area = (h1 * w1 + 1e-16) + h2 * w2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from ['x', 'y', 'w', 'h'] to ['x_1', 'y_1', 'x_2', 'y_2']
        b1_x1, b1_x2 = box1[:, 0], box1[:, 0] + box1[:, 2]
        b1_y1, b1_y2 = box1[:, 1], box1[:, 1] + box1[:, 3]
        b2_x1, b2_x2 = box2[:, 0], box2[:, 0] + box2[:, 2]
        b2_y1, b2_y2 = box2[:, 1], box2[:, 1] + box2[:, 3]
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, img_shape):
    """
    Assigns the ground truth bounding boxes to the anchors and grid locations.

    :param pred_boxes: grid of predicted bounding boxes with objectness score
    :param pred_cls: class prediction score
    :param target: ground truth bounding boxes N x ['batch_id', 'x', 'y', 'w', 'h', 'class_id']
    :param anchors: prior selected ground truth boxes
    :param ignore_thres: threshold to ignore a prediction for the loss function
    """
    nB, nA, nG_h, nG_w, _ = pred_boxes.shape
    nC = pred_cls.size(-1)
    device = pred_cls.device

    # Create Output tensors
    shape = [nB, nA, nG_h, nG_w]
    obj_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    noobj_mask = torch.ones(shape, dtype=torch.bool, device=device)
    class_mask = torch.zeros(shape, dtype=torch.float, device=device)
    iou_scores = torch.zeros(shape, dtype=torch.float, device=device)
    tx = torch.zeros(shape, dtype=torch.float, device=device)
    ty = torch.zeros(shape, dtype=torch.float, device=device)
    th = torch.zeros(shape, dtype=torch.float, device=device)
    tw = torch.zeros(shape, dtype=torch.float, device=device)
    tcls = torch.zeros([nB, nA, nG_h, nG_w, nC], dtype=torch.float, device=device)

    target_boxes = target[:, 1:5].float()

    # Convert x, y, to grit coordinates
    scale_factor = torch.FloatTensor((nG_h, nG_w))[None, :].to(device) / torch.FloatTensor(img_shape)[None, :].to(device)
    gxy = target_boxes[:, :2] * scale_factor
    ghw = target_boxes[:, 2:] * scale_factor

    # Convert x, y to center coordinates
    gxy = gxy + (ghw / 2)

    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, ghw) for anchor in anchors])
    best_ious, best_n = ious.max(0)

    # Separate target values
    b = target[:, 0].long()
    class_labels = target[:, -1].long()
    gx, gy = gxy.t()
    gh, gw = ghw.t()
    gi, gj = gxy.long().t()

    # Set masks
    obj_mask[b, best_n, gi, gj] = 1
    noobj_mask[b, best_n, gi, gj] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    noobj_mask[b, :, gi, gj] = (ious < ignore_thres).t() * noobj_mask[b, :, gi, gj]

    # Coordinates
    tx[b, best_n, gi, gj] = gx - gx.floor()
    ty[b, best_n, gi, gj] = gy - gy.floor()

    # Width and height
    th[b, best_n, gi, gj] = torch.log(gh / anchors[best_n][:, 0] + 1e-16)
    tw[b, best_n, gi, gj] = torch.log(gw / anchors[best_n][:, 1] + 1e-16)

    # One-hot encoding of label
    tcls[b, best_n, gi, gj, class_labels] = 1

    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gi, gj] = (pred_cls[b, best_n, gi, gj].argmax(-1) == class_labels).float()
    iou_scores[b, best_n, gi, gj] = bbox_iou(pred_boxes[b, best_n, gi, gj], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()

    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, th, tw, tcls, tconf


def parse_model_config(path, network_part):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    in_network_part = False
    for i, line in enumerate(lines):
        if line.startswith('--'):
            if line[2:-2] == network_part:
                in_network_part = True
            else:
                in_network_part = False
            continue
        if not in_network_part:
            continue

        if line[:14] == 'input_channels':
            module_defs.append({'input_channels': line.split("=")[1]})
            continue

        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def nonMaxSuppression(detected_bbox, iou=0.6):
    """
    Iterates over the bboxes to peform non maximum suppression within each batch.

    :param detected_bbox[0, :]: [batch_idx, top_left_corner_u,  top_left_corner_v, width, height, object_score, ...])
    :param iou: intersection over union, threshold for which the bbox are considered overlapping
    """
    i_sample = 0
    keep_bbox = []

    while i_sample < detected_bbox.shape[0]:
        same_batch_mask = detected_bbox[:, 0] == detected_bbox[i_sample, 0]
        nms_input = detected_bbox[same_batch_mask][:, [1, 2, 3, 4, 5]].clone()
        nms_input[:, [2, 3]] += nms_input[:, [0, 1]]

        # (u, v) or (x, y) should not matter
        keep_idx = torchvision.ops.nms(nms_input[:, :4], nms_input[:, 4], iou)
        keep_bbox.append(detected_bbox[same_batch_mask][keep_idx])
        i_sample += same_batch_mask.sum()

    if len(keep_bbox) != 0:
        filtered_bbox = torch.cat(keep_bbox, dim=0)
    else:
        filtered_bbox = torch.zeros([0, 8])

    return filtered_bbox