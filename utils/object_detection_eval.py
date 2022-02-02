import numpy as np


def bbox_iou(boxs1, boxs2):
    """
    Returns the matrix containing the IoU of between the bounding boxes in boxs1 and boxs2
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = boxs1[:, 0], boxs1[:, 1], boxs1[:, 2], boxs1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxs2[:, 0], boxs2[:, 1], boxs2[:, 2], boxs2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1[:, np.newaxis], b2_x1[np.newaxis, :])
    inter_rect_y1 = np.maximum(b1_y1[:, np.newaxis], b2_y1[np.newaxis, :])
    inter_rect_x2 = np.minimum(b1_x2[:, np.newaxis], b2_x2[np.newaxis, :])
    inter_rect_y2 = np.minimum(b1_y2[:, np.newaxis], b2_y2[np.newaxis, :])

    # Intersection area
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1 + 1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1 + 1, 0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area[:, np.newaxis] + b2_area[np.newaxis, :] - inter_area + 1e-16)

    return iou


def evaluatePascalVOCMetrics(gt_bboxs, detected_bboxs, IOUThreshold=0.5, nr_classes=2):
    """
    Evaluates the mean average precision according to the PASCAL VOC challenge

    :param detected_bboxs: 2D list of bounding box array with shape [x_min, y_min, x_max, y_max, class_id, class_score]
    """
    statistics = []
    num_detections = 0
    for img_det_bboxes in detected_bboxs:
        num_detections += len(img_det_bboxes)

    nr_classes = nr_classes
    classes_TP = np.zeros([nr_classes, num_detections])
    classes_pred_score = -np.ones([nr_classes, num_detections])  # -1 corresponds to detection not belonging to that class
    classes_NP = np.zeros([nr_classes]).ravel()

    i_dets = 0
    for img_gt_bboxes, img_det_bboxes in zip(gt_bboxs, detected_bboxs):
        if len(img_gt_bboxes) == 0 and len(img_det_bboxes) == 0:
            continue
        det_bboxs_array = np.array(img_det_bboxes)
        gt_bboxs_array = np.array(img_gt_bboxes)
        num_dets = det_bboxs_array.shape[0]

        if gt_bboxs_array.shape[0] == 0:
            predicted_class = det_bboxs_array[:, 4].astype(np.int)
            classes_pred_score[predicted_class, i_dets + np.arange(num_dets)] = det_bboxs_array[:, 5]
            i_dets += num_dets
            continue

        gt_class = gt_bboxs_array[:, 4].astype(np.int)
        np.add.at(classes_NP, gt_class, 1)

        if det_bboxs_array.shape[0] == 0:
            continue

        if np.unique(gt_bboxs_array, axis=0).shape[0] != gt_bboxs_array.shape[0]:
            print('#######')
            print('There are duplicates in the ground truth bounding boxes')
            # raise ValueError('There are duplicates in the ground truth bounding boxes')
        if np.unique(det_bboxs_array, axis=0).shape[0] != num_dets:
            raise ValueError('There are duplicates in the detected bounding boxes')

        # Add prediction score
        predicted_class = det_bboxs_array[:, 4].astype(np.int)
        classes_pred_score[predicted_class, i_dets + np.arange(num_dets)] = det_bboxs_array[:, 5]

        true_positive_bool = computeTruePositives(gt_bboxs_array, det_bboxs_array, IOUThreshold)

        predicted_class = predicted_class[true_positive_bool]
        correct_detection = np.arange(num_dets)[true_positive_bool]
        classes_TP[predicted_class, i_dets + correct_detection] = 1

        i_dets += num_dets

    for class_id in range(nr_classes):
        single_class_TP = classes_TP[class_id, :]
        single_classes_pred_score = classes_pred_score[class_id, :]
        single_class_NP = classes_NP[class_id]

        # Get valid detection corresponding to the class_id
        valid_class_dets = single_classes_pred_score != -1
        single_class_TP = single_class_TP[valid_class_dets]
        single_classes_pred_score = single_classes_pred_score[valid_class_dets]

        # Sort according prediction score
        score_sorted_idx = np.argsort(-single_classes_pred_score)
        single_class_TP = single_class_TP[score_sorted_idx]
        single_class_FP = 1 - single_class_TP

        acc_FP = np.cumsum(single_class_FP)
        acc_TP = np.cumsum(single_class_TP)
        rec = acc_TP / single_class_NP
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        [ap, mpre, mrec, ii] = CalculateAveragePrecision(rec, prec)

        # add class result in the dictionary to be returned
        statistics.append({
            'class': class_id,
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': single_class_NP,
            'total TP': np.sum(single_class_TP),
            'total FP': np.sum(single_class_FP)
        })

    return statistics


def computeTruePositives(gt_bboxs_array, det_bboxs_array, IOUThreshold):
    """Computes the true positives according the iou and class prediction"""
    num_dets = det_bboxs_array.shape[0]
    num_gts = gt_bboxs_array.shape[0]
    gt_class = gt_bboxs_array[:, 4].astype(np.int)
    predicted_class = det_bboxs_array[:, 4].astype(np.int)

    # Compute IoU for each combination. iou_matrix shape: [n_detections, n_ground_truth_bboxs]
    iou_matrix = bbox_iou(det_bboxs_array[:, :4], gt_bboxs_array[:, :4])

    # Only consider the iou's corresponding to the predicted class
    same_class_matrix = np.equal(predicted_class[:, np.newaxis], gt_class[np.newaxis, :])
    iou_matrix = iou_matrix * same_class_matrix

    # Check if iou is above threshold and extract bool for detection with highest iou with gt
    iou_above_threshold = iou_matrix >= IOUThreshold
    highest_iou_gt_idx = np.argmax(iou_matrix, axis=1)
    det_iou_above_threshold = iou_above_threshold[np.arange(num_dets), highest_iou_gt_idx]

    # Check if prediction classify the correct class. Is required if all iou are zero
    same_class_bool = np.equal(gt_bboxs_array[highest_iou_gt_idx, 4], det_bboxs_array[:, 4])

    # Combine both criteria
    det_iou_above_threshold = np.logical_and(det_iou_above_threshold, same_class_bool)

    # Check for highest classification score
    score_matrix = np.zeros([num_dets, num_gts])
    score_matrix[np.arange(num_dets), highest_iou_gt_idx] = det_bboxs_array[:, 5]

    score_matrix = score_matrix * iou_above_threshold * same_class_bool[:, np.newaxis]

    highest_score_det_idx = np.argmax(score_matrix, axis=0)
    highest_det_score_bool = np.equal(np.arange(num_dets), highest_score_det_idx[highest_iou_gt_idx])

    true_positive_bool = np.logical_and(det_iou_above_threshold, highest_det_score_bool)

    return true_positive_bool


def CalculateAveragePrecision(rec, prec):
    """Calculates the area under the precision-recall curve with some simplifications according to VOC Pascal"""
    mrec_array = np.zeros([rec.shape[0] + 2])
    mrec_array[1:-1] = rec
    mrec_array[-1] = 1
    mpre_array = np.zeros([prec.shape[0] + 2])
    mpre_array[1:-1] = prec

    # Compute the changes from the left to the right of precision step function
    mpre_array = np.maximum.accumulate(mpre_array[::-1])[::-1]

    change_recall_idx = (np.not_equal(mrec_array[1:] - mrec_array[:-1], 0)).nonzero()[0] + 1

    ap = np.sum((mrec_array[change_recall_idx] - mrec_array[change_recall_idx-1]) * mpre_array[change_recall_idx])

    return [ap, mpre_array[0:-1], mrec_array[0:-1], change_recall_idx]


def createRandomBBox(img_height):
    gt_bbox_min = np.random.random_integers(0, img_height - 20, size=[2])
    gt_bbox_xmax = np.random.random_integers(gt_bbox_min[0], img_height, size=[1])
    gt_bbox_ymax = np.random.random_integers(gt_bbox_min[1], img_height, size=[1])

    return np.concatenate([gt_bbox_min, gt_bbox_xmax, gt_bbox_ymax])
