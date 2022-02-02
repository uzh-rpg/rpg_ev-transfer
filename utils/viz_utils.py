import cv2
import torch
import numpy as np
import torchvision.utils


def createRGBGrid(tensor_list, nrow):
    """Creates a grid of rgb values based on the tensor stored in tensor_list"""
    vis_tensor_list = []
    for tensor in tensor_list:
        vis_tensor_list.append(visualizeTensors(tensor))

    return torchvision.utils.make_grid(torch.cat(vis_tensor_list, dim=0), nrow=nrow)


def createRGBImage(tensor):
    """Creates a grid of rgb values based on the tensor stored in tensor_list"""
    if tensor.shape[1] == 3:
        return tensor
    elif tensor.shape[1] == 1:
        return tensor.expand(-1, 3, -1, -1)
    elif tensor.shape[1] == 2:
        return visualizeHistogram(tensor)
    elif tensor.shape[1] > 3:
        return visualizeVoxelGrid(tensor)


def visualizeTensors(tensor):
    """Creates a rgb image of the given tensor. Can be event histogram, event voxel grid, grayscale and rgb."""
    if tensor.shape[1] == 3:
        return tensor
    elif tensor.shape[1] == 1:
        return tensor.expand(-1, 3, -1, -1)
    elif tensor.shape[1] == 2:
        return visualizeHistogram(tensor)
    elif tensor.shape[1] > 3:
        return visualizeVoxelGrid(tensor)


def visualizeHistogram(histogram):
    """Visualizes the input histogram"""
    batch, _, height, width = histogram.shape
    torch_image = torch.zeros([batch, 1, height, width], device=histogram.device)

    return torch.cat([histogram.clamp(0, 1), torch_image], dim=1)


def visualizeVoxelGrid(voxel_grid):
    """Visualizes the input histogram"""
    batch, nr_channels, height, width = voxel_grid.shape
    pos_events_idx = nr_channels // 2
    temporal_scaling = torch.arange(start=1, end=pos_events_idx+1, dtype=voxel_grid.dtype,
                                    device=voxel_grid.device)[None, :, None, None] / pos_events_idx
    pos_voxel_grid = voxel_grid[:, :pos_events_idx] * temporal_scaling
    neg_voxel_grid = voxel_grid[:, pos_events_idx:] * temporal_scaling

    torch_image = torch.zeros([batch, 1, height, width], device=voxel_grid.device)
    pos_image = torch.sum(pos_voxel_grid, dim=1, keepdim=True)
    neg_image = torch.sum(neg_voxel_grid, dim=1, keepdim=True)

    return torch.cat([pos_image.clamp(0, 1), neg_image.clamp(0, 1), torch_image], dim=1)


def visualizeConfusionMatrix(confusion_matrix, path_name=None):
    """
    Visualizes the confustion matrix using matplotlib.

    :param confusion_matrix: NxN numpy array
    :param path_name: if no path name is given, just an image is returned
    """
    import matplotlib.pyplot as plt
    nr_classes = confusion_matrix.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.matshow(confusion_matrix)
    ax.plot([-0.5, nr_classes - 0.5], [-0.5, nr_classes - 0.5], '-', color='grey')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Predicted')

    if path_name is None:
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    else:
        fig.savefig(path_name)
        plt.close()


def drawBoundingBoxes(np_image, bounding_boxes, class_name=None, ground_truth=True, rescale_image=False):
    """
    Draws the bounding boxes in the image

    :param np_image: [H, W, C]
    :param bounding_boxes: list of bounding boxes with shape [x, y, width, height].
    :param class_name: string
    """
    np_image = np_image.astype(np.float)
    resize_scale = 1.5
    if rescale_image:
        bounding_boxes[:, :4] = (bounding_boxes.astype(np.float)[:, :4] * resize_scale)
        new_dim = np.array(np_image.shape[:2], dtype=np.float) * resize_scale
        np_image = cv2.resize(np_image, tuple(new_dim.astype(int)[::-1]), interpolation=cv2.INTER_NEAREST)

    for i, bounding_box in enumerate(bounding_boxes):
        if bounding_box.sum() == 0:
            break
        if class_name is None:
            np_image = drawBoundingBox(np_image, bounding_box, ground_truth=ground_truth)
        else:
            np_image = drawBoundingBox(np_image, bounding_box, class_name[i], ground_truth)

    return np_image


def drawBoundingBox(np_image, bounding_box, class_name=None, ground_truth=False):
    """
    Draws a bounding box in the image.

    :param np_image: [H, W, C]
    :param bounding_box: [x, y, width, height].
    :param class_name: string
    """
    if ground_truth:
        bbox_color = np.array([0, 1, 1])
    else:
        bbox_color = np.array([1, 0, 1])
    height, width = bounding_box[2:4]

    np_image[bounding_box[0], bounding_box[1]:(bounding_box[1] + width)] = bbox_color
    np_image[bounding_box[0]:(bounding_box[0] + height), (bounding_box[1] + width)] = bbox_color
    np_image[(bounding_box[0] + height), bounding_box[1]:(bounding_box[1] + width)] = bbox_color
    np_image[bounding_box[0]:(bounding_box[0] + height), bounding_box[1]] = bbox_color

    if class_name is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)
        font_scale = 0.5
        thickness = 1
        bottom_left = tuple(((bounding_box[[1, 0]] + np.array([+1, height - 2]))).astype(int))

        # Draw Box
        (text_width, text_height) = cv2.getTextSize(class_name, font, fontScale=font_scale, thickness=thickness)[0]
        box_coords = ((bottom_left[0], bottom_left[1] + 2),
                      (bottom_left[0] + text_width + 2, bottom_left[1] - text_height - 2 + 2))
        color_format = (int(bbox_color[0]), int(bbox_color[1]), int(bbox_color[2]))
        # np_image = cv2.UMat(np_image)
        np_image = cv2.UMat(np_image).get()
        cv2.rectangle(np_image, box_coords[0], box_coords[1], color_format, cv2.FILLED)

        cv2.putText(np_image, class_name, bottom_left, font, font_scale, font_color, thickness, cv2.LINE_AA)

    return np_image


def visualizeFlow(tensor_flow_map):
    """
    Visualizes the direction flow based on the HSV model
    """
    np_flow_map = tensor_flow_map.cpu().detach().numpy()
    batch_s, channel, height, width = np_flow_map.shape
    viz_array = np.zeros([batch_s, height, width, 3], dtype=np.uint8)
    hsv = np.zeros([height, width, 3], dtype=np.uint8)

    for i, sample_flow_map in enumerate(np_flow_map):
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(sample_flow_map[0, :, :], sample_flow_map[1, :, :])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        viz_array[i] = bgr

    return torch.from_numpy(viz_array.transpose([0, 3, 1, 2]) / 255.).to(tensor_flow_map.device)
