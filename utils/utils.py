import torch


def cropBboxToFrame(bbox, image_shape):
    """Checks if bounding boxes are inside frame. If not crop to border"""
    array_height = torch.ones_like(bbox[:, 1]) * image_shape[0] - 1
    array_width = torch.ones_like(bbox[:, 2]) * image_shape[1] - 1

    bbox[:, :2] = torch.max(bbox[:, :2], torch.zeros_like(bbox[:, :2]))
    bbox[:, 0] = torch.min(bbox[:, 0], array_height)
    bbox[:, 1] = torch.min(bbox[:, 1], array_width)

    bbox[:, 2] = torch.min(bbox[:, 2], array_height - bbox[:, 0])
    bbox[:, 3] = torch.min(bbox[:, 3], array_width - bbox[:, 1])

    return bbox


def generateMask(height, width):
    """Generates a mask for covering up the bonnet of the mvsec samples"""
    mask = torch.ones([height, width])
    mask_height = 57
    bottom_width = 250
    bottom_height = 25
    top_width = 165

    mask[-bottom_height:, :] = 0
    slope = float(top_width - bottom_width) / float(mask_height - bottom_height)

    for i in range(0, mask_height - bottom_height):
        mask_out_width = int(slope * i) + bottom_width
        border = (width - mask_out_width) // 2
        mask[-(bottom_height + i), border:-border] = 0

    return mask
