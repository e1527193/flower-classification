def convert_to_yolo(bbox, width, height):
    modified = bbox.copy()
    modified['xmin%'] = round(bbox['xmin'] / width * 100, 2)
    modified['ymin%'] = round(bbox['ymin'] / height * 100, 2)
    modified['width%'] = round((bbox['xmax'] - bbox['xmin']) / width * 100, 2)
    modified['height%'] = round((bbox['ymax'] - bbox['ymin']) / height * 100,
                                2)
    return modified


def scale_bboxes(bboxes, resized_hw, original_hw):
    """Scale bounding boxes from a padded and resized image to fit on
    original image.

    :param xyxy_boxes Tensor[N, 4]: tensor of xmin, ymin, xmax, ymax
        per bounding box
    :param resized_hw Tuple: height and width of the resized image
    :param original_hw Tuple: height and width of the original image
    :returns Tensor[N, 4]: tensor of xmin, ymin, xmax, ymax per
        bounding box
    """
    scaled_boxes = bboxes.clone()
    scale_ratio = resized_hw[0] / original_hw[0], resized_hw[1] / original_hw[1]

    # Remove padding
    pad_scale = min(scale_ratio)
    padding = (resized_hw[1] - original_hw[1] * pad_scale) / 2, (
        resized_hw[0] - original_hw[0] * pad_scale) / 2
    scaled_boxes[:, [0, 2]] -= padding[0]  # x padding
    scaled_boxes[:, [1, 3]] -= padding[1]  # y padding
    scale_ratio = (pad_scale, pad_scale)

    scaled_boxes[:, [0, 2]] /= scale_ratio[1]
    scaled_boxes[:, [1, 3]] /= scale_ratio[0]

    # Clip bounding xyxy bounding boxes to image shape (height, width)
    scaled_boxes[:, 0].clamp_(0, original_hw[1])  # xmin
    scaled_boxes[:, 1].clamp_(0, original_hw[0])  # ymin
    scaled_boxes[:, 2].clamp_(0, original_hw[1])  # xmax
    scaled_boxes[:, 3].clamp_(0, original_hw[0])  # ymax

    return scaled_boxes
