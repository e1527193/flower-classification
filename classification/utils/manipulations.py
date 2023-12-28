import cv2


def draw_boxes(image, bboxes):
    """Draw bounding boxes in xmin, ymin, xmax, ymax format onto
    image.

    :param image: opencv2 image object in BGR
    :param List bboxes: bounding boxes in xmin, ymin, xmax, ymax
    format
    :returns: img with bounding boxes drawn
    """
    img = image.copy()
    for idx, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # Draw bounding box and number on original image
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        img = cv2.putText(img, str(idx), (xmin + 5, ymin + 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4,
                          cv2.LINE_AA)
        img = cv2.putText(img, str(idx), (xmin + 5, ymin + 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                          cv2.LINE_AA)
    return img


def get_cutout(img, xmin, xmax, ymin, ymax):
    """Cut out a bounding box from an image and transform it for
    object classification model.

    :param img: opencv2 image object in BGR
    :param int xmin: start of bounding box on x axis
    :param int xmax: end of bounding box on x axis
    :param int ymin: start of bounding box on y axis
    :param int ymax: end of bounding box on y axis
    :returns: tensor of cropped image in BGR
    """
    cropped_image = img[ymin:ymax, xmin:xmax]
    return cropped_image
