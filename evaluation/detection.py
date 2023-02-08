import cv2
import torch
import onnxruntime
import numpy as np
import pandas as pd
import albumentations as A

from torchvision import transforms, ops
from albumentations.pytorch import ToTensorV2

from utils.conversions import scale_bboxes
from utils.manipulations import get_cutout

def detect(img_path: str, yolo_path: str, resnet_path: str):
    """Load an image, detect individual plants and label them as
    healthy or wilted.

    :param str img_path: path to image
    :param str yolo_path: path to yolo weights
    :param str resnet_path: path to resnet weights
    :returns: tuple of recent image and dict of bounding boxes and
    their predictions

    """
    img = cv2.imread(img_path)

    # Get bounding boxes from object detection model
    box_coords = get_boxes(yolo_path, img.copy())

    box_coords.sort_values(by=['xmin'], ignore_index=True, inplace=True)

    predictions = []
    for _, row in box_coords.iterrows():
        xmin, xmax = int(row['xmin']), int(row['xmax'])
        ymin, ymax = int(row['ymin']), int(row['ymax'])

        # Get tensor of ROI in BGR
        cropped_image = get_cutout(img.copy(), xmin, xmax, ymin, ymax)

        # Classify ROI in RGB
        predictions.append(classify(resnet_path, cropped_image[..., ::-1]))

    # Gather top class and confidence values
    cls = []
    cls_conf = []
    for pred in predictions:
        ans, index = torch.topk(pred, 1)
        cls.append(index.int().item())
        cls_conf.append(round(ans.double().item(), 6))

    # Add predicted classes and confidence values to pandas dataframe
    box_coords['cls'] = cls
    box_coords['cls_conf'] = cls_conf

    return box_coords


def classify(resnet_path, img):
    """Classify img with object classification model.

    :param model: object classification model
    :param img: opencv2 image object in RGB
    :returns: tensor of class predictions
    """

    # Transform image for ResNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224))
    ])

    img = transform(img.copy())
    batch = img.unsqueeze(0)

    # Do inference
    session = onnxruntime.InferenceSession(resnet_path)
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: batch.numpy()}
    out = torch.tensor(np.array(session.run(outname, inp)))[0]

    # Apply softmax to get percentage confidence of classes
    out = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return out


def apply_nms(predictions,
              confidence_threshold: float = 0.3,
              nms_threshold: float = 0.65):
    """Apply Non Maximum Suppression to a list of bboxes.

    :param predictions List[Tensor[N, 7]]: predicted bboxes
    :param confidence_threshold float: discard all bboxes with lower
        confidence
    :param nms_threshold float: discard all overlapping bboxes with
        higher IoU
    :returns List[Tensor[N, 7]]: filtered bboxes
    """
    preds_nms = []
    for pred in predictions:
        pred = pred[pred[:, 6] > confidence_threshold]

        nms_idx = ops.batched_nms(
            boxes=pred[:, 1:5],
            scores=pred[:, 6],
            idxs=pred[:, 5],
            iou_threshold=nms_threshold,
        )
        preds_nms.append(pred[nms_idx])

    return preds_nms


def get_boxes(yolo_path, image):
    """Run object detection model on an image and get the bounding box
    coordinates of all matches.

    :param model: path to onnx object detection model (YOLO)
    :param img: opencv2 image object
    :returns: pandas dataframe of matches
    """
    # Convert from BGR to RGB
    img = image[..., ::-1].copy()

    resized_hw = (640, 640)
    original_hw = (image.shape[0], image.shape[1])

    transform = [
        A.LongestMaxSize(max(resized_hw)),
        A.PadIfNeeded(
            resized_hw[0],
            resized_hw[1],
            border_mode=0,
            value=(114, 114, 114),
        ),
        A.ToFloat(max_value=255),
        ToTensorV2(transpose_mask=True),
    ]

    # Pad (letterbox) and transform image to correct dims
    transform = A.Compose(transform)
    img = transform(image=img)

    # Add batch dimension
    img['image'] = img['image'].unsqueeze(0)

    # Do inference
    session = onnxruntime.InferenceSession(yolo_path)
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: img['image'].numpy()}
    out = torch.tensor(np.array(session.run(outname, inp)))[0]

    # Apply NMS to results
    preds_nms = apply_nms([out])[0]

    # Convert boxes from resized img to original img
    xyxy_boxes = preds_nms[:, [1, 2, 3, 4]]  # xmin, ymin, xmax, ymax
    bboxes = scale_bboxes(xyxy_boxes, resized_hw, original_hw).int().numpy()

    # Construct DataFrame with bboxes and their confidence
    box_coords = pd.DataFrame(np.c_[bboxes, preds_nms[:, 6]])
    box_coords.columns = ['xmin', 'ymin', 'xmax', 'ymax', 'box_conf']

    return box_coords
