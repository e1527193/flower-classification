import argparse
import cv2
import torch
from torchvision import transforms


def load_models(yolo_path: str, resnet_path: str):
    """Load the models for two-stage classification.

    :param yolo_path: path to yolo weights
    :param resnet_path: path to resnet weights
    :returns: tuple of models

    """
    first_stage = torch.hub.load("WongKinYiu/yolov7",
                                 "custom",
                                 yolo_path,
                                 trust_repo=True)
    second_stage = torch.load(resnet_path)
    return (first_stage, second_stage)


def detect(img_path: str, yolo_path: str, resnet_path: str):
    """Load an image, detect individual plants and label them as
    healthy or wilted.

    :param str img_path: path to image
    :param yolo_path: path to yolo weights
    :param resnet_path: path to resnet weights
    :returns: tuple of recent image and dict of bounding boxes and
    their predictions

    """
    img = cv2.imread(img_path)
    original = img.copy()
    (first_stage, second_stage) = load_models(yolo_path, resnet_path)

    # Get bounding boxes from object detection model
    box_coords = get_boxes(first_stage, img)
    box_coords.sort_values(by=['xmin'], ignore_index=True, inplace=True)

    predictions = {}
    for idx, row in box_coords.iterrows():
        xmin, xmax = int(row['xmin']), int(row['xmax'])
        ymin, ymax = int(row['ymin']), int(row['ymax'])

        # Get tensor of ROI in BGR
        cropped_image = get_cutout(img.copy(), xmin, xmax, ymin, ymax)

        # Classify ROI in RGB
        predictions[idx] = classify(second_stage, cropped_image[..., ::-1])

        # Draw bounding box and number on original image
        original = cv2.rectangle(original, (xmin, ymin), (xmax, ymax),
                                 (0, 255, 0), 2)
        original = cv2.putText(original, str(idx), (xmin + 5, ymin + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4,
                               cv2.LINE_AA)
        original = cv2.putText(original, str(idx), (xmin + 5, ymin + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255),
                               2, cv2.LINE_AA)

    return (original, predictions)


def get_boxes(model, img):
    """Run object detection model on an image and get the bounding box
    coordinates of all matches.

    :param model: object detection model (YOLO)
    :param img: opencv2 image object
    :returns: pandas dataframe of matches

    """
    box_coords = model(img[..., ::-1], size=640)
    return box_coords.pandas().xyxy[0]


def classify(model, img):
    """Classify img with object classification model.

    :param model: object classification model
    :param img: opencv2 image object in RGB
    :returns: tensor of class predictions

    """
    # Transform image for ResNet
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = data_transforms(img.copy())
    out = model(img.unsqueeze(0))
    # Apply softmax to get percentage confidence of classes
    out = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return out


def get_cutout(img, xmin, xmax, ymin, ymax):
    """Cut out a bounding box from an image and transform it for
    object classification model.

    :param img: opencv2 image object in BGR
    :param xmin: start of bounding box on x axis
    :param xmax: end of bounding box on x axis
    :param ymin: start of bounding box on y axis
    :param ymax: end of bounding box on y axis
    :returns: tensor of cropped image in BGR

    """
    cropped_image = img[ymin:ymax, xmin:xmax]
    return cropped_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='image file or webcam')
    opt = parser.parse_args()

    if opt.source:
        with torch.no_grad():
            detect(opt.source, 'runs/train/yolov7-custom7/weights/best.pt',
                   'resnet.pt')
