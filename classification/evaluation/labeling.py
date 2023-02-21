import logging
import argparse
import cv2
import json
import os

from utils.conversions import convert_to_yolo
from detection import detect

template = [{
    "data": {
        "image": "/data/local-files/?d=evaluation/images/0.jpg"
    },
    "predictions": [{
        "model_version":
        "one",
        "score":
        0.0,
        "result": [{
            "id": "result1",
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": 474,
            "original_height": 266,
            "image_rotation": 0,
            "confidence": 0,
            "value": {
                "rotation": 0,
                "x": 19.62,
                "y": 15.04,
                "width": 55.06,
                "height": 78.2,
                "rectanglelabels": ["Stressed"]
            }
        }]
    }]
}]


def create_json_annotation(image_path, yolo_path, resnet_path):
    """Create a JSON representation of identified bounding boxes.

    :param image_path str: path to image
    :param yolo_path str: path to YOLO model in ONNX format
    :param resnet_path str: path to ResNet model in ONNX format
    :returns Dict: bounding boxes in labelstudio JSON format
    """
    template[0]['data'][
        'image'] = "/data/local-files/?d=evaluation/" + image_path
    img = cv2.imread(image_path)
    (height, width) = img.shape[0], img.shape[1]
    bboxes = detect(image_path, yolo_path, resnet_path)
    result = template[0]['predictions'][0]['result']

    results = []
    for idx, row in bboxes.iterrows():
        modified = convert_to_yolo(row, width, height)
        json_result = {}
        json_result['id'] = 'result' + str(idx + 1)
        json_result['type'] = 'rectanglelabels'
        json_result['from_name'] = 'label'
        json_result['to_name'] = 'image'
        json_result['original_width'] = width
        json_result['original_height'] = height
        json_result['image_rotation'] = 0
        json_result['value'] = {}
        json_result['value']['rotation'] = 0
        json_result['value']['x'] = modified['xmin%']
        json_result['value']['y'] = modified['ymin%']
        json_result['value']['width'] = modified['width%']
        json_result['value']['height'] = modified['height%']
        if modified['cls'] == 0:
            json_result['value']['rectanglelabels'] = ['Healthy']
        else:
            json_result['value']['rectanglelabels'] = ['Stressed']
        results.append(json_result)

    template[0]['predictions'][0]['result'] = results
    return template


def write_labels_to_disk(image_dir, output_dir, yolo_path, resnet_path):
    """Read images from disk, classify them and output bounding boxes
    in labelstudio JSON format.

    :param image_dir str: directory containing images to label
    :param output_dir str: directory to save JSON files to
    :param yolo_path str: path to YOLO model in ONNX format
    :param resnet_path str: path to ResNet model in ONNX format
    :returns: None
    """
    image_dir = os.path.join(image_dir, '')
    for file in os.listdir(image_dir):
        filename = os.fsdecode(file)
        filename_wo_ext = os.path.splitext(filename)[0]
        rel_output_path = os.path.join(output_dir, filename_wo_ext + '.json')
        json_data = create_json_annotation(image_dir + filename, yolo_path,
                                           resnet_path)
        os.makedirs(os.path.dirname(os.path.join(output_dir, filename)),
                    exist_ok=True)
        logging.info('Writing json file for %s', filename)
        with open(rel_output_path, 'w') as f:
            json.dump(json_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        type=str,
                        help='source folder with images',
                        required=True)
    parser.add_argument('--output',
                        type=str,
                        help='output folder for json files',
                        required=True)
    parser.add_argument('--yolo',
                       type=str,
                       help='path to YOLO model in ONNX format',
                       required=True)
    parser.add_argument('--resnet',
                       type=str,
                       help='path to ResNet model in ONNX format',
                       required=True)
    parser.add_argument(
        '--log',
        type=str,
        help='log level (debug, info, warning, error, critical)',
        default='warning')
    opt = parser.parse_args()
    numeric_level = getattr(logging, opt.log.upper(), None)
    logging.basicConfig(format='%(levelname)s::%(asctime)s::%(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%S',
                        level=numeric_level)
    write_labels_to_disk(opt.source, opt.output, opt.yolo, opt.resnet)
