import argparse
import logging
import atexit
import os

import cv2
import json
import base64
import datetime
from dateutil.parser import parse
from copy import deepcopy

from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler

from multiprocessing import Manager

from evaluation.detection import detect
from utils.manipulations import draw_boxes

app = Flask(__name__)
scheduler = BackgroundScheduler(daemon=True)
manager = Manager()
pred = manager.dict()


def get_pred():
    tmp = deepcopy(pred)
    take_image()
    logging.debug('Starting image classification')
    preds = detect('current.jpg', '../weights/yolo.onnx',
                   '../weights/resnet.onnx')
    logging.debug('Finished image classification: %s', preds)
    logging.debug('Reading current.jpg for drawing bounding boxes')
    current = cv2.imread('current.jpg')
    logging.debug('Drawing bounding boxes on current.jpg')
    bbox_img = draw_boxes(
        current, preds[['xmin', 'ymin', 'xmax',
                        'ymax']].itertuples(index=False, name=None))
    logging.debug(
        'Finished drawing bounding boxes. Saving to current_bbox.jpg ...')
    cv2.imwrite('current_bbox.jpg', bbox_img)

    # Clear superfluous bboxes if less detected
    # if len(preds) < len(pred):
    #     logging.debug(
    #         'Current round contains less bboxes than previous round: old: %s\nnew: %s',
    #         json.dumps(preds.copy()), json.dumps(pred.copy()))
    #     for key in pred:
    #         if key not in preds:
    #             pred.pop(key)

    pred.clear()
    for idx, row in preds.iterrows():
        new = []
        state = int(round(float(row['cls_conf']) / 10, 0))
        new.append(state)
        new.append(datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            pred[str(idx)] = tmp[str(idx)]
        except KeyError:
            pass

        try:
            if tmp[idx][2] == -1 and state > 3:
                logging.debug(
                    'State is worse than 3 for the first time: %s ... populating third field',
                    str(state))
                new.append(
                    datetime.datetime.now(datetime.timezone.utc).isoformat())
            elif tmp[idx][2] != -1 and state <= 3:
                logging.debug(
                    'State changed from worse than 3 to better than 3')
                new.append(-1)
            elif tmp[idx][2] != -1 and state > 3:
                logging.debug('State is still worse than 3')
                new.append(tmp[idx][2])
        except:
            logging.debug('Third key does not exist')
            if state > 3:
                logging.debug(
                    'State is worse than 3. Populating third field with timestamp'
                )
                new.append(
                    datetime.datetime.now(datetime.timezone.utc).isoformat())
            else:
                logging.debug(
                    'State is better than 3. Populating third field with -1')
                new.append(-1)

        pred[idx] = new

    bbox_img_b64 = base64.b64encode(cv2.imencode('.jpg', bbox_img)[1]).decode()
    pred['image'] = bbox_img_b64
    logging.debug('Saved bbox_img to json')


def take_image():
    """Take an image with the webcam and save it to the specified
    path.

    :param str img_path: path image should be saved to
    :returns: captured image
    """
    capture_path = os.path.join('.', 'image-capture', 'capture')
    if os.path.isfile(capture_path) and os.access(capture_path, os.X_OK):
        logging.debug('Starting image capture')
        os.system('./image-capture/capture')
        logging.debug('Finished image capture')
    else:
        logging.critical(
            'Image capture binary is not at path %s. Shutting down server...',
            capture_path)
        app.terminate()


@app.route('/')
def index():
    copy = pred.copy()
    for key, value in copy.items():
        if key == 'image':
            continue
        if value[2] != -1:
            logging.debug('value: %s', value)
            # Calc difference to now
            time_below_thresh = parse(value[2])
            now = datetime.datetime.now(datetime.timezone.utc)
            delta = now - time_below_thresh
            value[2] = str(delta)
    return json.dumps(copy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

    scheduler.add_job(func=get_pred,
                      trigger='interval',
                      minutes=2,
                      next_run_time=datetime.datetime.now())
    scheduler.start()
    atexit.register(scheduler.shutdown)

    app.run()
