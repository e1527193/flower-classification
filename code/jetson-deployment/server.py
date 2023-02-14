import atexit
import jetson.utils

from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler

from multiprocessing import Manager

from code.evaluation.detection import detect

app = Flask(__name__)
scheduler = BackgroundScheduler(daemon=True)
manager = Manager()
pred = manager.dict()


@scheduler.task('interval', id='get_pred', minutes=30, misfire_grace_time=900)
def get_pred():
    img = take_image('./current_image.jpg')
    print('Job 1 executed')


def take_image(img_path: str):
    """Take an image with the webcam and save it to the specified
    path.

    :param str img_path: path image should be saved to
    :returns: captured image

    """
    input = jetson.utils.videoSource('csi://0')
    output = jetson.utils.videoOutput(img_path)
    img = input.Capture()
    output.Render(img)
    return img



@app.route('/')
def index():
    # TODO: call script and save initial image with bounding boxes
    # TODO: get predictions and output them in JSON via API
    # TODO: periodically get image from webcam and go to beginning
    # TODO: JSON format: [Nr, state (1-10), timestamp, time since below 3]
    return 'Server works'


if __name__ == '__main__':
    scheduler.add_job(func=get_pred, trigger='interval', minutes=30)
    scheduler.start()
    atexit.register(scheduler.shutdown())
    app.run()
