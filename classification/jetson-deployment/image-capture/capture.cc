#include <cstdio>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

string video_pipeline(int capture_width, int capture_height, int framerate,
                      int flip_method) {
  return "nvarguscamerasrc ! "
         "video/x-raw(memory:NVMM),width=(int)" +
         to_string(capture_width) + ",height=(int)" +
         to_string(capture_height) + ",framerate=(fraction)" +
         to_string(framerate) +
         "/1 ! "
         "nvvidconv flip-method=" +
         to_string(flip_method) +
         " ! "
	 "video/x-raw,format=(string)BGRx ! "
         "videoconvert ! "
	 "video/x-raw,format=(string)BGR,width=(int)3264,height=(int)2464 ! "
         "appsink";
}

// Calls nvarguscamerasrc to take a 1s video at maximum resolution,
// flips the video and then captures one image and writes it to disk.
int main() {
  int capture_width = 3264;
  int capture_height = 2464;
  int framerate = 21;
  int flip_method = 2;

  string pipeline =
      video_pipeline(capture_width, capture_height, framerate, flip_method);
  cout << "Using pipeline: \n\t" << pipeline << endl;

  VideoCapture cap(pipeline, CAP_GSTREAMER);
  if (!cap.isOpened()) {
    cout << "Failed to open camera." << endl;
    return -1;
  }

  Mat img;

  // Take 21 images (1s) to get automatic ISO
  for (int i = 0; i < 21; i++) {
    cap.read(img);
  }

  cap.read(img);

  try {
    imwrite("current.jpg", img);
  } catch (const Exception &ex) {
    fprintf(stderr, "Exception converting image to JPG format: %s\n",
            ex.what());
  }

  cap.release();
  return 0;
}
