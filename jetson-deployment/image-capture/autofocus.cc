#include <cstdio>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

string img_pipeline(int capture_width, int capture_height, int framerate,
                    int flip_method) {
  return "nvarguscamerasrc num-buffers=1 ! "
         "video/x-raw(memory:NVMM),width=(int)" +
         to_string(capture_width) + ",height=(int)" +
         to_string(capture_height) + ",framerate=(fraction)" +
         to_string(framerate) +
         "/1 ! "
         "nvvidconv flip-method=" +
         to_string(flip_method) +
         " ! "
         "jpegenc quality=95 ! "
         "videoconvert ! "
         "appsink";
}

string focus_pipeline(int capture_width, int capture_height, int framerate,
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

int focus(int val) {
  int value = (val << 4) & 0x3ff0;
  int data1 = (value >> 8) & 0x3f;
  int data2 = value & 0xf0;

  string data1_s = to_string(data1);
  string data2_s = to_string(data2);

  string command = ("i2cset -y 6 0x0c " + data1_s + " " + data2_s);

  cout << command << endl;

  const char *final = command.c_str();

  int ch;
  FILE *proc = popen(final, "r");
  if (proc == NULL) {
    puts("Unable to open process");
    return (1);
  }
  while ((ch = fgetc(proc)) != EOF) {
    putchar(ch);
  }
  pclose(proc);
  cout << "Focusing done!" << endl;
  return 0;
}

Scalar sobel(Mat img) {
  Mat img_gray;
  Mat img_sobel;

  cvtColor(img, img_gray, COLOR_RGB2GRAY);
  Sobel(img_gray, img_sobel, CV_16U, 1, 1);

  return mean(img_sobel)[0];
}

Scalar laplacian(Mat img) {
  Mat img_gray;
  Mat img_laplacian;

  cvtColor(img, img_gray, COLOR_RGB2GRAY);
  Laplacian(img_gray, img_laplacian, CV_16U, 1, 1);

  return mean(img_laplacian)[0];
}

int main() {
  int capture_width = 3264;
  int capture_height = 2464;
  int framerate = 21;
  int flip_method = 2;

  string pipeline =
      focus_pipeline(capture_width, capture_height, framerate, flip_method);
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
    imwrite("test.jpg", img);
  } catch (const Exception &ex) {
    fprintf(stderr, "Exception converting image to JPG format: %s\n",
            ex.what());
  }

  // while (true) {
  //   if (!cap.read(img)) {
  //     cout << "Capture read error" << endl;
  //     break;
  //   }

  //   // imshow("CSI Camera", img);

  //   if (dec_count < 6 && focal_distance < 1000) {
  //     // Adjust focus by 10
  //     focus(focal_distance);
  //     // Calculate image clarity
  //     auto val = laplacian(img);
  //     // Find max clarity
  //     if (val[0] > max_value) {
  //       max_index = focal_distance;
  //       max_value = val[0];
  //     }

  //     // If clarity starts to decrease
  //     if (val[0] < last_value) {
  //       dec_count += 1;
  //     } else {
  //       dec_count = 0;
  //     }

  //     // Img clarity is reduced by 6 consecutive frames
  //     if (dec_count < 6) {
  //       last_value = val[0];
  //       // Increase focal distance
  //       focal_distance += 10;
  //     }
  //   } else if (!focus_finished) {
  //     // Adjust focus to best
  //     focus(max_index);
  //     focus_finished = true;
  //   }

  //   vector<int> compression_params;
  //   compression_params.push_back(IMWRITE_JPEG_QUALITY);
  //   if (focus_finished) {
  //     bool result = false;
  //     try {
  //       result = imwrite("current.jpg", img, compression_params);
  //     } catch (const Exception &ex) {
  //       fprintf(stderr, "Exception converting image to JPG format: %s\n",
  //               ex.what());
  //     }
  //     if (result) {
  //       printf("Saved JPG file.\n");
  //     } else {
  //       printf("ERROR: Can't save JPG file.\n");
  //     }
  //     break;
  //   }

  // // Stop on ESC
  // if (keycode == 27) {
  //   break;
  // }
  // else if (keycode == 13) {
  //   max_index = 10;
  //   max_value = 0;
  //   last_value = 0;
  //   dec_count = 0;
  //   focal_distance = 10;
  //   focus_finished = false;
  // }
  // }

  cap.release();
  return 0;
}
