/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Darknet C++ detection demo
 */

#include "darknet.hpp"

#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <chrono>

#define DETECTION_THRESHOLD         0.24
#define DETECTION_HIER_THRESHOLD    0.5
#define NMS_THRESHOLD               0.4

int main(int argc, char *argv[])
{
    cv::VideoCapture cap;
    cv::Mat image;
    std::vector<float> blob;
    Darknet::PreprocessCv pre;
    Darknet::Detector detector;
    std::vector<Darknet::Detection> detections;

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_names_file> <input_cfg_file> <input_weights_file> [<videofile>]" << std::endl;
        return -1;
    }

    std::string input_names_file(argv[1]);
    std::string input_cfg_file(argv[2]);
    std::string input_weights_file(argv[3]);

    if (argc == 5) {
        std::string videofile(argv[4]);
        if (!cap.open(videofile)) {
            std::cerr << "Could not open video file" << std::endl;
            return -1;
        }
    } else if (!cap.open(0)) {
        std::cerr << "Could not open video input stream" << std::endl;
        return -1;
    }

    if (!detector.setup(input_names_file,
                        input_cfg_file,
                        input_weights_file,
                        NMS_THRESHOLD,
                        DETECTION_THRESHOLD,
                        DETECTION_HIER_THRESHOLD)) {
        std::cerr << "Setup failed" << std::endl;
        return -1;
    }

    pre.setup(detector.get_width(), detector.get_height());
    auto prevTime = std::chrono::system_clock::now();

    while(1) {

        if (!cap.read(image)) {
            std::cerr << "Video capture read failed/EoF" << std::endl;
            return -1;
        }

        // preprocess image
        if (!pre.run(image, blob)) {
            std::cerr << "Failed to convert opencv image to darknet image" << std::endl;
            return -1;
        }

        // run detector
        if (!detector.predict(blob)) {
            std::cerr << "Failed to run detector" << std::endl;
            return -1;
        }
        detector.get_detections(detections, image.cols, image.rows);

        // draw bounding boxes
        Darknet::image_overlay(detections, image);

        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> period = (now - prevTime);
        prevTime = now;
        std::cout << "FPS: " << 1 / period.count() << std::endl;

        cv::imshow("Overlay", image);
        cv::waitKey(1);

    }

    return 0;
}
