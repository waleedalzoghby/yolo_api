/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Darknet C++ detection demo with detections
 *  running in seperate thread
 */

#include "darknet.hpp"

#include "opencv2/highgui/highgui.hpp"
#include <thread>
#include <string>
#include <chrono>

#define DETECTION_THRESHOLD         0.24
#define DETECTION_HIER_THRESHOLD    0.5
#define NMS_THRESHOLD               0.4

static Darknet::Detector g_detector;
static std::vector<float> g_blob;
static bool g_new_detections = false;

static bool detect_in_image(void)
{
    if (!g_detector.predict(g_blob)) {
        std::cerr << "Failed to run detector" << std::endl;
        return false;
    }

    g_new_detections = true;
    return true;
}

int main(int argc, char *argv[])
{
    cv::VideoCapture cap;
    cv::Mat image, image_prev;
    std::vector<std::string> label_names;
    std::vector<float> blob;
    std::vector<Darknet::Detection> detections;
    Darknet::PreprocessCv pre;
    std::thread detector_thread;

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_names_file> <input_cfg_file> <input_weights_file> [<videofile>]" << std::endl;
        return 1;
    }

    std::string input_names_file(argv[1]);
    std::string input_cfg_file(argv[2]);
    std::string input_weights_file(argv[3]);

    if (argc == 5) {
        std::string videofile(argv[4]);
        if (!cap.open(videofile)) {
            std::cerr << "Could not open video file" << std::endl;
            return 1;
        }
    } else if (!cap.open(0)) {
        std::cerr << "Could not open video input stream" << std::endl;
        return 1;
    }

    // read label names
    if (!Darknet::read_text_file(label_names, input_names_file)) {
        std::cerr << "Failed to read names file" << std::endl;
        return 1;
    }

    // setup detector
    if (!g_detector.setup(input_cfg_file,
                        input_weights_file,
                        NMS_THRESHOLD,
                        DETECTION_THRESHOLD,
                        DETECTION_HIER_THRESHOLD)) {
        std::cerr << "Setup failed" << std::endl;
        return 1;
    }

    // setup preprocessor
    pre.setup(g_detector.get_width(), g_detector.get_height());
    auto prevTime = std::chrono::system_clock::now();

    while(1) {

        if (!cap.read(image)) {
            std::cerr << "Video capture read failed/EoF" << std::endl;
            return 1;
        }

        // preprocess image
        if (!pre.run(image, blob)) {
            std::cerr << "Failed to preprocess image" << std::endl;
            return 1;
        }

        if (detector_thread.joinable())
            detector_thread.join();

        // set input data ready for inference thread
        g_blob = blob;

        if (g_new_detections) {
            g_new_detections = false;

            // post process and get detections
            if (!g_detector.post_process(image.cols, image.rows)) {
                std::cerr << "Failed to post process" << std::endl;
                return 1;
            }

            detections = g_detector.get_detections();

            // start new detection thread
            detector_thread = std::thread(detect_in_image);

            // draw bounding boxes
            Darknet::image_overlay(detections, image_prev, label_names);

            auto now = std::chrono::system_clock::now();
            std::chrono::duration<double> period = (now - prevTime);
            prevTime = now;
            std::cout << "FPS: " << 1 / period.count() << std::endl;

            cv::imshow("Overlay", image_prev);
            cv::waitKey(1);

        } else {
            detector_thread = std::thread(detect_in_image);
        }

        image_prev = image.clone();
    }

    return 0;
}
