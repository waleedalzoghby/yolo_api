/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Jetson detection and restreaming demo
 */

#include "darknet.hpp"

#include "opencv2/core/core.hpp"
#include <opencv2/imgproc.hpp>
#include <string>
#include <chrono>
#include <thread>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define DETECTION_THRESHOLD         0.24
#define DETECTION_HIER_THRESHOLD    0.5
#define NMS_THRESHOLD               0.4
#define TARGET_FPS                  30

#define GST_CAPTURE_STRING          "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, " \
                                    "framerate=(fraction)" STR(TARGET_FPS) "/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
#define GST_OUTPUT_STRING_START     "appsrc ! videoconvert ! video/x-raw, format=(string)BGRx ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420 ! " \
                                    "omxh264enc ! video/x-h264, stream-format=byte-stream ! rtph264pay ! udpsink host="
#define GST_OUTPUT_STRING_END       " sync=false async=false "

static Darknet::Detector g_detector;
static std::vector<float> g_blob;
static bool g_new_detections = false;
static std::vector<std::string> g_label_names;

static bool detect_in_image(void)
{
    if (!g_detector.predict(g_blob)) {
        std::cerr << "Failed to run detector" << std::endl;
        return false;
    }

    g_new_detections = true;
    return true;
}

static void print_stats(std::vector<Darknet::Detection> detections)
{
    static auto prevTime = std::chrono::system_clock::now();

    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> period = (now - prevTime);
    prevTime = now;
    std::cout << "FPS: " << 1 / period.count();
    std::cout << " Labels: ";

    for (auto detection : detections) {
        std::string label;
        if (g_label_names.size() > static_cast<size_t>(detection.label_index))
            label = g_label_names[detection.label_index];
        else
            label = std::to_string(detection.label_index);
        std::cout << label << ", " << detection.probability << "; ";
    }

    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    cv::Mat image;
    std::vector<float> blob;
    Darknet::PreprocessCv pre;
    std::vector<Darknet::Detection> latest_detections;
    std::vector<Darknet::Detection> latest_filtered_detections;
    std::thread detector_thread;
    std::vector<int> detection_filter( {0} );

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input_names_file> <input_cfg_file> <input_weights_file> <ip_addr_destination>" << std::endl;
        return -1;
    }

    std::string input_names_file(argv[1]);
    std::string input_cfg_file(argv[2]);
    std::string input_weights_file(argv[3]);
    std::string ip_addr_dest(argv[4]);

    if (!cap.open(GST_CAPTURE_STRING)) {
        std::cerr << "Could not open video input stream" << std::endl;
        return -1;
    }

    if (!cap.read(image)) {
        std::cerr << "Failed to capture initial camera image" << std::endl;
        return -1;
    }

    if (!writer.open(GST_OUTPUT_STRING_START + ip_addr_dest + GST_OUTPUT_STRING_END, 0, TARGET_FPS, image.size())) {
        std::cerr << "Could not open video output stream" << std::endl;
        return -1;
    }

    // read label names
    if (!Darknet::read_text_file(g_label_names, input_names_file)) {
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
        return -1;
    }

    // setup preprocessor
    pre.setup(g_detector.get_width(), g_detector.get_height());

    while(1) {

        if (!cap.read(image)) {
            std::cerr << "Video capture read failed/EoF" << std::endl;
            return -1;
        }

        // preprocess image
        if (!pre.run(image, blob)) {
            std::cerr << "Failed to preprocess image" << std::endl;
            return -1;
        }

        // if detector thread is finished, start it again
        if (g_new_detections) {
            g_new_detections = false;

            if (detector_thread.joinable())
                detector_thread.join();

            // post process and get detections
            if (!g_detector.post_process(image.cols, image.rows)) {
                std::cerr << "Failed to post process" << std::endl;
                return 1;
            }

            latest_detections = g_detector.get_detections();
            g_blob = blob;
            detector_thread = std::thread(detect_in_image);

            // filter detections
            Darknet::filter_detections(latest_detections, latest_filtered_detections, detection_filter);
        }

        // overlay detections
        Darknet::image_overlay(latest_filtered_detections, image, g_label_names);
        print_stats(latest_filtered_detections);

        // restream
        writer.write(image);
    }

    return 0;
}
