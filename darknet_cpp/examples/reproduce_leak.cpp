/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Network construction and destruction leak test
 *               This example will construct and destruct a network
 *               in a loop. Meanwhile, RAM and CUDA memory can be monitored
 *               to see if memory usage is stable
 */

#include "darknet.hpp"

#include <iostream>
#include <string>
#include <chrono>

#define DETECTION_THRESHOLD         0.24
#define DETECTION_HIER_THRESHOLD    0.5
#define NMS_THRESHOLD               0.4

int main(int argc, char *argv[])
{
    Darknet::Detector* detector;
    std::vector<Darknet::Detection> detections;

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_names_file> <input_cfg_file> <input_weights_file>" << std::endl;
        return -1;
    }

    std::string input_names_file(argv[1]);
    std::string input_cfg_file(argv[2]);
    std::string input_weights_file(argv[3]);

    for (int i=0; i<100; i++) {
        detector = new Darknet::Detector();

        if (!detector->setup(input_names_file,
                            input_cfg_file,
                            input_weights_file,
                            NMS_THRESHOLD,
                            DETECTION_THRESHOLD,
                            DETECTION_HIER_THRESHOLD)) {
            std::cerr << "Setup failed" << std::endl;
            return -1;
        }

        //std::this_thread::sleep_for(std::chrono::seconds(10));

        delete detector;
    }

    return 0;
}
