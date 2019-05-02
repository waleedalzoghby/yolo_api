/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Identification example
 */

#include "darknet.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

int main(int argc, char *argv[])
{
    cv::Mat image1, image2;
    std::vector<float> blob1, blob2;
    std::vector<float> id1, id2;
    Darknet::PreprocessCv pre;
    Darknet::Identifier identifier;

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input_cfg_file> <input_weights_file> <image1> <image2>" << std::endl;
        return 1;
    }

    std::string input_cfg_file(argv[1]);
    std::string input_weights_file(argv[2]);
    std::string image1_file(argv[3]);
    std::string image2_file(argv[4]);

    // read images
    image1 = cv::imread(image1_file);
    image2 = cv::imread(image2_file);

    if (image1.empty() || image2.empty()) {
        std::cerr << "Failed to read images" << std::endl;
        return 1;
    }

    // setup identifier
    if (!identifier.setup(input_cfg_file, input_weights_file)) {
        std::cerr << "Setup failed" << std::endl;
        return 1;
    }

    // setup preprocessor
    pre.setup(identifier.get_width(), identifier.get_height());

    auto prevTime = std::chrono::system_clock::now();

    // preprocess images
    if (!pre.run(image1, blob1) || !pre.run(image2, blob2)) {
        std::cerr << "Failed to preprocess images" << std::endl;
        return 1;
    }

    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> period = (now - prevTime);
    std::cout << "Preprocess time: " << period.count() / 2.0 << std::endl;
    prevTime = std::chrono::system_clock::now();

    // calculate image identifiers
    if (!identifier.predict(blob1) || !identifier.get_identifier(id1) ||
        !identifier.predict(blob2) || !identifier.get_identifier(id2)) {
        std::cerr << "Failed to predict image identifier" << std::endl;
        return 1;
    }

    now = std::chrono::system_clock::now();
    period = (now - prevTime);
    std::cout << "Predict time: " << period.count() / 2.0 << std::endl;

    std::cout << "L2 difference = " << cv::norm(id1, id2) << std::endl;

    return 0;
}
