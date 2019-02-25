/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Identification example with multi batch processing
 */

#include "darknet.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

int main(int argc, char *argv[])
{
    std::vector<cv::Mat> images;
    std::vector<float> blob;
    std::vector<float> id;
    Darknet::PreprocessCv pre;
    Darknet::Identifier identifier;

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_cfg_file> <input_weights_file> <image1> [<image2>] ..." << std::endl;
        return 1;
    }

    std::string input_cfg_file(argv[1]);
    std::string input_weights_file(argv[2]);

    // read images
    for (int i=3; i<argc; ++i) {
        auto image = cv::imread(argv[i]);
        if (image.empty()) {
            std::cerr << "Failed to read image " << argv[i] << std::endl;
            return 1;
        }
        images.push_back(image);
    }

    // setup identifier
    if (!identifier.setup(input_cfg_file, input_weights_file)) {
        std::cerr << "Setup failed" << std::endl;
        return 1;
    }

    // check that batch in cfg file is large enough
    if (identifier.get_batch() < static_cast<int>(images.size())) {
        std::cerr << "Batch size (" << identifier.get_batch() << ") must be >= the number of given images ("
                  << images.size() << ")" << std::endl;
        return 1;
    }

    // setup preprocessor
    pre.setup(identifier.get_width(), identifier.get_height(), identifier.get_batch());

    auto prevTime = std::chrono::system_clock::now();

    // preprocess images in batch
    if (!pre.run(images, blob)) {
        std::cerr << "Failed to preprocess images" << std::endl;
        return 1;
    }

    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> period = (now - prevTime);
    std::cout << "Preprocess time: " << period.count() / 2.0 << std::endl;
    prevTime = std::chrono::system_clock::now();

    // calculate image identifiers in batch
    if (!identifier.predict(blob)) {
        std::cerr << "Failed to predict image identifiers" << std::endl;
        return 1;
    }

    now = std::chrono::system_clock::now();
    period = (now - prevTime);
    std::cout << "Predict time: " << period.count() / 2.0 << std::endl;

    // get individual image identifiers
    for (size_t i=0; i<images.size(); ++i) {
        if (!identifier.get_identifier(id, i)) {
            std::cerr << "Failed to get image identifier" << std::endl;
            return 1;
        }
        std::cout << "Abs L2 norm " << i << ": " << cv::norm(id) << std::endl;
    }

    return 0;
}
