/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Preprocess opencv Mat images for network inference
 *               preprocessing applies:
 *                  * image resizing if needed (without changing aspect ratio)
 *                  * letterboxing (grey borders)
 *                  * conversion to float and normalization
 *                  * channel ordering
 */
#ifndef PREPROCESS_CV_HPP
#define PREPROCESS_CV_HPP

#ifdef OPENCV

#include <opencv2/opencv.hpp>

namespace Darknet
{

class PreprocessCv
{
public:

    /*
     *  width:          should match the network input width
     *  height:         should match the network input height
     *  batch:          should match net network batch size, this is also the number of images that need
     *                  to be provided to the run method
     *  channel_map:    Determines the order of how the image channels must be arraged
     *                  Example: If the channels of the input image have order BGR
     *                  and the network expects RGB, then channel_map = [2, 1, 0]
     *                  Example2: If the channels of the input image have order RGB
     *                  and the network expects RGB, then channel_map = [0, 1, 2]
     */
    void setup(size_t width, size_t height, size_t batch = 1, std::vector<unsigned int> channel_map = std::vector<unsigned int>{2, 1, 0});

    /*
     *  Preprocess an input image
     *  image:      input image
     *  blob:       resulting preprocessed blob that can be send to the network for inference
     */
    bool run(const cv::Mat& image, std::vector<float>& blob);

    /*
     *  Preprocess a batch of input images
     *  images:     list of input images (size must equal batch size)
     *  blob:       resulting preprocessed blob that can be send to the network for inference
     */
    bool run(const std::vector<cv::Mat>& images, std::vector<float>& blob);

private:
    bool cv_to_tensor_data(const cv::Mat image, float* blob);

    size_t m_width;
    size_t m_height;
    size_t m_batch;
    std::vector<unsigned int> m_channel_map;
    int m_channels;
    size_t m_expected_blob_size;
    size_t m_batch_step;
    cv::Mat m_image_resized;
    cv::Mat m_image_resized_float;
};

}

#endif /* OPENCV */

#endif /* PREPROCESS_CV_HPP */
