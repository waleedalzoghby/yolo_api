/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Preprocess opencv Mat for network inference
 */

#include "preprocess_cv.hpp"
#include "logging.hpp"

#ifdef OPENCV

using namespace Darknet;

void PreprocessCv::setup(size_t width, size_t height, size_t batch, std::vector<unsigned int> channel_map)
{
    m_width = width;
    m_height = height;
    m_batch = batch;
    m_channel_map = channel_map;

    m_channels = m_channel_map.size();
    m_expected_blob_size = m_width * m_height * m_channels * m_batch;
    m_batch_step = m_width * m_height * m_channels;

    // internal storage Mats
    m_image_resized = cv::Mat(m_height, m_width, CV_8UC(m_channels));
    m_image_resized_float = cv::Mat(m_height, m_width, CV_32FC(m_channels));
}

bool PreprocessCv::run(const cv::Mat& image, std::vector<float>& blob)
{
    std::vector<cv::Mat> images;

    images.push_back(image);
    return run(images, blob);
}

bool PreprocessCv::run(const std::vector<cv::Mat>& images, std::vector<float>& blob)
{
    // ensure blob has the right size
    if (blob.size() != m_expected_blob_size) {
        blob.resize(m_expected_blob_size);
    }

    if (images.size() > m_batch) {
        EPRINTF("Number of images (%lu) must be smaller than the configured batch size (%lu)\n", images.size(), m_batch);
        return false;
    }

    // preprocess every image into a batch slot
    for (size_t i=0; i<images.size(); ++i) {
        if (!cv_to_tensor_data(images[i], &blob[i * m_batch_step]))
            return false;
    }

    return true;
}

bool PreprocessCv::cv_to_tensor_data(const cv::Mat image, float* blob)
{
    const size_t in_width = image.cols;
    const size_t in_height = image.rows;
    const cv::Scalar border_color = cv::Scalar(0.5, 0.5, 0.5);
    cv::Rect rect_image, rect_greyborder1, rect_greyborder2;
    cv::Mat roi_image, roi_image_float, roi_greyborder1, roi_greyborder2;
    std::vector<cv::Mat> float_mat_channels(3);

    if (image.channels() != m_channels) {
        EPRINTF("Number of image channels (%d) does not match number of configured channels (%d)\n", image.channels(), m_channels);
        return false;
    }

    if (image.depth() != CV_8U) {
        EPRINTF("Mat is not a CV_8U mat\n");
        return false;
    }

    // assign Mat objs in case some steps can be bypassed
    roi_image_float = m_image_resized_float;

    // if image does not fit network input resolution, check if resize and or letterboxing is needed
    if (in_width != m_width || in_height != m_height) {

        // if aspect ratio differs, apply letterboxing
        if (in_height * m_width != in_width * m_height) {

            // calculate rectangles for letterboxing
            if (in_height * m_width < in_width * m_height) {
                const int image_h = (in_height * m_width) / in_width;
                const int border_h = std::ceil((m_height - image_h) / 2.0);
                rect_image = cv::Rect(0, border_h, m_width, image_h);
                rect_greyborder1 = cv::Rect(0, 0, m_width, border_h);
                rect_greyborder2 = cv::Rect(0, (m_height + image_h) / 2, m_width, border_h);
            } else {
                const int image_w = (in_width * m_height) / in_height;
                const int border_w = std::ceil((m_width - image_w) / 2.0);
                rect_image = cv::Rect(border_w, 0, image_w, m_height);
                rect_greyborder1 = cv::Rect(0, 0, border_w, m_height);
                rect_greyborder2 = cv::Rect((m_width + image_w) / 2, 0, border_w, m_height);
            }

            roi_image = cv::Mat(m_image_resized, rect_image);
            roi_image_float = cv::Mat(m_image_resized_float, rect_image);          // image area

            roi_greyborder1 = cv::Mat(m_image_resized_float, rect_greyborder1);    // grey area top/left
            roi_greyborder2 = cv::Mat(m_image_resized_float, rect_greyborder2);    // grey area bottom/right

            // paint borders grey
            roi_greyborder1.setTo(border_color);
            roi_greyborder2.setTo(border_color);

        } else {
            roi_image = m_image_resized;
        }

        // resize if dimensions of the input image are different from the
        // ROI dimensions within the network input area
        if (image.size() != roi_image.size()) {
            cv::resize(image, roi_image, roi_image.size(), 0, 0, cv::INTER_LINEAR);
        } else {
            roi_image = image;
        }

    } else {
        roi_image = image;
    }

    // uint8 to float image and normalise (between 0 and 1)
    roi_image.convertTo(roi_image_float, CV_32FC(m_channels), 1/255.0);

    // Remap image channels from XYZXYZXYZ...XYZ -> XXX...XXXYYY...YYYZZZ...ZZZ
    for (size_t i=0; i<m_channel_map.size(); ++i) {
        float_mat_channels[m_channel_map[i]] = cv::Mat(m_height, m_width, CV_32FC1, &blob[i * m_width * m_height]);
    }

    cv::split(m_image_resized_float, float_mat_channels);

    return true;
}

#endif /* OPENCV */
