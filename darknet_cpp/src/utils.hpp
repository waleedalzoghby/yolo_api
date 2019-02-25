/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Utilities for post processing detections
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include "detection.hpp"
#include <vector>

#ifdef OPENCV

#include <opencv2/opencv.hpp>

namespace Darknet
{
    /*
     *  Render detection bouding boxes on a given input image
     *  detections: list of detections
     *  image:      image to use to overlay the detections
     *  NOTE:       assumes the width/height of the image match the width/height dimensions of the detections
     */
    void image_overlay(const std::vector<Detection> detections, cv::Mat& image);
}

#endif /* OPENCV */

namespace Darknet
{
    /*
     *  Filter detections based on their label
     *  input:      detection list to filter
     *  output:     filtered detections
     *  include:    list of labels to include in the filtered detections, all detections with labels not in this list will be removed
     */
    void filter_detections(const std::vector<Detection> input, std::vector<Detection>& output, std::vector<std::string> include);
}

#endif /* UTILS_HPP */
