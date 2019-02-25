/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Detector API
 */

#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include "predictor.hpp"
#include "detection.hpp"

namespace Darknet
{

class Detector : public Predictor
{
public:
    Detector();

    /* NOTE: detector currently only supports batch size 1 */

    /*
     *  Setup network for detection
     *  label_names_file:   file with class names, newline separated
     *  net_cfg_file:       network configuration file that describes the network architecture
     *  weight_cfg_file:    weights file that contains the trained network weights
     *  nms:                non maxima suppression threshold (number between 0 and 1)
     *  thresh:             detection threshold. Detection probabilities lower than this threshold
     *                      will not be considered detections (number between 0 and 1)
     *  hier_thres:         Hierarchical threshold ??? (number between 0 and 1)
     *
     *  returns true on success
     */
    bool setup(std::string label_names_file,
                std::string net_cfg_file,
                std::string weight_cfg_file,
                float nms,
                float thresh,
                float hier_thresh);

    /*
     *  Get detections and calculate NMS
     *  detections:     list of retrieved detections
     *  width:          width dimension of the detections (normally the original image width)
     *  height:         height dimension of the detections (normally the original image height)
     *  returns true on success
     *
     *  The detection values x, y, width, height have dimensions according to the given width/height
     *  if width or height are zero, relative coordinates are used (between 0 and 1)
     *
     */
    bool get_detections(std::vector<Detection>& detections, size_t width = 0, size_t height = 0);

private:

    /* Pimpl idiom: hide original implementation from this api */
    class   impl;
    std::shared_ptr<impl> pimpl;
};

} /* namespace Darknet */

#endif /* DETECTOR_HPP */
