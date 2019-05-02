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
     *  Setup network for detection, call this one i.s.o. the setup method in the Predictor class
     *  net_cfg_file:       network configuration file that describes the network architecture
     *  weight_cfg_file:    weights file that contains the trained network weights
     *  nms:                non maxima suppression threshold (number between 0 and 1)
     *  thresh:             detection threshold. Detection probabilities lower than this threshold
     *                      will not be considered detections (number between 0 and 1)
     *  hier_thres:         Hierarchical threshold ??? (number between 0 and 1)
     *
     *  returns true on success
     */
    bool setup(std::string net_cfg_file,
                std::string weight_cfg_file,
                float nms,
                float thresh,
                float hier_thresh);

    /*
     *  Post process detections for one forward pass (call after predict)
     *  This method calculates bounding boxes, probabilties and applies NMS
     *
     *  width:          width dimension of the detections (normally the original image width)
     *  height:         height dimension of the detections (normally the original image height)
     *  batch_idx:      Currently ignored
     *  returns true on success
     *
     *  The detection values x, y, width, height have dimensions according to the given width/height
     *  if width or height are zero, relative coordinates are used (values between 0 and 1)
     */
    bool post_process(size_t width = 0, size_t height = 0, int batch_idx = 0);

    /*
     *  Get the post processed detections (call after post_process)
     *  returns a list of detections
     */
    std::vector<Detection> get_detections();

    /*
     *  Get the post processed detections (call after post_process)
     *  detections:     Buffer to hold a list of retrieved detections provided by the user.
     *                  The buffer must be large enough. Call get_num_detections() to know the
     *                  minimum required buffer size.
     *  size:           Size of the buffer (in number of Detection structs)
     *  returns true on success
     */
    bool get_detections(Detection* detections, size_t size);

    /*
     *  Return the number of detections in the last output
     */
    size_t get_num_detections();

private:

    /* Pimpl idiom: hide original implementation from this api */
    class   impl;
    std::shared_ptr<impl> pimpl;
};

} /* namespace Darknet */

#endif /* DETECTOR_HPP */
