/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Detector API implementation
 */

#include "detector.hpp"
#include "predictor_impl.hpp"
#include "logging.hpp"
#include <fstream>

using namespace Darknet;

class Detector::impl : public Predictor::impl
{
public:
    impl();
    bool setup(std::string label_names_file,
                std::string net_cfg_file,
                std::string weight_cfg_file,
                float nms,
                float thresh,
                float hier_thresh);
    bool get_detections(std::vector<Detection>& detections, size_t width, size_t height);

private:
    std::vector<std::string> m_class_names;
    int     m_classes;
    float   m_nms;
    float   m_threshold;
    float   m_hier_threshold;
};

/*
 *  Implementations
 */

Detector::impl::impl() :
        m_class_names(),
        m_classes(0),
        m_nms(0),
        m_threshold(0),
        m_hier_threshold(0) {}

bool Detector::impl::setup(std::string label_names_file,
                std::string net_cfg_file,
                std::string weight_cfg_file,
                float nms,
                float thresh,
                float hier_thresh)
{
    m_nms = nms;
    m_threshold = thresh;
    m_hier_threshold = hier_thresh;

    if (!file_exists(label_names_file)) {
        EPRINTF("Label names file %s not found\n", label_names_file.c_str());
        return false;
    }

    std::ifstream label_names_stream(label_names_file);
    std::string name;

    while (std::getline(label_names_stream, name))
        m_class_names.push_back(name);

    if (!Predictor::impl::setup(net_cfg_file, weight_cfg_file))
        return false;

    layer l = m_net->layers[m_net->n-1];
    m_classes = l.classes;
    DPRINTF("Setup: layers = %d, %d, %d, classes = %d\n", l.w, l.h, l.n, m_classes);

    return true;
}

bool Detector::impl::get_detections(std::vector<Detection>& detections, size_t width, size_t height)
{
    int i;
    int nboxes;
    detection* dets;
    int out_width = width;
    int out_height = height;
    int relative = 0;

    if (!m_bSetup)
        return false;

    if (out_width == 0 || out_height == 0) {
        out_width = m_net->w;
        out_height = m_net->h;
        relative = 1;
    }

    dets = get_network_boxes(m_net, out_width, out_height, m_threshold, m_hier_threshold, 0, relative, &nboxes);

    if (m_nms > 0)
        do_nms(dets, nboxes, m_classes, m_nms);

    // Extract detections in correct format
    detections.clear();
    for (i = 0; i < nboxes; ++i) {
        float prob;
        size_t class_index = max_index(dets[i].prob, m_classes);

        if (class_index >= m_class_names.size()) {
            EPRINTF("Class index exceeds class names list, probably the model does not match the names list\n");
            free_detections(dets, nboxes);
            return false;
        }

        prob = dets[i].prob[class_index];

        if (prob > m_threshold) {
            Detection detection;
            detection.x = dets[i].bbox.x;
            detection.y = dets[i].bbox.y;
            detection.width = dets[i].bbox.w;
            detection.height = dets[i].bbox.h;
            detection.probability = prob;
            detection.label_index = class_index;
            detection.label = m_class_names[class_index];
            detections.push_back(detection);
        }
    }

    free_detections(dets, nboxes);

    return true;
}

/*
 *  Wrappers
 */

Detector::Detector() :
    pimpl{ std::make_shared<Detector::impl>() }
{
    set_impl(pimpl);
}

bool Detector::setup(std::string label_names_file,
                std::string net_cfg_file,
                std::string weight_cfg_file,
                float nms,
                float thresh,
                float hier_thresh)
{
    return pimpl->setup(label_names_file, net_cfg_file, weight_cfg_file, nms,
                            thresh, hier_thresh);
}

bool Detector::get_detections(std::vector<Detection>& detections, size_t width, size_t height)
{
    return pimpl->get_detections(detections, width, height);
}
