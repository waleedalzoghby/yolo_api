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
    bool setup(std::string net_cfg_file,
                std::string weight_cfg_file,
                float nms,
                float thresh,
                float hier_thresh);
    bool post_process(size_t width, size_t height);
    bool get_detections(Detection* detections, size_t size);
    std::vector<Detection> get_detections();
    size_t get_num_detections();

private:
    int     m_classes;
    float   m_nms;
    float   m_threshold;
    float   m_hier_threshold;
    std::vector<Detection> m_detections;
};

/*
 *  Implementations
 */

Detector::impl::impl() :
        m_classes(0),
        m_nms(0),
        m_threshold(0),
        m_hier_threshold(0),
        m_detections(0) {}

bool Detector::impl::setup(std::string net_cfg_file,
                std::string weight_cfg_file,
                float nms,
                float thresh,
                float hier_thresh)
{
    m_nms = nms;
    m_threshold = thresh;
    m_hier_threshold = hier_thresh;

    if (!Predictor::impl::setup(net_cfg_file, weight_cfg_file))
        return false;

    layer l = m_net->layers[m_net->n-1];
    m_classes = l.classes;
    DPRINTF("Setup: layers = %d, %d, %d, classes = %d\n", l.w, l.h, l.n, m_classes);

    return true;
}

bool Detector::impl::post_process(size_t width, size_t height)
{
    int i;
    int nboxes;
    detection* dets;
    int relative = 0;

    if (!m_bSetup) {
        EPRINTF("Not setup!\n");
        return false;
    }

    if (width == 0 || height == 0) {
        width = m_net->w;
        height = m_net->h;
        relative = 1;
    }

    dets = get_network_boxes(m_net, width, height, m_threshold, m_hier_threshold, 0, relative, &nboxes);

    // nms sets objectness and class probs to zero of suppressed boxes
    if (m_nms > 0)
        do_nms(dets, nboxes, m_classes, m_nms);

    m_detections.clear();

    for (i = 0; i < nboxes; ++i) {
        float prob;

        // find the index where the class probability is the highest
        size_t class_index = max_index(dets[i].prob, m_classes);

        prob = dets[i].prob[class_index];

        if (prob > m_threshold) {
            Detection detection;
            detection.x = dets[i].bbox.x;
            detection.y = dets[i].bbox.y;
            detection.width = dets[i].bbox.w;
            detection.height = dets[i].bbox.h;
            detection.probability = prob;
            detection.label_index = class_index;
            m_detections.push_back(detection);
        }
    }

    free_detections(dets, nboxes);

    return true;
}

bool Detector::impl::get_detections(Detection* detections, size_t size)
{
    if (!m_bSetup) {
        EPRINTF("Not setup!\n");
        return false;
    }

    if (size < m_detections.size()) {
        EPRINTF("Buffer size (%lu) too small to fit number of detections (%lu)\n", size, m_detections.size());
        return false;
    }

    // return a copy
    memcpy(detections, &m_detections[0], m_detections.size() * sizeof(Detection));

    return true;
}

std::vector<Detection> Detector::impl::get_detections()
{
    if (!m_bSetup) {
        EPRINTF("Not setup!\n");
        return std::vector<Detection>();
    }

    // return a copy (implicit vector copy)
    return m_detections;
}

size_t Detector::impl::get_num_detections()
{
    if (!m_bSetup) {
        EPRINTF("Not setup!\n");
        return 0;
    }

    return m_detections.size();
}

/*
 *  Wrappers
 */

Detector::Detector() :
    pimpl{ std::make_shared<Detector::impl>() }
{
    set_impl(pimpl);
}

bool Detector::setup(std::string net_cfg_file,
                std::string weight_cfg_file,
                float nms,
                float thresh,
                float hier_thresh)
{
    return pimpl->setup(net_cfg_file, weight_cfg_file, nms,
                            thresh, hier_thresh);
}

bool Detector::post_process(size_t width, size_t height, int batch_idx)
{
    (void)batch_idx;
    return pimpl->post_process(width, height);
}

std::vector<Detection> Detector::get_detections()
{
    return pimpl->get_detections();
}

bool Detector::get_detections(Detection* detections, size_t size)
{
    return pimpl->get_detections(detections, size);
}

size_t Detector::get_num_detections()
{
    return pimpl->get_num_detections();
}
