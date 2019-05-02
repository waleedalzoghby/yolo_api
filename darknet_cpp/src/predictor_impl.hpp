/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Predictor implementation class
 */

#ifndef PREDICTOR_IMPL_HPP
#define PREDICTOR_IMPL_HPP

#include "predictor.hpp"
#include "darknet.h"                /* original darknet !!! */

namespace Darknet
{

class Predictor::impl
{
public:
    impl();
    ~impl();
    bool setup(std::string net_cfg_file, std::string weight_cfg_file);
    void teardown();
    bool predict(const float* data, size_t size);
    int get_width();
    int get_height();
    int get_channels();
    int get_batch();

protected:
    bool file_exists(const std::string& file);

    bool    m_bSetup;
    network *m_net;
};

}

#endif /* PREDICTOR_IMPL_HPP */
