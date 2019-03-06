/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Base predictor API
 */

#include "predictor.hpp"
#include "predictor_impl.hpp"

using namespace Darknet;

Predictor::~Predictor()
{
}

void Predictor::set_impl(std::shared_ptr<Predictor::impl> impl)
{
    pimpl = impl;
}

bool Predictor::setup(std::string net_cfg_file, std::string weight_cfg_file)
{
    return pimpl->setup(net_cfg_file, weight_cfg_file);
}

void Predictor::teardown()
{
    pimpl->teardown();
}

bool Predictor::predict(const std::vector<float>& data)
{
    return pimpl->predict(&data[0], data.size());
}

bool Predictor::predict(const float* data, size_t size)
{
    return pimpl->predict(data, size);
}

int Predictor::get_width()
{
    return pimpl->get_width();
}

int Predictor::get_height()
{
    return pimpl->get_height();
}

int Predictor::get_channels()
{
    return pimpl->get_channels();
}

int Predictor::get_batch()
{
    return pimpl->get_batch();
}
