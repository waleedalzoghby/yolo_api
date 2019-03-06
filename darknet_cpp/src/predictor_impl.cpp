/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Predictor implementation class
 */

#include "predictor_impl.hpp"
#include "logging.hpp"
#include <fstream>

using namespace Darknet;

Predictor::impl::impl() :
        m_bSetup(false),
        m_net(nullptr) {}

Predictor::impl::~impl()
{
    teardown();
}

bool Predictor::impl::setup(std::string net_cfg_file, std::string weight_cfg_file)
{
    if (m_bSetup) {
        EPRINTF("Network already setup!\n");
        return false;
    }

    if (!file_exists(net_cfg_file)) {
        EPRINTF("Network cfg file %s not found\n", net_cfg_file.c_str());
        return false;
    }

    if (!file_exists(weight_cfg_file)) {
        EPRINTF("Weights file %s not found\n", weight_cfg_file.c_str());
        return false;
    }

    m_net = load_network(net_cfg_file.c_str(), weight_cfg_file.c_str(), 0);
    if (!m_net) {
        EPRINTF("Failed to load network %s, %s\n", net_cfg_file.c_str(), weight_cfg_file.c_str());
        return false;
    }

    DPRINTF("Setup: net->n = %d\n", m_net->n);
    DPRINTF("Setup: Done\n");
    m_bSetup = true;
    return true;
}

void Predictor::impl::teardown()
{
    m_bSetup = false;

    if (m_net) {
        free_network(m_net);
        m_net = nullptr;
    }
}

bool Predictor::impl::predict(const float* data, size_t size)
{
    if (!m_bSetup) {
        EPRINTF("Not Setup!\n");
        return false;
    }

    size_t expected_input_size = m_net->w * m_net->h * m_net->c * m_net->batch;
    if (size != expected_input_size) {
        EPRINTF("Expected data input size to be %lu, got %lu", expected_input_size, size);
        return false;
    }

    (void) network_predict(m_net, const_cast<float*>(data));

    return true;
}

int Predictor::impl::get_width()
{
    if (!m_bSetup)
        return 0;

    return m_net->w;
}

int Predictor::impl::get_height()
{
    if (!m_bSetup)
        return 0;

    return m_net->h;
}

int Predictor::impl::get_channels()
{
    if (!m_bSetup)
        return 0;

    return m_net->c;
}

int Predictor::impl::get_batch()
{
    if (!m_bSetup)
        return 0;

    return m_net->batch;
}

bool Predictor::impl::file_exists(const std::string& file)
{
    std::ifstream f(file.c_str());
    return f.good();
}
