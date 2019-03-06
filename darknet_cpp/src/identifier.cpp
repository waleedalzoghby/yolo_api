/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Identifier API
 */

#include "identifier.hpp"
#include "predictor_impl.hpp"
#include "logging.hpp"

using namespace Darknet;

/*
 *  Implementation class, see header for method descriptions
 */

class Identifier::impl : public Predictor::impl
{
public:
    bool get_identifier(float* identifier, int size, int batch_idx);
    int get_identifier_size();
};

/*
 *  Implementations
 */

bool Identifier::impl::get_identifier(float* identifier, int size, int batch_idx)
{
    if (!m_bSetup) {
        EPRINTF("Not Setup!\n");
        return false;
    }

    if (batch_idx < 0 || batch_idx >= m_net->batch) {
        EPRINTF("Batch index must be smaller than %d and greater than zero\n", m_net->batch);
        return false;
    }

    if (size < m_net->outputs) {
        EPRINTF("Input buffer (%d) must be >= network output size (%d)\n", size, m_net->outputs);
        return false;
    }

    memcpy(identifier, &m_net->output[batch_idx * m_net->outputs], m_net->outputs * sizeof(float));

    return true;
}

int Identifier::impl::get_identifier_size()
{
    return m_net->outputs;
}

/*
 *  Wrappers
 */

Identifier::Identifier() :
    pimpl{ std::make_shared<Identifier::impl>() }
{
    set_impl(pimpl);
}

bool Identifier::get_identifier(std::vector<float>& identifier, int batch_idx)
{
    identifier.resize(pimpl->get_identifier_size());
    return pimpl->get_identifier(&identifier[0], identifier.size(), batch_idx);
}

bool Identifier::get_identifier(float* identifier, size_t size, int batch_idx)
{
    return pimpl->get_identifier(identifier, size, batch_idx);
}

int Identifier::get_identifier_size()
{
    return pimpl->get_identifier_size();
}
