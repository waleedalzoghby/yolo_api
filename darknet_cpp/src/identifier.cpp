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
    bool get_identifier(std::vector<float>& identifier, int batch_idx);
};

/*
 *  Implementations
 */

bool Identifier::impl::get_identifier(std::vector<float>& identifier, int batch_idx)
{
    if (batch_idx < 0 || batch_idx >= m_net->batch) {
        EPRINTF("Batch index must be smaller than %d and greater than zero\n", m_net->batch);
        return false;
    }

    identifier.resize(m_net->outputs);
    memcpy(&identifier[0], &m_net->output[batch_idx * m_net->outputs], m_net->outputs * sizeof(float));

    return true;
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
    return pimpl->get_identifier(identifier, batch_idx);
}
