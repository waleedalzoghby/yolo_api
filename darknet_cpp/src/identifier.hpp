/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Identifier API
 */
#ifndef IDENTIFIER_HPP
#define IDENTIFIER_HPP

#include "predictor.hpp"

namespace Darknet
{

class Identifier : public Predictor
{
public:
    Identifier();

    /*
     *  Retrieve the identification vector of the last prediction
     *  identifier: resulting output vector
     *  batch_idx:  index of the batch element to fetch, must be smaller than the batch size
     *
     *  returns true on success
     */
    bool get_identifier(std::vector<float>& identifier, int batch_idx = 0);

private:

    /* Pimpl idiom: hide original implementation from this api */
    class   impl;
    std::shared_ptr<impl> pimpl;
};

}

#endif /* IDENTIFIER_HPP */
