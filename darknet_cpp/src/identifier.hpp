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

    /*
     *  Retrieve the identification vector of the last prediction
     *  identifier: Buffer provided by the user to store the resulting output vector in.
     *              If the buffer is not large enough, the method will return false.
     *              Call get_identifier_size to obtain the minimum required buffer size.
     *  size:       Size of the provided buffer in number of floats
     *  batch_idx:  index of the batch element to fetch, must be smaller than the batch size
     *
     *  returns true on success
     */
    bool get_identifier(float* identifier, size_t size, int batch_idx = 0);

    /*
     *  Get the size of the indentifier vector
     */
    int get_identifier_size();

private:

    /* Pimpl idiom: hide original implementation from this api */
    class   impl;
    std::shared_ptr<impl> pimpl;
};

}

#endif /* IDENTIFIER_HPP */
