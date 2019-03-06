/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Base predictor API
 */

#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include <string>
#include <memory>
#include <vector>

namespace Darknet
{

class Predictor
{
public:
    virtual ~Predictor();

    /*
     *  General network setup
     *  net_cfg_file:       network configuration file that describes the network architecture
     *  weight_cfg_file:    weights file that contains the trained network weights
     *
     *  returns true on success
     */
    virtual bool setup(std::string net_cfg_file, std::string weight_cfg_file);

    /*
     *  Cleanup the network
     */
    void teardown();

    /*
     *  Run the network on the given preprocessed input data
     *  data:   preprocessed data blob that matches the network input
     *  returns true on success
     */
    bool predict(const std::vector<float>& data);

    /*
     *  Run the network on the given preprocessed input data
     *  data:   buffer with preprocessed data blob that matches the network input
     *  size:   size of the buffer in number of floats
     *  returns true on success
     */
    bool predict(const float* data, size_t size);

    /*
     *  Return the input width of the network
     */
    int get_width();

    /*
     *  Return the input height of the network
     */
    int get_height();

    /*
     *  Return the number of input channels of the network
     */
    int get_channels();

    /*
     *  Return network batch size
     */
    int get_batch();

protected:
    class   impl;

    void set_impl(std::shared_ptr<impl> impl);

    /* Pimpl idiom: hide original implementation from this api */
    std::shared_ptr<impl> pimpl;
};

} /* namespace Darknet */

#endif /* PREDICTOR_HPP */
