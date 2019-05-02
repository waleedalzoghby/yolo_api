/*
 *  Author: Maarten Vandersteegen EAVISE
 *  Description: Detection structure
 */

#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <string>

namespace Darknet
{
    struct Detection
    {
        float x;            // box relative center x position
        float y;            // box relative center y position
        float width;        // box relative width
        float height;       // box relative height
        float probability;  // label probability
        int label_index;    // label index (starts with 0)
    };
}

#endif /* DETECTION_HPP */
