/*
 *  Class API exports for windows DLLs
 *  Since it is hard to directly operate on C++ classes from .NET, this file
 *  provides plain C wrapper functions for class base methods
 */
#include "darknet.hpp"

using namespace Darknet;

#ifdef WIN32
#define EXPORT_DLL __declspec(dllexport)
#else
#define EXPORT_DLL
#endif

extern "C" {
    // predictor.hpp
	EXPORT_DLL bool predictor_setup(Identifier* self, const char* net_cfg_file, const char* weight_cfg_file) { return self->setup(net_cfg_file, weight_cfg_file); }
	EXPORT_DLL bool predictor_predict(Identifier* self, float* data, size_t size) { return self->predict(data, size); }
	EXPORT_DLL int predictor_get_width(Identifier* self) { return self->get_width(); }
	EXPORT_DLL int predictor_get_height(Identifier* self) { return self->get_height(); }
	EXPORT_DLL int predictor_get_channels(Identifier* self) { return self->get_channels(); }
	EXPORT_DLL int predictor_get_batch(Identifier* self) { return self->get_batch(); }

    // identifier.hpp
	EXPORT_DLL Identifier* identifier_ctor() { return new Identifier(); }
	EXPORT_DLL void identifier_dtor(Identifier* self) { delete self; }
	EXPORT_DLL bool identifier_get_identifier(Identifier* self, float* identifier, size_t size, int batch_idx) { return self->get_identifier(identifier, size, batch_idx); }
	EXPORT_DLL size_t identifier_get_identifier_size(Identifier* self) { return self->get_identifier_size(); }

    // detector.hpp
    EXPORT_DLL Detector* detector_ctor() { return new Detector(); }
    EXPORT_DLL void detector_dtor(Detector* self) { delete self; }
    EXPORT_DLL bool detector_setup(Detector* self, const char* net_cfg_file, const char* weight_cfg_file, float nms, float thresh, float hier_thresh)
        { return self->setup(net_cfg_file, weight_cfg_file, nms, thresh, hier_thresh); }
    EXPORT_DLL bool detector_post_process(Detector* self, size_t width, size_t height, int batch_idx) { return self->post_process(width, height, batch_idx); }
    EXPORT_DLL bool detector_get_detections(Detector* self, Detection* detections, size_t size) { return self->get_detections(detections, size); }
    EXPORT_DLL size_t detector_get_num_detections(Detector* self) { return self->get_num_detections(); }
}
