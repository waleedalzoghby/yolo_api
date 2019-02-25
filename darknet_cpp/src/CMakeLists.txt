include_directories (${OpenCV_INCLUDE_DIRS})
include_directories (${CUDA_INCLUDE_DIRS})
include_directories ("${DARKNET_ROOT}/include")

set (HEADERS
    darknet.hpp
    detection.hpp
    detector.hpp
    identifier.hpp
    logging.hpp
    predictor.hpp
    preprocess_cv.hpp
    utils.hpp
)
set (PRIVATE_HEADERS
    predictor_impl.hpp
)

set (SOURCES
    detector.cpp
    identifier.cpp
    predictor.cpp
    predictor_impl.cpp
    preprocess_cv.cpp
    utils.cpp
)

set (SRC ${SOURCES} ${HEADERS} ${PRIVATE_HEADERS})

set (DEBUG_STR "0")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set (DEBUG_STR "1")
endif()

# original darknet library
# this target is always out of date, but since the darknet makefile already handles
# 'out of date', this will work as expected
add_custom_target (
    darknet
    COMMAND make GPU=${WITH_GPU_STR} CUDNN=${WITH_CUDNN_STR} OPENCV=0 DEBUG=${DEBUG_STR} ASAN=${WITH_ADDRESS_SANITIZER_STR} ARCH="${COMPUTE_ARCH_FLAGS}" obj libdarknet.a
    WORKING_DIRECTORY ${DARKNET_ROOT}
)

# clean target for both darknet and darknet_cpp
add_custom_target (
    cleanall
    COMMAND ${CMAKE_MAKE_PROGRAM} clean
    COMMAND make -C ${DARKNET_ROOT} clean
)

add_library (darknet_cpp_static STATIC ${SRC})
add_library (darknet_cpp SHARED ${SRC})

add_dependencies (darknet_cpp darknet)
target_link_libraries (darknet_cpp
    ${DARKNET_ROOT}/libdarknet.a
    "-pthread"
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBS}
)

# install library target and header files
install (TARGETS darknet_cpp
    DESTINATION "lib"
)

install (FILES ${HEADERS}
    DESTINATION "include/darknet_cpp"
)
