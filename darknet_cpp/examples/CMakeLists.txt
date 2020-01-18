include_directories ("${DARKNET_ROOT}/include")
include_directories (${CUDA_INCLUDE_DIRS})
include_directories (${darknet_cpp_SOURCE_DIR}/src)
link_directories (${darknet_cpp_SOURCE_DIR}/src)

# examples that require opencv
if(${WITH_OPENCV})

    add_executable(darknet_cpp_detection darknet_cpp_detection.cpp)
    target_link_libraries(darknet_cpp_detection
        darknet_cpp_static
        ${DARKNET_ROOT}/libdarknet.a
	"-pthread"
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDNN_LIBS}
    )
    add_dependencies (darknet_cpp_detection darknet)


endif()

