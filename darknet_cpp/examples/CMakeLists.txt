include_directories ("${DARKNET_ROOT}/include")
include_directories (${CUDA_INCLUDE_DIRS})
include_directories (${darknet_cpp_SOURCE_DIR}/src)
link_directories (${darknet_cpp_SOURCE_DIR}/src)

# examples that require opencv
if(${WITH_OPENCV})

    add_executable(darknet_cpp_jetson_detection darknet_cpp_jetson_detection.cpp)
    target_link_libraries(darknet_cpp_jetson_detection
        darknet_cpp_static
        ${DARKNET_ROOT}/libdarknet.a
        "-lpthread"
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDNN_LIBS}
    )
    add_dependencies (darknet_cpp_jetson_detection darknet)

    add_executable(darknet_cpp_detection darknet_cpp_detection.cpp)
    target_link_libraries(darknet_cpp_detection
        darknet_cpp_static
        ${DARKNET_ROOT}/libdarknet.a
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDNN_LIBS}
    )
    add_dependencies (darknet_cpp_detection darknet)

    add_executable(darknet_cpp_identifier darknet_cpp_identifier.cpp)
    target_link_libraries(darknet_cpp_identifier
        darknet_cpp_static
        ${DARKNET_ROOT}/libdarknet.a
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDNN_LIBS}
    )
    add_dependencies (darknet_cpp_identifier darknet)

    add_executable(darknet_cpp_identifier_multibatch darknet_cpp_identifier_multibatch.cpp)
    target_link_libraries(darknet_cpp_identifier_multibatch
        darknet_cpp_static
        ${DARKNET_ROOT}/libdarknet.a
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDNN_LIBS}
    )
    add_dependencies (darknet_cpp_identifier_multibatch darknet)

    add_executable(darknet_cpp_detection_threaded darknet_cpp_detection_threaded.cpp)
    target_link_libraries(darknet_cpp_detection_threaded
        darknet_cpp_static
        ${DARKNET_ROOT}/libdarknet.a
        "-lpthread"
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDNN_LIBS}
    )
    add_dependencies (darknet_cpp_detection_threaded darknet)

endif()

add_executable(reproduce_leak reproduce_leak.cpp)
target_link_libraries(reproduce_leak
    darknet_cpp_static
    ${DARKNET_ROOT}/libdarknet.a
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBS}
)
add_dependencies (reproduce_leak darknet)
