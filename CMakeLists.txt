cmake_minimum_required(VERSION 3.9)
project(opencl_mandelbrot)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework OpenCL")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
add_executable(opencl_mandelbrot main.cpp cl.hpp)

target_link_libraries(opencl_mandelbrot ${OpenCV_LIBS})