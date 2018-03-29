#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "cl.hpp"

using namespace cl;
using namespace std;

const int DIMS[] = {2,2};
const int M = DIMS[0];
const int N = DIMS[1];

string readSourceFile(string path) {
    ifstream input(path, ios_base::in);
    if(input) {
        return string((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
    } else {
        return string("");
    }
}

void convertToColourAndSaveImage(const cl_int* image, const uint width, const uint height, string fileName) {
    cv::Mat outputImage(height, width, CV_16UC3);
    float grayScale = 0.0f;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            // map iterations to a colour scheme
            int iter = image[i * width + j];
            grayScale = (float) iter / 20000;
            //grayScale = pow(grayScale, 0.25);
            cv::Vec3w colour{
                    static_cast<unsigned short>(0xffff * pow(grayScale, 0.75f)),
                    static_cast<unsigned short>(0xffff * grayScale),
                    static_cast<unsigned short>(0xffff * grayScale * grayScale)
            };
            outputImage.at<cv::Vec3w>(i, j) = colour;
        }
    }
    cv::imwrite(fileName, outputImage);
}
void doMandelbrot(const int imageWidth, const int imageHeight) {
    cl_int error;
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform defaultPlatform = platforms[0];
    if(platforms.empty()) {
        cout << "no platforms. check OpenCL installation.\n";
        exit(1);
    }
    vector<cl::Device> devices;
    defaultPlatform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device defaultDevice = devices[0];

    if(devices.empty()) {
        cout <<"No OPENCL capable GPUs available!" << endl;
        exit(1);
    }
    cl::Program::Sources sources;
    string source = readSourceFile("./kernels.cl");
    sources.push_back({source.c_str(), source.length()});
    cl::Context context(devices);
    cl::Program program(context, source, true, &error);

    if(error != CL_SUCCESS) {
        cout<<"Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) << std::endl;
        exit(1);
    } else {
        cout << "successfully built kernel source: " << "./kernels.cl" << endl;
    }
    cl_int* image = (cl_int*) malloc(imageWidth * imageHeight * sizeof(cl_int));
    memset(image, 0, imageWidth * imageHeight * sizeof(cl_int));
    cl::Buffer mandelbrotImage(context, CL_MEM_READ_ONLY, sizeof(cl_int) * imageWidth * imageHeight);
    cl_float2 c{0.5f, 0.0f};

    cl::Kernel kernel = cl::Kernel(program, "mandelbrot", &error);

    if(error != CL_SUCCESS) {
        cout << "Error creating kernel: " << error << endl;
        exit(1);
    }

    kernel.setArg(0, mandelbrotImage);
    kernel.setArg(1, c);
    kernel.setArg(2, imageWidth);
    kernel.setArg(3, imageHeight);

    cl::CommandQueue commandQueue(context, defaultDevice);

    if(commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(imageWidth, imageHeight)) != CL_SUCCESS) {
        cout << "error queueing kernel" << endl;
    }

    commandQueue.enqueueReadBuffer(mandelbrotImage,
                                   CL_TRUE,
                                   0,
                                   sizeof(cl_int) * imageWidth * imageHeight,
                                   image);

    convertToColourAndSaveImage(image, imageWidth, imageHeight, "mandelbrot.jpeg");
    free(image);
}
int main() {
    doMandelbrot(2000, 2000);
    return 0;
}