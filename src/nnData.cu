#include "nnData.cuh"
#include "cudaAssert.cuh"
#include "defines.cuh"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdio>


__global__ void print_i_o(float *dev_input, float *dev_output, int numInp, int numOut, int numData)
{
    for (int i = 0; i < numData * numInp; i++)
    {
        printf("%f ", dev_input[i]);
    }
    printf("\n");
    for (int i = 0; i < numData * numOut; i++)
    {
        printf("%f ", dev_output[i]);
    }
    printf("\n");
}

NeuralNetData::NeuralNetData(std::string filePath)
{

    std::ifstream file(filePath.c_str());
    std::stringstream iss;
    if (!file.is_open())
    {
        throw std::invalid_argument("couldn't open file");
    }

    iss << file.rdbuf();

    iss >> numData >> inputSize >> outputSize;

    std::cout << numData << " " << inputSize << " " << outputSize << "\n";

    std::cout << numData * inputSize * 4 / 8000 << "kB\n";

    host_input = new float[numData * inputSize];
    host_output = new float[numData * outputSize];

    for (int item = 0; item < numData; item++)
    {
        for (int inp_i = 0; inp_i < inputSize; inp_i++)
        {
            float x;
            iss >> x;
            std::cout << x << " ";
            host_input[index(inp_i, item, numData)] = x;
        }

        for (int out_i = 0; out_i < outputSize; out_i++)
        {
            float x;
            iss >> x;
            std::cout << x << " ";
            host_output[index(out_i, item, numData)] = x;
        }
        std::cout << "\n";
    }

    printf("\n---------\n");

    // for (int i = 0; i < numData * inputSize; i++)
    // {
    //     printf("%f ", host_input[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < numData * outputSize; i++)
    // {
    //     printf("%f ", host_output[i]);
    // }

    printf("\n---------\n");

    file.close();
    cursor = 0;
}

void NeuralNetData::loadBatch(float *input, float *output, int batchSize)
{
    for (int i = 0; i < batchSize; i++)
    {
        int itemIdx = rand() % numData;
        // itemIdx = i % numData;
        itemIdx = (cursor + i) % numData;
        for (int inp_i = 0; inp_i < inputSize; inp_i++)
        {
            input[index(inp_i, i, batchSize)] = host_input[index(inp_i, itemIdx, numData)];
        }
        for (int out_i = 0; out_i < outputSize; out_i++)
        {
            output[index(out_i, i, batchSize)] = host_output[index(out_i, itemIdx, numData)];
        }
    }
    cursor += batchSize;
}
