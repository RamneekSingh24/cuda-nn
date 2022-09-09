
#include "layer.cuh"
#include "cudaAssert.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cstdio>

CudaNNLayer::CudaNNLayer(int inputSize, int outputSize, float *values, float *values_transposed, float *na_values,
                         float *value_gradients, float *weights, float *weights_transposed, float *weight_gradients,
                         float *biases, float *bias_gradients, NeuronActivation activationFunction)
    : numNeurons(outputSize), inputSize(inputSize), outputSize(outputSize),
      values(values), values_transposed(values_transposed), non_activated_values(na_values), value_gradients(value_gradients),
      weights(weights), weights_transposed(weights_transposed), weight_gradients(weight_gradients), biases(biases),
      bias_gradients(bias_gradients), activationFunction(activationFunction){};

__global__ void print_layer(float *input, float *output, float *weights,
                            float *biases, int inputSize, int outputSize, int batchSize, NeuronActivation activation)
{
    printf("\n-----------------\n");
    printf("\ninput\n");

    for (int i = 0; i < inputSize * batchSize; i++)
    {
        printf("%f ", input[i]);
    }
    printf("\noutput\n");
    for (int i = 0; i < outputSize * batchSize; i++)
    {
        printf("%f ", output[i]);
    }

    printf("\nweights\n");
    for (int i = 0; i < inputSize * outputSize; i++)
    {
        printf("%f ", weights[i]);
    }

    printf("\nbiases\n");
    for (int i = 0; i < outputSize; i++)
    {
        printf("%f ", biases[i]);
    }

    printf("\n-----------------\n");
}

#define BLOCK_DIM 16

__global__ void transposeKernel(float *input, float *output, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if ((x < width) && (y < height))
    {
        unsigned int index_in = y * width + x;
        block[threadIdx.y][threadIdx.x] = input[index_in];
    }

    // synchronise to ensure all writes to block[][] have completed
    __syncthreads();

    // write the transposed matrix tile to global memory (odata) in linear order
    x = blockIdx.y * BLOCK_DIM + threadIdx.x;
    y = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if ((x < height) && (y < width))
    {
        unsigned int index_out = y * height + x;
        output[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

__device__ __forceinline__ float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

__global__ void forwardPropKernel(float *input, float *output, float *output_nact, float *weights,
                                  float *biases, int inputSize, int outputSize, int batchSize, NeuronActivation activation)
{
    // weights -> outputSize x inputSize
    // input -> inputSize x batchSize
    // output -> outputSize x batchSize
    // baises -> outputSize x 1

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // x'th value of the y'th data item
    int x = idx / batchSize;
    int y = idx % batchSize;

    // compute output[x][y]
    if (x < outputSize && y < batchSize)
    {
        float value = 0.0;
        for (int i = 0; i < inputSize; i++)
        {
            value += weights[index(x, i, inputSize)] * input[index(i, y, batchSize)];
        }
        value += biases[x];
        output_nact[index(x, y, batchSize)] = value;
        if (activation == Sigmoid)
        {
            value = sigmoid(value);
        }
        output[index(x, y, batchSize)] = value;
    }
}

void CudaNNLayer::forwardProp(float *input)
{
    // weights -> outputSize x inputSize
    // input -> inputSize x batchSize
    // output -> outputSize x batchSize
    // baises -> outputSize x 1

    // print_layer<<<1, 1>>>(input, values, weights, biases,
    //   inputSize, outputSize, BATCH_SIZE, activationFunction);

    dim3 blockSize(THREADS_PER_BLOCK);
    dim3 gridSize((outputSize * BATCH_SIZE + blockSize.x - 1) / blockSize.x);
    forwardPropKernel<<<blockSize, gridSize>>>(input, values, non_activated_values, weights, biases,
                                               inputSize, outputSize, BATCH_SIZE, activationFunction);
    // print_layer<<<1, 1>>>(input, values, weights, biases,
    //   inputSize, outputSize, BATCH_SIZE, activationFunction);

    gridSize = dim3((BATCH_SIZE + BLOCK_DIM - 1) / BLOCK_DIM, (outputSize + BLOCK_DIM - 1) / BLOCK_DIM, 1);
    blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    transposeKernel<<<gridSize, blockSize>>>(values, values_transposed, BATCH_SIZE, outputSize);
    cudaCheckError(cudaDeviceSynchronize());
}

__device__ float activationDerivative(float x, NeuronActivation activation)
{
    if (activation == Identity)
    {
        return 1.0;
    }
    else if (activation == Sigmoid)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }
    return 1.0;
}

__global__ void backwardPropKernel(float *input_grad_vals, float *output_val_grads,
                                   float *neuron_vals_nact, float *weights_transposed,
                                   int inputSize, int outputSize,
                                   int batchSize, NeuronActivation activation)
{
    // input : outputSize x BATCH_SIZE
    // w transpose: inputSize x outputSize
    // out = w tranpose * input * scalar = inputSize x BATCH_SIZE

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // x'th value gradient of the y'th data item
    int x = idx / batchSize;
    int y = idx % batchSize;

    // compute output_val_grads[x][y]
    if (x < inputSize && y < batchSize)
    {
        float value = 0.0;
        // w transpose . input
        for (int i = 0; i < outputSize; i++)
        {
            // w transp(x, i) * input(i, y)
            // w(i, x) * input(i, y) * f' (non_activated(i, y))
            value += weights_transposed[index(x, i, inputSize)] *
                     input_grad_vals[index(i, y, batchSize)] *
                     activationDerivative(neuron_vals_nact[index(i, y, batchSize)], activation);
        }

        output_val_grads[index(x, y, batchSize)] = value;
    }
}

void CudaNNLayer::backwardProp(float *input, cudaStream_t &stream, bool async)
{
    // input : outputSize x BATCH_SIZE
    // w transpose: inputSize x outputSize
    // out = w tranpose * input *elementwise f'(non_activated) = inputSize x BATCH_SIZE
    // weight gradients = input * values transpose(inp of fwd prop transpose)
    //                  = outputSize  x inputSize
    // bias = sum(input's col) = outputSize x 1

    dim3 gridSize, blockSize;
    // the kernel requires w transpose
    gridSize = dim3((inputSize + BLOCK_DIM - 1) / BLOCK_DIM, (outputSize + BLOCK_DIM - 1) / BLOCK_DIM, 1);
    blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    transposeKernel<<<gridSize, blockSize>>>(weights, weights_transposed, inputSize, outputSize);

    blockSize = dim3(THREADS_PER_BLOCK);
    gridSize = dim3((inputSize * BATCH_SIZE + blockSize.x - 1) / blockSize.x);
    backwardPropKernel<<<blockSize, gridSize, 0, stream>>>(input, value_gradients,
                                                           non_activated_values, weights_transposed,
                                                           inputSize, outputSize,
                                                           BATCH_SIZE, activationFunction);

    if (!async)
    {
        cudaCheckError(cudaDeviceSynchronize());
    }
}

__global__ void updateWeightsKernel(float *input_grad_vals, float *input_fwd_prop_input_vals_transposed,
                                    float *neuron_vals_nact, float *weights,
                                    float *biases, int inputSize, int outputSize,
                                    int batchSize, NeuronActivation activation)
{
    // input : outputSize x BATCH_SIZE
    // fwd_prop_inp_values transpose = BATCH_SIZE x inputSize
    // weight gradients = input_gradient * fwd_prop_inp_values transpose(inp of fwd prop transpose)
    //                  = outputSize  x inputSize
    // bias = sum(input's col) = outputSize x 1

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // (x,y) th weight gradient
    // weight connecting xth neuron and yth neuron in the prev layer
    int x = idx / inputSize;
    int y = idx % inputSize;

    // compute weight[x][y]
    if (x < outputSize && y < inputSize)
    {
        float w_gradient = 0.0;
        float bias_gradient = 0.0;
        // input . fwd_prop_inp_values_graidents
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            // input_grad (x, i) * fwd_prop_inp transp (i, y)
            // input_grad(x, i) * f' (non_activated(x, i)) * fwd_prop_inp(y, i)
            float inp_with_act_grad = input_grad_vals[index(x, i, batchSize)] *
                                      activationDerivative(neuron_vals_nact[index(x, i, batchSize)], activation);
            w_gradient += inp_with_act_grad *
                          input_fwd_prop_input_vals_transposed[index(i, y, inputSize)];
            bias_gradient += inp_with_act_grad;
        }
        w_gradient /= batchSize;
        bias_gradient /= batchSize;
        // printf("%d %d %d\n", blockIdx.x, threadIdx.x, index(x,y,inputSize));
        weights[index(x, y, inputSize)] -= LEARNING_RATE * w_gradient;
        if (y == 0)
        {
            biases[x] -= LEARNING_RATE * bias_gradient;
        }
    }
}

void CudaNNLayer::updateWeights(float *input_val_gradients, float *input_fwd_prop_input_vals_transposed, cudaStream_t &stream, bool async)
{

    // printf("\nUpdagting weights...\n");
    // input : outputSize x BATCH_SIZE
    // w transpose: inputSize x outputSize
    // out = w tranpose * input * scalar = inputSize x BATCH_SIZE
    // weight gradients = input * values transpose(inp of fwd prop transpose)
    //                  = outputSize  x inputSize
    // bias = sum(input's col) = outputSize x 1
    dim3 blockSize(THREADS_PER_BLOCK);
    dim3 gridSize((outputSize * inputSize + blockSize.x - 1) / blockSize.x);
    updateWeightsKernel<<<blockSize, gridSize, 0, stream>>>(input_val_gradients, input_fwd_prop_input_vals_transposed,
                                                            non_activated_values, weights,
                                                            biases, inputSize, outputSize,
                                                            BATCH_SIZE, activationFunction);

    if (!async)
    {
        cudaCheckError(cudaDeviceSynchronize());
    }
}
