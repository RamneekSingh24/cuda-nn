
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
    __shared__ float block[BLOCK_DIM][BLOCK_DIM];

    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

    // global memory reads are coalesced
    if (x < width && y < height)
    {
        block[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    x = blockIdx.y * BLOCK_DIM + threadIdx.x;
    y = blockIdx.x * BLOCK_DIM + threadIdx.y;

    // global memory writes are coalesced
    if (x < height && y < width)
    {
        output[y * height + x] = block[threadIdx.x][threadIdx.y];
    }
}

__device__ __forceinline__ float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

// __device__ void matrixMult(float *A_inp, float *B_inp, float *C_out, int A_width, int A_height, int B_width, int B_height)
// {
// }

__global__ void forwardPropKernel(float *input, float *output, float *output_nact, float *weights,
                                  float *biases, int inputSize, int outputSize, int batchSize, NeuronActivation activation)
{

    // out = W * inp + b

    // weights -> outputSize x inputSize
    // input -> inputSize x batchSize
    // output -> outputSize x batchSize
    // baises -> outputSize x 1

    __shared__ float weightTile[BLOCK_DIM][BLOCK_DIM];
    __shared__ float inputTile[BLOCK_DIM][BLOCK_DIM];

    int numTiles = (inputSize + BLOCK_DIM - 1) / BLOCK_DIM;
    float value = 0.0f;
    // out_tile[i, j] = sum_{k}(in_tile1[i,k] * in_tile2[k,j])
    for (int tileNum = 0; tileNum < numTiles; tileNum++)
    {
        // copy the weights tile in shrared mem
        int x = tileNum * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < inputSize && y < outputSize)
        {
            weightTile[threadIdx.y][threadIdx.x] = weights[index(y, x, inputSize)];
        }
        else
        {
            // pad
            weightTile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // copy the input tile in shared mem

        y = tileNum * blockDim.y + threadIdx.y;
        x = blockIdx.x + blockDim.x + threadIdx.x;

        if (x < batchSize && y < inputSize)
        {
            inputTile[threadIdx.y][threadIdx.x] = input[index(y, x, batchSize)];
        }
        else
        {
            inputTile[threadIdx.y][threadIdx.x] = 0;
        }
        // sync threads so that the whole tile is built
        __syncthreads();
        // tile multiplication
        for (int i = 0; i < BLOCK_DIM; i++)
        {
            value += weightTile[threadIdx.y][i] * inputTile[i][threadIdx.x];
        }
        // wait until all the threads are done with the computation
        // before moving on to the next tile and updated the shared mem.
        __syncthreads();
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < batchSize && y < outputSize)
    {
        value += biases[y];

        output_nact[index(y, x, batchSize)] = value;

        if (activation == Sigmoid)
        {
            value = sigmoid(value);
        }

        output[index(y, x, batchSize)] = value;
    }

    /*
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
    */
}

void CudaNNLayer::forwardProp(float *input)
{
    // weights -> outputSize x inputSize
    // input -> inputSize x batchSize
    // output -> outputSize x batchSize
    // baises -> outputSize x 1

    // print_layer<<<1, 1>>>(input, values, weights, biases,
    //   inputSize, outputSize, BATCH_SIZE, activationFunction);

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((BATCH_SIZE + BLOCK_DIM - 1) / BLOCK_DIM, (outputSize + BLOCK_DIM - 1) / BLOCK_DIM);

    forwardPropKernel<<<gridSize, blockSize>>>(input, values, non_activated_values, weights, biases,
                                               inputSize, outputSize, BATCH_SIZE, activationFunction);

    // dim3 blockSize(THREADS_PER_BLOCK);
    // dim3 gridSize((outputSize * BATCH_SIZE + blockSize.x - 1) / blockSize.x);
    // forwardPropKernel<<<blockSize, gridSize>>>(input, values, non_activated_values, weights, biases,
    //                                            inputSize, outputSize, BATCH_SIZE, activationFunction);

    // print_layer<<<1, 1>>>(input, values, weights, biases,
    //   inputSize, outputSize, BATCH_SIZE, activationFunction);

    blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    gridSize = dim3((BATCH_SIZE + BLOCK_DIM - 1) / BLOCK_DIM, (outputSize + BLOCK_DIM - 1) / BLOCK_DIM, 1);
    transposeKernel<<<gridSize, blockSize>>>(values, values_transposed, BATCH_SIZE, outputSize);

    cudaCheckError(cudaDeviceSynchronize());

    // cudaCheckError(cudaDeviceSynchronize());
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

    __shared__ float weightTransposeTile[BLOCK_DIM][BLOCK_DIM];
    __shared__ float inputTile[BLOCK_DIM][BLOCK_DIM];

    int numTiles = (outputSize + BLOCK_DIM - 1) / BLOCK_DIM;
    float value = 0.0f;
    // out_tile[i, j] = sum_{k}(in_tile1[i,k] * in_tile2[k,j])
    for (int tileNum = 0; tileNum < numTiles; tileNum++)
    {
        // copy the weights tile in shrared mem
        int x = tileNum * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < outputSize && y < inputSize)
        {
            weightTransposeTile[threadIdx.y][threadIdx.x] = weights_transposed[index(y, x, outputSize)];
        }
        else
        {
            // pad
            weightTransposeTile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // copy the input tile in shared mem

        y = tileNum * blockDim.y + threadIdx.y;
        x = blockIdx.x + blockDim.x + threadIdx.x;

        //  input : outputSize x BATCH_SIZE
        if (x < batchSize && y < outputSize)
        {
            inputTile[threadIdx.y][threadIdx.x] = input_grad_vals[index(y, x, batchSize)] *
                                                  activationDerivative(neuron_vals_nact[index(y, x, batchSize)], activation);
        }
        else
        {
            inputTile[threadIdx.y][threadIdx.x] = 0;
        }
        // sync threads so that the whole tile is built
        __syncthreads();
        // tile multiplication
        for (int i = 0; i < BLOCK_DIM; i++)
        {
            value += weightTransposeTile[threadIdx.y][i] * inputTile[i][threadIdx.x];
        }
        // wait until all the threads are done with the computation
        // before moving on to the next tile and updated the shared mem.
        __syncthreads();
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < batchSize && y < inputSize)
    {
        output_val_grads[index(y, x, batchSize)] = value;
    }

    /*
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
    */
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

    blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    gridSize = dim3((BATCH_SIZE + BLOCK_DIM - 1) / BLOCK_DIM, (inputSize + BLOCK_DIM - 1) / BLOCK_DIM, 1);

    backwardPropKernel<<<gridSize, blockSize, 0, stream>>>(input, value_gradients,
                                                           non_activated_values, weights_transposed,
                                                           inputSize, outputSize,
                                                           BATCH_SIZE, activationFunction);
    /*
    blockSize = dim3(THREADS_PER_BLOCK);
    gridSize = dim3((inputSize * BATCH_SIZE + blockSize.x - 1) / blockSize.x);
    backwardPropKernel<<<blockSize, gridSize, 0, stream>>>(input, value_gradients,
                                                           non_activated_values, weights_transposed,
                                                           inputSize, outputSize,
                                                           BATCH_SIZE, activationFunction);
     */

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
    // input_gradient : outputSize x BATCH_SIZE
    // fwd_prop_inp_values transpose = BATCH_SIZE x inputSize
    // weight gradients = input_gradient * fwd_prop_inp_values transpose
    //                  = outputSize  x inputSize
    // bias = sum(input's col) = outputSize x 1

    __shared__ float inputGradientTiles[BLOCK_DIM][BLOCK_DIM];
    __shared__ float fwdPropInputTransposedTiles[BLOCK_DIM][BLOCK_DIM];

    int numTiles = (BATCH_SIZE + BLOCK_DIM - 1) / BLOCK_DIM;
    float weightGradient = 0.0f;
    float biasGradient = 0.0f;

    // out_tile[i, j] = sum_{k}(in_tile1[i,k] * in_tile2[k,j])
    for (int tileNum = 0; tileNum < numTiles; tileNum++)
    {
        // copy the weights tile in shrared mem
        int x = tileNum * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < BATCH_SIZE && y < outputSize)
        {
            inputGradientTiles[threadIdx.y][threadIdx.x] = input_grad_vals[index(y, x, batchSize)] *
                                                           activationDerivative(neuron_vals_nact[index(y, x, batchSize)], activation);
        }
        else
        {
            // pad
            inputGradientTiles[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // copy the input tile in shared mem

        y = tileNum * blockDim.y + threadIdx.y;
        x = blockIdx.x + blockDim.x + threadIdx.x;

        // fwd_prop_inp_values transpose = BATCH_SIZE x inputSize
        if (x < inputSize && y < BATCH_SIZE)
        {
            fwdPropInputTransposedTiles[threadIdx.y][threadIdx.x] = input_fwd_prop_input_vals_transposed[index(y, x, inputSize)];
        }
        else
        {
            fwdPropInputTransposedTiles[threadIdx.y][threadIdx.x] = 0;
        }
        // sync threads so that the whole tile is built
        __syncthreads();
        // tile multiplication
        for (int i = 0; i < BLOCK_DIM; i++)
        {
            weightGradient += inputGradientTiles[threadIdx.y][i] * fwdPropInputTransposedTiles[i][threadIdx.x];
            biasGradient += inputGradientTiles[threadIdx.y][i];
        }
        // wait until all the threads are done with the computation
        // before moving on to the next tile and updated the shared mem.
        __syncthreads();
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < inputSize && y < outputSize)
    {
        weights[index(y, x, inputSize)] -= LEARNING_RATE * weightGradient;
        if (x == 0)
        {
            biases[y] -= LEARNING_RATE * weightGradient;
        }
    }

    /*
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
    */
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

    dim3 blockSize = dim3(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize = dim3((inputSize + BLOCK_DIM - 1) / BLOCK_DIM, (outputSize + BLOCK_DIM - 1) / BLOCK_DIM, 1);

    updateWeightsKernel<<<gridSize, blockSize, 0, stream>>>(input_val_gradients, input_fwd_prop_input_vals_transposed,
                                                            non_activated_values, weights,
                                                            biases, inputSize, outputSize,
                                                            BATCH_SIZE, activationFunction);

    // dim3 blockSize(THREADS_PER_BLOCK);
    // dim3 gridSize((outputSize * inputSize + blockSize.x - 1) / blockSize.x);
    // updateWeightsKernel<<<blockSize, gridSize, 0, stream>>>(input_val_gradients, input_fwd_prop_input_vals_transposed,
    //                                                         non_activated_values, weights,
    //                                                         biases, inputSize, outputSize,
    //                                                         BATCH_SIZE, activationFunction);

    if (!async)
    {
        cudaCheckError(cudaDeviceSynchronize());
    }
}
