#include "nn.cuh"
#include "cudaAssert.cuh"
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <device_launch_parameters.h>

NeuralNet::NeuralNet(int inputSize, std::vector<int> internalLayerSizes, std::vector<NeuronActivation> activations,
                     int outputSize)
{
    srand(42);

    int prevLayerSize = inputSize;
    int numWeights = 0;
    int numNeurons = inputSize;

    for (size_t i = 0; i < internalLayerSizes.size(); i++)
    {
        numWeights += prevLayerSize * internalLayerSizes[i];
        prevLayerSize = internalLayerSizes[i];
        numNeurons += internalLayerSizes[i];
    }

    numWeights += prevLayerSize * outputSize;
    numNeurons += outputSize;

    // init weights
    float *host_weights = new float[numWeights];
    float *dev_weights;
    float *dev_weight_gradients;
    float *dev_weights_transposed;
    cudaCheckError(cudaMalloc(&dev_weights, sizeof(float) * numWeights));
    cudaCheckError(cudaMalloc(&dev_weight_gradients, sizeof(float) * numWeights));
    cudaCheckError(cudaMalloc(&dev_weights_transposed, sizeof(float) * numWeights));

    for (int i = 0; i < numWeights; i++)
    {
        host_weights[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        // host_weights[i] = 3.0 + i;
    }

    cudaCheckError(cudaMemcpy(dev_weights, host_weights, sizeof(float) * numWeights, cudaMemcpyHostToDevice));

    delete[] host_weights;

    // init biases
    int numBiases = numNeurons - inputSize;
    float *host_biases = new float[numBiases];
    float *dev_biases;
    float *dev_bias_gradients;
    cudaCheckError(cudaMalloc(&dev_biases, sizeof(float) * numBiases));
    cudaCheckError(cudaMalloc(&dev_bias_gradients, sizeof(float) * numBiases));

    for (int i = 0; i < numBiases; i++)
    {
        host_biases[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        // host_biases[i] = 1.0 + i;
    }

    cudaCheckError(cudaMemcpy(dev_biases, host_biases, sizeof(float) * numBiases, cudaMemcpyHostToDevice));

    delete[] host_biases;

    // init neuron values and layers
    float *dev_neuron_values;
    float *dev_neuron_values_transposed;
    cudaCheckError(cudaMalloc(&dev_neuron_values, sizeof(float) * numNeurons * BATCH_SIZE));
    cudaCheckError(cudaMalloc(&dev_neuron_values_transposed, sizeof(float) * numNeurons * BATCH_SIZE));

    // init non_activated neuron vals
    float *dev_na_neuron_values;
    cudaCheckError(cudaMalloc(&dev_na_neuron_values, sizeof(float) * numNeurons * BATCH_SIZE));

    float *dev_val_gradients;
    cudaCheckError(cudaMalloc(&dev_val_gradients, sizeof(float) * numNeurons * BATCH_SIZE));

    int numNeronsCovered = 0;
    int numWeightsCovered = 0;

    std::vector<int> layerSizes = internalLayerSizes;
    layerSizes.insert(layerSizes.begin(), inputSize);
    layerSizes.push_back(outputSize);

    CudaNNLayer inputLayer(0, layerSizes[0], dev_neuron_values, dev_neuron_values_transposed,
                           dev_na_neuron_values, dev_val_gradients, NULL, NULL, NULL, NULL, NULL);
    numNeronsCovered = layerSizes[0];
    NNlayers.push_back(inputLayer);

    for (size_t i = 1; i < layerSizes.size(); i++)
    {
        CudaNNLayer layer(layerSizes[i - 1],
                          layerSizes[i],
                          dev_neuron_values + numNeronsCovered * BATCH_SIZE,
                          dev_neuron_values_transposed + numNeronsCovered * BATCH_SIZE,
                          dev_na_neuron_values + numNeronsCovered * BATCH_SIZE,
                          dev_val_gradients + numNeronsCovered * BATCH_SIZE,
                          dev_weights + numWeightsCovered,
                          dev_weights_transposed + numWeightsCovered,
                          dev_weight_gradients + numWeightsCovered,
                          dev_biases + (numNeronsCovered - inputSize),
                          dev_bias_gradients + numWeightsCovered,
                          activations[i]);

        NNlayers.push_back(layer);
        numNeronsCovered += layerSizes[i];
        numWeightsCovered += layerSizes[i] * layerSizes[i - 1];
    }
}

__global__ void calcMseGradkernel(float *y, float *predicted_y, float *output_mse_grad, float outputSize, float batchSize)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < outputSize * batchSize)
    {
        output_mse_grad[idx] = 2.0 * (predicted_y[idx] - y[idx]) / batchSize;
    }
}

// not a bottleneck so for now just use a very simple implementation
__global__ void calcMseTotalKernel(float *y, float *predicted_y, float *mse, float outputSize, float batchSize)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < outputSize * batchSize)
    {
        atomicAdd(mse, (predicted_y[idx] - y[idx]) * (predicted_y[idx] - y[idx]) / batchSize);
    }
}

void NeuralNet::run(NeuralNetData data, int numEpochs)
{
    float *batch_input = new float[BATCH_SIZE * data.inputSize];
    float *batch_ouput = new float[BATCH_SIZE * data.outputSize];
    float *dev_batch_output;

    float *dev_last_layer_mse_grads;
    float *dev_mse_value;
    float host_mse = 1e9;

    cudaMalloc(&dev_last_layer_mse_grads, sizeof(float) * BATCH_SIZE * data.outputSize);
    cudaMalloc(&dev_batch_output, sizeof(float) * BATCH_SIZE * data.outputSize);
    cudaMalloc(&dev_mse_value, sizeof(float));

    if (data.inputSize != NNlayers[0].outputSize)
    {
        throw std::logic_error("failed to run nn: logic error");
    }

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    for (int ep = 1; ep <= numEpochs; ep++)
    {
        // train on whole data split by batch sizes
        for (int i = 0; i < data.numData / BATCH_SIZE; i++)
        {
            std::vector<cudaEvent_t> events(NNlayers.size());

            for (size_t i = 0; i < NNlayers.size(); i++)
            {
                cudaEventCreate(&events[i]);
            }

            data.loadBatch(batch_input, batch_ouput, BATCH_SIZE);

            // for (int i = 0; i < BATCH_SIZE * data.inputSize; i++)
            // {
            //     printf("%f ", batch_input[i]);
            // }
            // printf("\n");

            cudaMemcpy(NNlayers[0].values, batch_input,
                       sizeof(float) * BATCH_SIZE * data.inputSize, cudaMemcpyHostToDevice);

            cudaMemcpy(dev_batch_output, batch_ouput,
                       sizeof(float) * BATCH_SIZE * data.outputSize, cudaMemcpyHostToDevice);

            // fwd prop
            for (size_t layerNum = 1; layerNum < NNlayers.size(); layerNum++)
            {
                NNlayers[layerNum].forwardProp(NNlayers[layerNum - 1].values);
            }

            // calc mse gradients

            dim3 blockSize(THREADS_PER_BLOCK);
            dim3 gridSize((data.outputSize * BATCH_SIZE + blockSize.x - 1) / blockSize.x);
            calcMseGradkernel<<<blockSize, gridSize>>>(dev_batch_output,
                                                       NNlayers.back().values,
                                                       dev_last_layer_mse_grads,
                                                       data.outputSize,
                                                       BATCH_SIZE);
            cudaCheckError(cudaDeviceSynchronize());

            // backward prop
            size_t numLayers = NNlayers.size();

            float *dev_input_val_grads = dev_last_layer_mse_grads;

            // now handle previous layers
            for (size_t i = numLayers - 1; i >= 1; i--)
            {
                NNlayers[i].backwardProp(dev_input_val_grads, stream1, true);
                cudaCheckError(cudaEventRecord(events[i], stream1));
                dev_input_val_grads = NNlayers[i].value_gradients;
            }

            // update weights and biases
            dev_input_val_grads = dev_last_layer_mse_grads;
            // now handle previous layers

            for (size_t i = numLayers - 1; i >= 1; i--)
            {
                if (i < numLayers - 1)
                {
                    cudaCheckError(cudaStreamWaitEvent(stream2, events[i + 1], 0));
                }
                NNlayers[i].updateWeights(dev_input_val_grads,
                                          NNlayers[i - 1].values_transposed,
                                          (i % 2 == 0 ? stream2 : stream3),
                                          true);
                dev_input_val_grads = NNlayers[i].value_gradients;
            }
            for (size_t i = 0; i < NNlayers.size(); i++)
            {
                cudaEventDestroy(events[i]);
            }
        }
        cudaCheckError(cudaDeviceSynchronize());

        // calc mse gradients
        cudaMemset(dev_mse_value, 0.0f, sizeof(float));

        dim3 blockSize = dim3(THREADS_PER_BLOCK);
        dim3 gridSize = dim3((data.outputSize * BATCH_SIZE + blockSize.x - 1) / blockSize.x);
        calcMseTotalKernel<<<blockSize, gridSize>>>(dev_batch_output,
                                                    NNlayers.back().values,
                                                    dev_mse_value,
                                                    data.outputSize,
                                                    BATCH_SIZE);
        cudaCheckError(cudaDeviceSynchronize());
        cudaCheckError(cudaMemcpy(&host_mse, dev_mse_value, sizeof(float), cudaMemcpyDeviceToHost));
        if (ep % 10 == 0 || ep == 1)
        {
            printf("Epoch: %d, Current MSE: %f\n", ep, host_mse);
        }
    }

    delete[] batch_input;
    delete[] batch_ouput;
}
