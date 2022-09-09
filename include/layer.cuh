#ifndef LAYER_H
#define LAYER_H

#include "activations.cuh"
#include "defines.cuh"


class CudaNNLayer
{
public:
    NeuronActivation activationFunction;
    int numNeurons;

    int inputSize;
    int outputSize;

    // values of the neurons stored during forward prop.
    // the next layer uses this as input during fwd prop.
    // outputSize * BATCH_SIZE
    float *values;

    // TODO use
    // non activated values of the neurons stored during forward prop.
    // the next layer uses this as input during fwd prop.
    // outputSize * BATCH_SIZE
    float *non_activated_values;

    // TODO calc and use
    // error gradients wrt neuron values in this layer.
    // the prev layer uses this as input during bckwd prop.
    // outputSize * BATCH_SIZE
    float *value_gradients;

    // weights for the incoming connections.
    // size = outputSize x inputSize
    float *weights;

    // baises for neurons of this layer, size = numNeurons x 1
    float *biases;

    // outputSize x inputSize
    // UNUSED AS OF NOW
    float *weight_gradients;

    // numNeurons x 1
    // UNUSED AS OF NOW
    float *bias_gradients;

    void printLayer();

    CudaNNLayer(int inputSize, int outputSize, float *values, float *na_values,
                float *value_gradients, float *weights, float *weight_gradients,
                float *biases, float *bias_gradients, NeuronActivation activationFunction = Identity);

    void forwardProp(float *input);

    // input is the graditents of the neuron values in the next layer
    // outputSize * BATCH_SIZE
    // (1) calculate and store val_gradients,
    void backwardProp(float *input);

    // update weights
    // (1) calcuate weight,bias gradients
    // (2) update weights and biases
    void updateWeights(float *input_val_gradients, float *input_fwd_prop_input_vals);

    // We can pipeline the kernels for backwardProp and updateWeights
};

#endif