#include <iostream>
#include "nnData.cuh"
#include "nn.cuh"
#include "activations.cuh"

using namespace std;
int main()
{
    NeuralNetData nnData("datasets/abelone.train");
    int numNeurons = 500;
    printf("Num hidden neurons: %dx3\n", numNeurons);

    vector<int> internalLayers = {numNeurons, numNeurons, numNeurons};
    vector<NeuronActivation> activations = {Sigmoid, Sigmoid, Sigmoid, Sigmoid};
    NeuralNet nn(nnData.inputSize, internalLayers, activations, nnData.outputSize);
    // cout << "\n\nNN training start\n\n";
    nn.run(nnData, 100);
};