
#include "layer.cuh"
#include "nnData.cuh"
#include <vector>

class NeuralNet
{
public:
    NeuralNet(int inputSize, std::vector<int> internalLayerSizes, std::vector<NeuronActivation> activations, int outputSize);
    void run(NeuralNetData data, int numEpochs = 1);
    std::vector<CudaNNLayer> NNlayers;
};