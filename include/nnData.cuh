#ifndef NN_DATA_H
#define NN_DATA_H
#include <string>

class NeuralNetData
{
public:
    float *host_input;
    float *host_output;
    int numData, inputSize, outputSize;

    int cursor;

    NeuralNetData(std::string filePath);

    void loadBatch(float *input, float *output, int batchSize);
};
#endif