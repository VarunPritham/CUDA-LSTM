#ifndef LSTM_H
#define LSTM_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "iostream"
#include "vector"
#include <algorithm>


using namespace std;

class LSTM {
public:
    LSTM(int memCells, int inputVecSize,int timesteps);
    virtual ~LSTM();
    int forward(double* d_input, int timeSteps, size_t inputSize);
    int backward(double* output, int timeSteps);
    double * d_cell_output;
    double * h_cell_output;
    int train(vector<double>* input, vector<double> output, int trainDataSize, int timeSteps, float learningRate, int iterations, int inputVecSize);
    double predict(std::vector<double> * input);
    int initWeights();
private:

    // CUDA device pointers for weights, biases, states, etc.
    double * d_c_Weight;
    double * d_i_Weight;
    double * d_f_Weight;
    double * d_o_Weight;

    double * d_cell_state;
    double * d_outputErrors;
    double * d_c_output;
    double * d_i_output;
    double * d_f_output;
    double * d_o_output;

    double * d_i_grad;
    double * d_f_grad;
    double * d_c_grad;
    double * d_o_grad;

    double * h_c_Weight;
    double * h_i_Weight;
    double * h_f_Weight;
    double * h_o_Weight;

    double * h_c_bias;
    double * h_i_bias;
    double * h_f_bias;
    double * h_o_bias;

    double * h_cell_state;

    double * h_c_grad;
    double * h_i_grad;
    double * h_f_grad;
    double * h_o_grad;

    // gate output value arrays
    double * h_c_output;
    double * h_i_output;
    double * h_f_output;
    double * h_o_output;

    // Host variables
    int n_cells;
    int input_dim;
    int time_steps;


};

#endif
