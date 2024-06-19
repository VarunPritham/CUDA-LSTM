#include "LSTM.cuh"
#include <cmath>
#include <iostream>
#include "vector"
#include <numeric>
#include <array>
#include <ctime>
using namespace std;

// CUDA Kernels
__global__ void sigmoidKernel(double* x, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        x[index] = 1.0 / (1.0 + exp(-x[index]));
    }
}

__global__ void tanhKernel(double* x, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        x[index] = tanh(x[index]);
    }
}

__global__ void innerProductKernel(double* weights, double* inputs, double* outputs, size_t inputSize, int numCells, int timeSteps) {

    int cellId = blockIdx.x * blockDim.x + threadIdx.x;
    int timeStep = blockIdx.y * blockDim.y + threadIdx.y;

    if (cellId < numCells && timeStep < timeSteps) {
        double sum = 0.0;
        for (int i = 0; i < inputSize; ++i) {
            sum += weights[cellId * inputSize + i] * inputs[timeStep * inputSize + i];
        }
        outputs[cellId * timeSteps + timeStep] = sum;
    }
}

void applySigmoid(double* d_x, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sigmoidKernel<<<numBlocks, blockSize>>>(d_x, size);
    cudaDeviceSynchronize();
}

void applyTanh(double* d_x, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    tanhKernel<<<numBlocks, blockSize>>>(d_x, size);
    cudaDeviceSynchronize();
}

__device__ double derivative_of_sigmoid(double x) {
    double sigmoid = 1.0 / (1.0 + exp(-x));
    return sigmoid * (1.0 - sigmoid);
}


// CUDA function to call this kernel
__device__ double derivative_of_tanh(double x) {
    double tanh_val = tanh(x);
    return 1.0 - tanh_val * tanh_val;
}

__global__ void computeInitialErrorKernel(double* outputErrors, const double* targetOutput, int numCells, int timeSteps) {
    int cellId = blockIdx.x * blockDim.x + threadIdx.x;
    int timeStep = blockIdx.y * blockDim.y + threadIdx.y;

    if (cellId < numCells && timeStep < timeSteps) {
        int index = cellId * timeSteps + timeStep;
        outputErrors[index] = targetOutput[index] - outputErrors[index]; // Assuming targetOutput is the correct size
    }
}


__global__ void updateCellStateAndOutput(double* aGate, double* iGate, double* fGate, double* oGate,
                                         double* cellState, double* cellOutput, int numCells, int timeSteps) {
    int cellId = blockIdx.x * blockDim.x + threadIdx.x;

    if (cellId < numCells) {
        double prevState = 0.0; // Assuming the first state is zero
        for (int t = 0; t < timeSteps; ++t) {
            // Compute the new cell state
            double a = aGate[cellId * timeSteps + t];
            double i = iGate[cellId * timeSteps + t];
            double f = fGate[cellId * timeSteps + t];
            double o = oGate[cellId * timeSteps + t];

            double newState = a * i + f * prevState;
            cellState[cellId * timeSteps + t] = newState;

            // Compute the cell output
            cellOutput[cellId * timeSteps + t] = tanh(newState) * o;

            prevState = newState; // Update prevState for the next timestep
        }
    }
}

__global__ void backwardKernel(
        double* inputGateOutputs, double* forgetGateOutputs, double* cellStateOutputs, double* outputGateOutputs,
        double* outputErrors, double* inputGateGradients, double* forgetGateGradients, double* cellStateGradients,
        double* outputGateGradients, int numCells, int timeSteps) {

    int cellId = blockIdx.x * blockDim.x + threadIdx.x;
    int timeStep = blockIdx.y * blockDim.y + threadIdx.y;

    if (cellId < numCells && timeStep < timeSteps) {
        double error = outputErrors[cellId * timeSteps + timeStep];
        double inputGateOutput = inputGateOutputs[cellId * timeSteps + timeStep];
        double forgetGateOutput = forgetGateOutputs[cellId * timeSteps + timeStep];
        double cellStateOutput = cellStateOutputs[cellId * timeSteps + timeStep];
        double outputGateOutput = outputGateOutputs[cellId * timeSteps + timeStep];

        // Compute derivatives of the activation functions for each gate
        double dInputGate = derivative_of_sigmoid(inputGateOutput);
        double dForgetGate = derivative_of_sigmoid(forgetGateOutput);
        double dOutputGate = derivative_of_sigmoid(outputGateOutput);
        double dCellState = derivative_of_tanh(cellStateOutput);

        // Compute the gradients for cell state and gates
        double cellStateGradient = error * dOutputGate * dCellState;
        double inputGateGradient = cellStateGradient * tanh(cellStateOutput) * dInputGate;
        double prevCellStateOutput = (timeStep > 0) ? cellStateOutputs[cellId * timeSteps + timeStep - 1] : 0.0;
        double forgetGateGradient = cellStateGradient * prevCellStateOutput * dForgetGate;
        double outputGateGradient = error * tanh(cellStateOutput) * dOutputGate;

        // Update gradients arrays
        cellStateGradients[cellId * timeSteps + timeStep] = cellStateGradient;
        inputGateGradients[cellId * timeSteps + timeStep] = inputGateGradient;
        forgetGateGradients[cellId * timeSteps + timeStep] = forgetGateGradient;
        outputGateGradients[cellId * timeSteps + timeStep] = outputGateGradient;

        // Propagate error to previous time step
        if (timeStep > 0) {
            // Error for cell state
            double cellStateError = error * dOutputGate * derivative_of_tanh(cellStateOutput);

            // Propagate error through the forget gate to the previous cell state
            double prevForgetGateOutput = forgetGateOutputs[cellId * timeSteps + timeStep - 1];
            double forgetGateError = cellStateError * prevForgetGateOutput;

            // Propagate error through the input gate to the previous input
            double prevInputGateOutput = inputGateOutputs[cellId * timeSteps + timeStep - 1];
            double inputGateError = cellStateError * prevInputGateOutput * derivative_of_sigmoid(prevInputGateOutput);

            // Combine errors for the previous cell state
            double propagatedError = forgetGateError + inputGateError;

            // Update the error for the previous time step
            outputErrors[cellId * timeSteps + timeStep - 1] += propagatedError;
        }
    }
}

__global__ void updateWeightsKernel(double* weights, double* gradients, int size, double learningRate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // Basic gradient descent update rule
        weights[index] -= learningRate * gradients[index];
    }
}

// A host function to call the kernel
void updateWeights(double* d_weights, double* d_gradients, int size, double learningRate) {
    int blockSize = 256;  // You can tune this size
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateWeightsKernel<<<numBlocks, blockSize>>>(d_weights, d_gradients, size, learningRate);

    // Error checking
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "updateWeightsKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching updateWeightsKernel!" << std::endl;
    }
}

void updateAllWeights(double* d_i_weights, double* d_i_gradients,
                      double* d_f_weights, double* d_f_gradients,
                      double* d_c_weights, double* d_c_gradients,
                      double* d_o_weights, double* d_o_gradients,
                      int weightsSize, double learningRate) {

    int weightsBlockSize = 256; // Tune this size
    int weightsNumBlocks = (weightsSize + weightsBlockSize - 1) / weightsBlockSize;

    // Update weights for each gate
    updateWeightsKernel<<<weightsNumBlocks, weightsBlockSize>>>(d_i_weights, d_i_gradients, weightsSize, learningRate);
    updateWeightsKernel<<<weightsNumBlocks, weightsBlockSize>>>(d_f_weights, d_f_gradients, weightsSize, learningRate);
    updateWeightsKernel<<<weightsNumBlocks, weightsBlockSize>>>(d_c_weights, d_c_gradients, weightsSize, learningRate);
    updateWeightsKernel<<<weightsNumBlocks, weightsBlockSize>>>(d_o_weights, d_o_gradients, weightsSize, learningRate);


    // Error checking (for the last kernel call)
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "updateWeightsKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching updateWeightsKernel!" << std::endl;
    }
}



LSTM::LSTM(int memCells, int inputVecSize,int timesteps) {

    n_cells = memCells;
    input_dim = inputVecSize;
    time_steps = timesteps; // Initialize with default value, to be set during training


//    int x = initWeights();


    // Example of memory allocation for weights and biases

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "before cudaMalloc in constructor: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error...
    }


    cudaMalloc(&d_c_Weight, n_cells * input_dim * sizeof(double));

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "after cudaMalloc in constructor\n", cudaGetErrorString(cudaStatus));
        // Handle error...
    }
    cudaMalloc(&d_i_Weight, n_cells * input_dim * sizeof(double));
    cudaMalloc(&d_f_Weight, n_cells * input_dim * sizeof(double));
    cudaMalloc(&d_o_Weight, n_cells * input_dim * sizeof(double));

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "after cudaMalloc 1 in constructor\n", cudaGetErrorString(cudaStatus));
        // Handle error...
    }

    cudaMalloc(&d_cell_output, n_cells * timesteps * sizeof(double));
    cudaMalloc(&d_cell_state, n_cells * timesteps * sizeof(double));
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "after cudaMalloc 2 in constructor\n", cudaGetErrorString(cudaStatus));
        // Handle error...
    }

    cudaMalloc(&d_c_output, n_cells * input_dim * sizeof(double));
    cudaMalloc(&d_i_output, n_cells * input_dim * sizeof(double));
    cudaMalloc(&d_f_output, n_cells * input_dim * sizeof(double));
    cudaMalloc(&d_o_output, n_cells * input_dim * sizeof(double));

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "after cudaMalloc 3 in constructor\n", cudaGetErrorString(cudaStatus));
        // Handle error...
    }

    cout << "Partially done cuda mallocs " <<endl;

//    cudaMemcpy(d_c_Weight, h_c_Weight, n_cells * input_dim * sizeof(double), cudaMemcpyHostToDevice);
//
//
//    cudaMemcpy(d_f_Weight, h_f_Weight, n_cells * input_dim * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_i_Weight, h_i_Weight, n_cells * input_dim * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_o_Weight, h_o_Weight, n_cells * input_dim * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_cell_output, h_c_Weight, n_cells * timesteps * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_cell_state, h_f_Weight, n_cells * timesteps * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_c_grad, h_c_grad, n_cells * input_dim * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_i_grad, h_i_grad, n_cells * input_dim * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_f_grad, h_f_grad, n_cells * input_dim * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_o_grad, h_o_grad, n_cells * input_dim * sizeof(double), cudaMemcpyHostToDevice);


// Allocate memory on the GPU for each array
    cudaMalloc((void**)&d_outputErrors,  n_cells * input_dim * sizeof(double));
    cudaMalloc((void**)&d_i_grad, n_cells * input_dim * sizeof(double));
    cudaMalloc((void**)&d_f_grad, n_cells * input_dim * sizeof(double));
    cudaMalloc((void**)&d_c_grad, n_cells * input_dim * sizeof(double));
    cudaMalloc((void**)&d_o_grad, n_cells * input_dim * sizeof(double));

}
LSTM::~LSTM() {
    cudaFree(d_c_Weight);
    cudaFree(d_i_Weight);
    cudaFree(d_f_Weight);
    cudaFree(d_o_Weight);

    cudaFree(d_cell_output);
    cudaFree(d_cell_state);
    cudaFree(d_c_output);
    cudaFree(d_i_output);
    cudaFree(d_f_output);
    cudaFree(d_o_output);

    cudaFree(d_i_grad);
    cudaFree(d_f_grad);
    cudaFree(d_c_grad);
    cudaFree(d_o_grad);
    cudaFree(d_outputErrors);
    // Free other allocated memory as well
}


int LSTM::forward(double* d_input, int timeSteps, size_t inputSize) {

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error before forward : " << cudaGetErrorString(status) << std::endl;
    }

    time_steps = timeSteps;
    // Kernel launch parameters
    dim3 blockSize(16, 16); // Example block size, can be tuned
    dim3 numBlocks((n_cells + blockSize.x - 1) / blockSize.x, (timeSteps + blockSize.y - 1) / blockSize.y);


    // Launch kernels for computing inner products
    innerProductKernel<<<numBlocks, blockSize>>>(d_c_Weight, d_input, d_c_output, inputSize, n_cells, timeSteps);
    innerProductKernel<<<numBlocks, blockSize>>>(d_i_Weight, d_input, d_i_output, inputSize, n_cells, timeSteps);
    innerProductKernel<<<numBlocks, blockSize>>>(d_f_Weight, d_input, d_f_output, inputSize, n_cells, timeSteps);
    innerProductKernel<<<numBlocks, blockSize>>>(d_o_Weight, d_input, d_o_output, inputSize, n_cells, timeSteps);

    // Synchronize after kernel execution
    cudaDeviceSynchronize();
     status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error in kernel launch Inner Product: " << cudaGetErrorString(status) << std::endl;
    }

    // Apply activation functions
    applyTanh(d_c_output, n_cells * timeSteps); // Assuming applyTanh handles kernel launch and synchronization
    applySigmoid(d_i_output, n_cells * timeSteps); // Similarly for i, f, and o gates
    applySigmoid(d_f_output, n_cells * timeSteps); // Similarly for i, f, and o gates
    applySigmoid(d_o_output, n_cells * timeSteps); // Similarly for i, f, and o gates

    dim3 blockSize_2(256); // Adjusted for 1D grid
    dim3 numBlocks_2((n_cells + blockSize.x - 1) / blockSize.x);
    updateCellStateAndOutput<<<numBlocks_2, blockSize_2>>>(d_c_output, d_i_output, d_f_output, d_o_output, d_cell_state, d_cell_output, n_cells, timeSteps);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error in kernel launch updateCellStateAndOutput: " << cudaGetErrorString(status) << std::endl;
    }


    return 0;
}

int LSTM:: backward(double* output, int timeSteps) {

    dim3 blockSize1(256); // 256 threads per block
    int totalElements = n_cells * timeSteps;
    dim3 gridSize1((totalElements + blockSize1.x - 1) / blockSize1.x); // Ensures enough blocks to cover all elements

    computeInitialErrorKernel<<<gridSize1, blockSize1>>>(d_outputErrors, output, n_cells, timeSteps);


// Check for errors after kernel execution
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "computeInitialErrorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error...
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeInitialErrorKernel!\n", cudaStatus);
        // Handle error...
    }

    // Number of cells in the LSTM layer (assumed to be a member of LSTMNet)
    int numCells = this->n_cells;


    // Define block and grid sizes for CUDA kernel
    dim3 blockSize(256); // This is an example, you may need to tune this
    dim3 gridSize((numCells + blockSize.x - 1) / blockSize.x, (timeSteps + blockSize.y - 1) / blockSize.y);

// Launch the backward kernel
    backwardKernel<<<gridSize, blockSize>>>(
            d_i_output,
            d_f_output,
            d_c_output,
            d_o_output,
            d_outputErrors,
            d_i_grad,
            d_f_grad,
            d_c_grad,
            d_o_grad,
            n_cells,
            timeSteps
    );


    // Check for errors in kernel launch and after execution
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "backwardKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching backwardKernel!\n", cudaStatus);
        return -1;
    }

    return 0;
}


int LSTM::train(vector<double>* input, vector<double> output, int trainDataSize, int timeSteps, float learningRate, int iterations, int inputVecSize) {


//    cout << "Started Training" << endl;

    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < trainDataSize; i += timeSteps) {
            // Calculate the size of the current batch
            int batchSize = min(timeSteps, trainDataSize - i);

            // Allocate memory for the input and output batches on the device
            double* d_input;
            double* d_labels;

            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "before cudaMalloc failed in Train: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error...
            }

//            printf("batch size = %d",batchSize);
            cudaMalloc(&d_input, batchSize * sizeof(double));
            cudaMalloc(&d_labels, batchSize * sizeof(double));
             cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed in Train: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error...
            }

            // Copy input and output batch data from host to device
            cudaMemcpy(d_input, input->data() + i, batchSize * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_labels, output.data() + i, batchSize * sizeof(double), cudaMemcpyHostToDevice);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed in Train", cudaGetErrorString(cudaStatus));
                // Handle error...
            }

            // Forward pass
            forward(d_input, timeSteps, inputVecSize*batchSize);


            // Compute initial error and perform backward pass
            backward(d_labels, timeSteps);

            updateAllWeights(d_i_Weight, d_i_grad,
                             d_f_Weight, d_f_grad,
                             d_c_Weight, d_c_grad,
                             d_o_Weight, d_o_grad,
                             n_cells * input_dim, learningRate);

            // Free batch memory
            cudaFree(d_input);
            cudaFree(d_labels);
        }
        // Optional: adjust learning rate or other hyperparameters
    }

    // Free device memory allocated for LSTM network
    // freeDeviceMemory();

    return 0;
}

int LSTM::initWeights() {

    h_c_Weight = new double [n_cells * input_dim];
    h_i_Weight = new double[n_cells * input_dim];
    h_f_Weight = new double[n_cells * input_dim];
    h_o_Weight = new double[n_cells * input_dim];


//    memCellOutArr = new std::vector<double>[n_cells];
    h_cell_state = new double[n_cells * time_steps];
    h_cell_output = new double[n_cells * time_steps];

//    h_c_output = new double[n_cells * input_dim];
//    h_i_output = new double[n_cells * input_dim];
//    h_f_output = new double[n_cells * input_dim];
//    h_o_output = new double[n_cells * input_dim];
//
//    h_c_grad = new double[n_cells * input_dim];
//    h_i_grad = new double[n_cells * input_dim];
//    h_f_grad = new double[n_cells * input_dim];
//    h_o_grad = new double[n_cells * input_dim];

    double w, max, min;
    min = -0.01;
    max = 0.01;

    for(int i = 0; i < n_cells * input_dim; i++) {

        h_c_Weight[i] = (double)(rand()) / RAND_MAX;
        h_i_Weight[i] = (double)(rand()) / RAND_MAX;
        h_f_Weight[i] = (double)(rand()) / RAND_MAX;
        h_o_Weight[i] = (double)(rand()) / RAND_MAX;

        h_cell_state[i] = 0;
        h_cell_output[i] = 0;



    }
    return 0;
}

double LSTM::   predict(std::vector<double> * input) {

    size_t size_d = input->size() * input_dim;
    double *d_input;
    cudaMalloc(&d_input, size_d * sizeof(double));
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc 1 failed in predict: %s\n", cudaGetErrorString(cudaStatus));
    }

//    // Copy data from host to device
    cudaMemcpy(d_input, input->data(), size_d * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy 1 failed in predict: %s\n", cudaGetErrorString(cudaStatus));
    }

    forward(d_input, 1, size_d);

    h_cell_output = new double[n_cells * time_steps];
    cudaMalloc(&d_cell_output, n_cells * time_steps * sizeof(double));

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc 2 failed in predict: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaMemcpy(h_cell_output, d_cell_output, n_cells * time_steps * sizeof(double), cudaMemcpyDeviceToHost);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy 2 failed in predict: %s\n", cudaGetErrorString(cudaStatus));
    }
    double result = 0;
    for (int i = 0; i < n_cells; i++) {
        result += h_cell_output[time_steps - 1];
    }

    return result;

}