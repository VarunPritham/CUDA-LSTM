#include <vector>

#include "LSTM.cuh"
#include "DataProcessor.h"
#include "FileProcessor.h"


int multivarPredicts() {
    int memCells = 10; // number of memory cells
    int inputVecSize = 5; // input vector size
    int trainDataSize = 5000; // train data size
    int timeSteps = 10; // data points used for one forward step
    float learningRate = 0.0001;
    int iterations = 100; // training iterations with training data
    int lines = 5000;

    DataProcessor * dataproc;
    dataproc = new DataProcessor();
    FileProcessor * fileProc;
    fileProc = new FileProcessor();

    int colIndxs[] = {0,0,1,1,1,1,1};
    int targetValCol = 7;

    cout << "Step 1 " << endl;

    std::vector<double> * timeSeries;
    timeSeries = fileProc->readMultivariate("./datatraining.txt",lines,inputVecSize,colIndxs,targetValCol);

    // Creating the input vector Array
    std::vector<double> * input;
    input = new std::vector<double>[trainDataSize];
    for (int i = 0; i < trainDataSize; i++) {
        input[i] = dataproc->process(timeSeries[i],0);
    }

    cout << "Step 2 " << endl;

    // Creating the target vector using the time series
    std::vector<double>::const_iterator first = timeSeries[lines].begin();
    std::vector<double>::const_iterator last = timeSeries[lines].begin() + trainDataSize;
    std::vector<double> targetVector(first, last);
    for (std::vector<double>::iterator it = targetVector.begin(); it != targetVector.end(); ++it) {
        if (*it == 0) *it = -1;
    }

    int size = std::distance(first, last);
    double* labels = new double[size]; // Allocate array

    int index = 0;
    for (auto it = first; it != last; ++it) {
        labels[index++] = *it; // Copy elements
    }


    size = input->size();
    double* data = new double[size]; // Allocate array
    for (int i = 0; i < size; ++i) {
        data[i] = (*input)[i]; // Copy elements
    }
    // Training the LSTM net

    cout << "Calling now" << endl;
    LSTM* lstm = new LSTM(memCells,inputVecSize,timeSteps);
    lstm->train(input, targetVector, size, timeSteps, learningRate, iterations,inputVecSize);


    // Predictions
    int predictions = 2000; // prediction points
    lines = 2000; // lines read from the files

    timeSeries = fileProc->readMultivariate("./datatest.txt",lines,inputVecSize,colIndxs,targetValCol);

    cout << "Complted reading timeseries " << timeSeries->size() << endl;

    input = new std::vector<double>[1];
    double result;
    double min = 0, max = 0;
    std::vector<double> resultVec;

    cout << "File Read was complete" <<endl;
    for (int i = 0; i < predictions; i++) {
        input[0] = dataproc->process(timeSeries[i],0);
        result = lstm->predict(input);
        resultVec.push_back(result);

        if (i == 0){
            min = result;
            max = result;
        } else {

            if (min > result) min = result;
            if (max < result) max = result;
        }
    }

    cout << "Prediction completed" <<endl;
    std::cout<<"min: "<<min<<std::endl;
    std::cout<<"max: "<<max<<std::endl;

    double line = 0; //(min + max)/2;
    std::cout<<"margin: "<<line<<std::endl<<std::endl;


    int occu = 0, notoccu = 0;

    int corr = 0;
    int incorr = 0;

    // Open the file to write the time series predictions
    std::ofstream out_file;
    std::ofstream out_file2;
    out_file.open("./multiResults.txt",std::ofstream::out | std::ofstream::trunc);
    out_file2.open("./multiTargets.txt",std::ofstream::out | std::ofstream::trunc);

    cout << "File reads completed again "<< resultVec.size() << "  " << timeSeries->size() << endl;
    for (int i = 0; i < predictions; i++) {
        out_file<<timeSeries[lines].at(i)<<","<<resultVec.at(i)<<"\n";
        out_file2<<timeSeries[lines].at(i)<<",";
        if (timeSeries[lines].at(i) == 1) {
            out_file2<<1<<"\n";
        } else out_file2<<-1<<"\n";

        if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 1)) {
            corr++;
            occu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 0)) {
            corr++;
            notoccu++;
        } else if ( (resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 1)) {
            incorr++;
            occu++;
        } else if ( (resultVec.at(i) > line) && (timeSeries[lines].at(i) == 0)) {
            incorr++;
            notoccu++;
        }
    }

    std::cout<<std::endl;

    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Data "<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Occupied: "<<occu<<std::endl;
    std::cout<<"NotOccupied: "<<notoccu<<std::endl<<std::endl;
    std::cout<<"----------------------"<<std::endl;
    std::cout<<"Correct predictions: "<<corr<<std::endl;
    std::cout<<"Incorrect predictions: "<<incorr<<std::endl<<std::endl;

    std::cout<<std::endl<<"Accuracy: "<<(corr/(double)predictions)*100<<"%"<<std::endl<<std::endl;

    return 0;
}


int main() {
    multivarPredicts();
}
